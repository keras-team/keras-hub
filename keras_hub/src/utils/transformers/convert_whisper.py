import re

import numpy as np

from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = WhisperBackbone

# Timestamp tokens: `<|0.00|>`, `<|0.02|>`, ..., `<|30.00|>`.
_TIMESTAMP_TOKEN_RE = re.compile(r"^<\|\d+\.\d+\|>$")
# The control tokens that `WhisperTokenizer` exposes as `special_tokens`.
_CONTROL_TOKENS = (
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nocaptions|>",
    "<|nospeech|>",
    "<|notimestamps|>",
)


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["encoder_layers"],
        "num_heads": transformers_config["encoder_attention_heads"],
        "hidden_dim": transformers_config["d_model"],
        "intermediate_dim": transformers_config["encoder_ffn_dim"],
        "decoder_num_layers": transformers_config["decoder_layers"],
        "decoder_num_heads": transformers_config["decoder_attention_heads"],
        "decoder_intermediate_dim": transformers_config["decoder_ffn_dim"],
        "num_mels": transformers_config["num_mel_bins"],
        "dropout": transformers_config["dropout"],
        # Keras halves this internally for the encoder position embedding,
        # so we double the HF value to match `embed_positions` length.
        "max_encoder_sequence_length": (
            2 * transformers_config["max_source_positions"]
        ),
        "max_decoder_sequence_length": transformers_config[
            "max_target_positions"
        ],
    }


def _qkv_kernel_hook(hf_tensor, keras_shape):
    return np.reshape(np.transpose(hf_tensor), keras_shape)


def _qkv_bias_hook(hf_tensor, keras_shape):
    return np.reshape(hf_tensor, keras_shape)


def _fc_kernel_hook(hf_tensor, _):
    return np.transpose(hf_tensor, axes=(1, 0))


def _conv_kernel_hook(hf_tensor, _):
    # HF Conv1d weight is (out, in, k); Keras Conv1D kernel is (k, in, out).
    return np.transpose(hf_tensor, axes=(2, 1, 0))


def _port_attention_block(loader, attn_layer, hf_prefix):
    # Whisper attention has bias on q/v/out but NOT on k.
    loader.port_weight(
        keras_variable=attn_layer._query_dense.kernel,
        hf_weight_key=f"{hf_prefix}.q_proj.weight",
        hook_fn=_qkv_kernel_hook,
    )
    loader.port_weight(
        keras_variable=attn_layer._query_dense.bias,
        hf_weight_key=f"{hf_prefix}.q_proj.bias",
        hook_fn=_qkv_bias_hook,
    )
    loader.port_weight(
        keras_variable=attn_layer._key_dense.kernel,
        hf_weight_key=f"{hf_prefix}.k_proj.weight",
        hook_fn=_qkv_kernel_hook,
    )
    loader.port_weight(
        keras_variable=attn_layer._value_dense.kernel,
        hf_weight_key=f"{hf_prefix}.v_proj.weight",
        hook_fn=_qkv_kernel_hook,
    )
    loader.port_weight(
        keras_variable=attn_layer._value_dense.bias,
        hf_weight_key=f"{hf_prefix}.v_proj.bias",
        hook_fn=_qkv_bias_hook,
    )
    loader.port_weight(
        keras_variable=attn_layer._output_dense.kernel,
        hf_weight_key=f"{hf_prefix}.out_proj.weight",
        hook_fn=_qkv_kernel_hook,
    )
    loader.port_weight(
        keras_variable=attn_layer._output_dense.bias,
        hf_weight_key=f"{hf_prefix}.out_proj.bias",
    )


def _port_layer_norm(loader, ln_layer, hf_prefix):
    loader.port_weight(
        keras_variable=ln_layer.gamma, hf_weight_key=f"{hf_prefix}.weight"
    )
    loader.port_weight(
        keras_variable=ln_layer.beta, hf_weight_key=f"{hf_prefix}.bias"
    )


def _port_feedforward(loader, layer, hf_prefix):
    loader.port_weight(
        keras_variable=layer._feedforward_intermediate_dense.kernel,
        hf_weight_key=f"{hf_prefix}.fc1.weight",
        hook_fn=_fc_kernel_hook,
    )
    loader.port_weight(
        keras_variable=layer._feedforward_intermediate_dense.bias,
        hf_weight_key=f"{hf_prefix}.fc1.bias",
    )
    loader.port_weight(
        keras_variable=layer._feedforward_output_dense.kernel,
        hf_weight_key=f"{hf_prefix}.fc2.weight",
        hook_fn=_fc_kernel_hook,
    )
    loader.port_weight(
        keras_variable=layer._feedforward_output_dense.bias,
        hf_weight_key=f"{hf_prefix}.fc2.bias",
    )


def convert_weights(backbone, loader, transformers_config):
    # Encoder convolutional stem.
    loader.port_weight(
        keras_variable=backbone.encoder_conv_layer_1.kernel,
        hf_weight_key="encoder.conv1.weight",
        hook_fn=_conv_kernel_hook,
    )
    loader.port_weight(
        keras_variable=backbone.encoder_conv_layer_1.bias,
        hf_weight_key="encoder.conv1.bias",
    )
    loader.port_weight(
        keras_variable=backbone.encoder_conv_layer_2.kernel,
        hf_weight_key="encoder.conv2.weight",
        hook_fn=_conv_kernel_hook,
    )
    loader.port_weight(
        keras_variable=backbone.encoder_conv_layer_2.bias,
        hf_weight_key="encoder.conv2.bias",
    )

    # Encoder sinusoidal positional embedding (non-trainable, but the keras
    # init is TruncatedNormal so we must port the HF weight to recover the
    # sinusoids).
    loader.port_weight(
        keras_variable=backbone.encoder_position_embedding.position_embeddings,
        hf_weight_key="encoder.embed_positions.weight",
    )

    # Encoder transformer layers.
    for i in range(backbone.num_layers):
        layer = backbone.encoder_transformer_layers[i]
        prefix = f"encoder.layers.{i}"
        _port_layer_norm(
            loader,
            layer._self_attention_layer_norm,
            f"{prefix}.self_attn_layer_norm",
        )
        _port_attention_block(
            loader, layer._self_attention_layer, f"{prefix}.self_attn"
        )
        _port_layer_norm(
            loader,
            layer._feedforward_layer_norm,
            f"{prefix}.final_layer_norm",
        )
        _port_feedforward(loader, layer, prefix)

    _port_layer_norm(loader, backbone.encoder_layer_norm, "encoder.layer_norm")

    # Decoder embeddings. The token embedding is tied with `proj_out` on the
    # HF side; ReversibleEmbedding handles the reverse projection internally,
    # so porting `embed_tokens.weight` is enough.
    loader.port_weight(
        keras_variable=backbone.decoder_embeddings.token_embedding.embeddings,
        hf_weight_key="decoder.embed_tokens.weight",
    )
    loader.port_weight(
        keras_variable=(
            backbone.decoder_embeddings.position_embedding.position_embeddings
        ),
        hf_weight_key="decoder.embed_positions.weight",
    )

    # Decoder transformer layers.
    for i in range(backbone.decoder_num_layers):
        layer = backbone.decoder_transformer_layers[i]
        prefix = f"decoder.layers.{i}"
        _port_layer_norm(
            loader,
            layer._self_attention_layer_norm,
            f"{prefix}.self_attn_layer_norm",
        )
        _port_attention_block(
            loader, layer._self_attention_layer, f"{prefix}.self_attn"
        )
        _port_layer_norm(
            loader,
            layer._cross_attention_layer_norm,
            f"{prefix}.encoder_attn_layer_norm",
        )
        _port_attention_block(
            loader, layer._cross_attention_layer, f"{prefix}.encoder_attn"
        )
        _port_layer_norm(
            loader,
            layer._feedforward_layer_norm,
            f"{prefix}.final_layer_norm",
        )
        _port_feedforward(loader, layer, prefix)

    _port_layer_norm(loader, backbone.decoder_layer_norm, "decoder.layer_norm")


def _load_generation_config(preset):
    if check_file_exists(preset, "generation_config.json"):
        return load_json(preset, "generation_config.json")
    return {}


def _is_multilingual(generation_config, vocabulary_size):
    if "is_multilingual" in generation_config:
        return bool(generation_config["is_multilingual"])
    # Same fallback as Hugging Face: English-only checkpoints have a 51864
    # token vocabulary and multilingual checkpoints 51865 or more.
    return vocabulary_size >= 51865


def _load_vocabulary_and_merges(preset):
    """Return the full vocabulary (with added tokens) and the merges."""
    if check_file_exists(preset, "tokenizer.json"):
        # `transformers` v5 `save_pretrained` writes only `tokenizer.json`.
        tokenizer_config = load_json(preset, "tokenizer.json")
        vocabulary = dict(tokenizer_config["model"]["vocab"])
        merges = [
            " ".join(merge) if isinstance(merge, (list, tuple)) else merge
            for merge in tokenizer_config["model"]["merges"]
        ]
        for added_token in tokenizer_config.get("added_tokens", []):
            vocabulary[added_token["content"]] = added_token["id"]
        return vocabulary, merges

    vocabulary = load_json(preset, "vocab.json")
    with open(get_file(preset, "merges.txt"), encoding="utf-8") as f:
        merges = [
            line.rstrip("\n")
            for line in f
            if line.strip() and not line.startswith("#version")
        ]
    if check_file_exists(preset, "added_tokens.json"):
        vocabulary.update(load_json(preset, "added_tokens.json"))
    return vocabulary, merges


def _language_tokens(vocabulary, generation_config):
    """The language tokens, from `lang_to_id` or from the vocabulary layout."""
    if generation_config.get("lang_to_id"):
        return {
            token: int(token_id)
            for token, token_id in generation_config["lang_to_id"].items()
        }
    # Whisper places the language tokens right after `<|startoftranscript|>`
    # and right before `<|translate|>`.
    start = vocabulary["<|startoftranscript|>"]
    end = vocabulary["<|translate|>"]
    return {
        token: token_id
        for token, token_id in vocabulary.items()
        if start < token_id < end
    }


def convert_tokenizer(cls, preset, **kwargs):
    vocabulary, merges = _load_vocabulary_and_merges(preset)
    transformers_config = load_json(preset, "config.json")
    generation_config = _load_generation_config(preset)

    # Only the named control tokens are special. The 1,501 timestamp tokens
    # stay in the vocabulary but are not made unsplittable, which keeps BPE
    # splitting fast.
    special_tokens = {
        token: vocabulary[token]
        for token in _CONTROL_TOKENS
        if token in vocabulary
    }
    language_tokens = None
    if _is_multilingual(generation_config, transformers_config["vocab_size"]):
        language_tokens = _language_tokens(vocabulary, generation_config)
    return cls(
        vocabulary=vocabulary,
        merges=merges,
        special_tokens=special_tokens,
        language_tokens=language_tokens,
        **kwargs,
    )


def load_audio_converter_config(preset, transformers_config):
    """Return WhisperAudioConverter kwargs from HF preprocessor config."""
    if check_file_exists(preset, "preprocessor_config.json"):
        feature_extractor = load_json(preset, "preprocessor_config.json")
    else:
        # `transformers` v5 `WhisperProcessor.save_pretrained` nests the
        # feature extractor config in `processor_config.json`.
        feature_extractor = load_json(preset, "processor_config.json")[
            "feature_extractor"
        ]
    return {
        "num_mels": feature_extractor["feature_size"],
        "num_fft_bins": feature_extractor["n_fft"],
        "stride": feature_extractor["hop_length"],
        "sampling_rate": feature_extractor["sampling_rate"],
        "max_audio_length": feature_extractor["chunk_length"],
    }


def load_preprocessor_config(preset, transformers_config):
    """Return `WhisperAudioToTextPreprocessor` kwargs.

    `decoder_sequence_length` follows `max_target_positions`, and the language
    and task come from `forced_decoder_ids` in `generation_config.json` when
    it pins them.
    """
    config = {
        "decoder_sequence_length": transformers_config["max_target_positions"],
    }
    generation_config = _load_generation_config(preset)
    if not _is_multilingual(
        generation_config, transformers_config["vocab_size"]
    ):
        return config
    id_to_language = {
        token_id: token
        for token, token_id in generation_config.get("lang_to_id", {}).items()
    }
    id_to_task = {
        token_id: task
        for task, token_id in generation_config.get("task_to_id", {}).items()
    }
    forced_decoder_ids = generation_config.get("forced_decoder_ids") or []
    for _, token_id in forced_decoder_ids:
        if token_id in id_to_language:
            config["language"] = id_to_language[token_id]
        elif token_id in id_to_task:
            config["task"] = id_to_task[token_id]
    return config


def load_task_config(preset, transformers_config):
    """Return the token suppression config for `WhisperAudioToText`."""
    generation_config = _load_generation_config(preset)
    config = {}
    for key in ("suppress_tokens", "begin_suppress_tokens"):
        # Older checkpoints only carry these in `config.json`.
        value = generation_config.get(key, transformers_config.get(key))
        if value:
            config[key] = [int(token_id) for token_id in value]
    return config
