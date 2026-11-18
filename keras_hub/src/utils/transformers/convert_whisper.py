import json
import re

import numpy as np

from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = WhisperBackbone

# `<|en|>`, `<|fr|>`, `<|jw|>`, `<|yue|>` etc. (1-3 lowercase letters).
_LANGUAGE_TOKEN_RE = re.compile(r"^<\|[a-z]{1,3}\|>$")


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["encoder_layers"],
        "num_heads": transformers_config["encoder_attention_heads"],
        "hidden_dim": transformers_config["d_model"],
        "intermediate_dim": transformers_config["encoder_ffn_dim"],
        "num_mels": transformers_config["num_mel_bins"],
        "dropout": transformers_config.get("dropout", 0.0),
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
    for i in range(backbone.num_layers):
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


def _split_special_and_language_tokens(added_tokens):
    """Split the HF added-tokens map into Whisper special vs language tokens.

    HF Whisper packs language codes (`<|en|>`, `<|jw|>`, ...), task tokens
    (`<|transcribe|>`, ...), structural tokens (`<|startoftranscript|>`,
    `<|endoftext|>`, `<|notimestamps|>`), and per-frame timestamp tokens
    (`<|0.00|>`, ...) into the same `added_tokens.json`. WhisperTokenizer
    expects these split into two maps; the multilingual variant is signaled
    by a non-None `language_tokens` argument.
    """
    language_tokens = {}
    special_tokens = {}
    for token, token_id in added_tokens.items():
        if _LANGUAGE_TOKEN_RE.match(token):
            language_tokens[token] = token_id
        else:
            special_tokens[token] = token_id
    return special_tokens, language_tokens


def convert_tokenizer(cls, preset, **kwargs):
    vocab_file = get_file(preset, "vocab.json")
    merges_file = get_file(preset, "merges.txt")
    added_tokens_file = get_file(preset, "added_tokens.json")
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(added_tokens_file, "r", encoding="utf-8") as f:
        added_tokens = json.load(f)
    # `<|endoftext|>` lives in `vocab.json` for HF Whisper while every other
    # `<|...|>` token (start-of-transcript, language codes, timestamps, ...)
    # lives in `added_tokens.json`. Pull them all into one map.
    pipe_tokens = {
        token: token_id
        for token, token_id in {**vocab, **added_tokens}.items()
        if token.startswith("<|") and token.endswith("|>")
    }
    special_tokens, language_tokens = _split_special_and_language_tokens(
        pipe_tokens
    )
    return cls(
        vocabulary=vocab_file,
        merges=merges_file,
        special_tokens=special_tokens,
        language_tokens=language_tokens or None,
        **kwargs,
    )


def load_audio_converter_config(preset, transformers_config):
    """Return WhisperAudioConverter kwargs from HF preprocessor config."""
    feature_extractor = load_json(preset, "preprocessor_config.json")
    return {
        "num_mels": feature_extractor["feature_size"],
        "num_fft_bins": feature_extractor["n_fft"],
        "stride": feature_extractor["hop_length"],
        "sampling_rate": feature_extractor["sampling_rate"],
        "max_audio_length": feature_extractor["chunk_length"],
    }
