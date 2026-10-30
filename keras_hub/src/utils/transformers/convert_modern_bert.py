import numpy as np
from keras import layers

from keras_hub.src.models.modern_bert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = ModernBertBackbone

# sentence-transformers pooling modes, keyed by the boolean flag that older
# sentence-transformers versions write to `1_Pooling/config.json`, in the
# order sentence-transformers concatenates them. `ModernBertTextEmbedder`
# supports `"cls"`, `"mean"` and `"max"` under the same names.
_SENTENCE_TRANSFORMERS_POOLING_FLAGS = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len_tokens",
    "pooling_mode_weightedmean_tokens": "weightedmean",
    "pooling_mode_lasttoken": "lasttoken",
}


def convert_backbone_config(transformers_config):
    """Convert a Hugging Face ModernBERT config to KerasHub backbone params."""

    if "rope_parameters" in transformers_config:
        rope_parameters = transformers_config["rope_parameters"]
        global_rope = rope_parameters["full_attention"]
        local_rope = rope_parameters["sliding_attention"]

        rotary_max_wavelength = global_rope["rope_theta"]
        local_rotary_max_wavelength = local_rope["rope_theta"]
    else:
        rotary_max_wavelength = transformers_config["global_rope_theta"]
        local_rotary_max_wavelength = transformers_config["local_rope_theta"]

    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "local_attention_window": transformers_config["local_attention"],
        "global_attn_every_n_layers": transformers_config[
            "global_attn_every_n_layers"
        ],
        "rotary_max_wavelength": float(rotary_max_wavelength),
        "local_rotary_max_wavelength": float(local_rotary_max_wavelength),
        "layer_norm_epsilon": transformers_config["norm_eps"],
    }


def _split_wi(hf_tensor, keras_shape, index):
    """
    Split HF GeGLU input projection into gate and input components.
    HF ModernBERT layout: Chunk 0 = wi_0 (gate), Chunk 1 = wi_1 (input).
    """
    del keras_shape

    # Shape checks on checkpoint data, so these must survive `python -O`.
    if hf_tensor.ndim != 2:
        raise ValueError(
            "Expected the GeGLU input projection to be 2D, received shape "
            f"{hf_tensor.shape}."
        )
    if hf_tensor.shape[0] % 2 != 0:
        raise ValueError(
            "Expected the GeGLU input projection's first dimension to be "
            "even so it can be split into gate and value halves, received "
            f"shape {hf_tensor.shape}."
        )

    gate, value = np.split(hf_tensor, 2, axis=0)

    weight = gate if index == 0 else value

    return weight.T


def _split_bias(hf_tensor, keras_shape, index):
    """Split HF GeGLU bias into wi_0 and wi_1."""
    chunks = np.split(hf_tensor, 2, axis=0)
    return chunks[index]


def _get_norm_variable(norm_layer):
    """Extract the scale variable from a normalization layer.

    Raises rather than returning `None` on failure. A silently skipped norm
    produces a model that loads without error and computes wrong numbers,
    which is a far worse failure than an explicit conversion error.

    The one legitimate no-op is layer 0's `attn_norm`, which is
    `keras.layers.Identity` because ModernBERT has no attention norm on the
    first layer (HF uses `nn.Identity` and ships no
    `layers.0.attn_norm.weight`). That case returns `None`.
    """
    if isinstance(norm_layer, layers.Identity):
        return None
    if norm_layer is None:
        raise ValueError(
            "Expected a normalization layer to port weights into, received "
            "`None`."
        )
    gamma = getattr(norm_layer, "gamma", None)
    if gamma is None:
        raise ValueError(
            f"Normalization layer `{norm_layer.name}` has no `gamma` scale "
            "variable to port weights into. ModernBERT's norms are "
            "configured with `scale=True`, so this indicates the backbone "
            "was built with an unexpected configuration."
        )
    return gamma


def convert_weights(backbone, loader, _):
    """Port HuggingFace ModernBERT weights into KerasHub backbone."""
    # Token Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="embeddings.tok_embeddings.weight",
    )

    # Embedding Norm. Always present on the backbone, so a failure to resolve
    # the scale variable raises rather than silently skipping.
    loader.port_weight(
        keras_variable=_get_norm_variable(backbone.embedding_norm),
        hf_weight_key="embeddings.norm.weight",
    )

    # Transformer Encoder Layers
    for index in range(backbone.num_layers):
        keras_layer = backbone.transformer_layers[index]

        # Wqkv (Query, Key, Value) with head dimension interleaving
        if hasattr(keras_layer.attn, "qkv"):
            loader.port_weight(
                keras_variable=keras_layer.attn.qkv.kernel,
                hf_weight_key=f"layers.{index}.attn.Wqkv.weight",
                hook_fn=lambda x, _: np.transpose(x),
            )
            if getattr(keras_layer.attn.qkv, "bias", None) is not None:
                loader.port_weight(
                    keras_variable=keras_layer.attn.qkv.bias,
                    hf_weight_key=f"layers.{index}.attn.Wqkv.bias",
                )

        # Output Dense (Wo)
        loader.port_weight(
            keras_variable=keras_layer.attn.output_dense.kernel,
            hf_weight_key=f"layers.{index}.attn.Wo.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )

        if getattr(keras_layer.attn.output_dense, "bias", None) is not None:
            loader.port_weight(
                keras_variable=keras_layer.attn.output_dense.bias,
                hf_weight_key=f"layers.{index}.attn.Wo.bias",
            )

        # Attention Norm. Layer 0 is `Identity` and HF ships no
        # `layers.0.attn_norm.weight`, so there is nothing to port there.
        attn_norm_var = _get_norm_variable(keras_layer.attn_norm)
        if attn_norm_var is None:
            if index != 0:
                raise ValueError(
                    f"Layer {index} has an `Identity` attention norm. Only "
                    "layer 0 is expected to omit its attention norm."
                )
        else:
            loader.port_weight(
                keras_variable=attn_norm_var,
                hf_weight_key=f"layers.{index}.attn_norm.weight",
            )

        # MLP Norm. Present on every layer.
        loader.port_weight(
            keras_variable=_get_norm_variable(keras_layer.mlp_norm),
            hf_weight_key=f"layers.{index}.mlp_norm.weight",
        )

        # MLP Wi (GeGLU gate/input projection)
        loader.port_weight(
            keras_variable=keras_layer.mlp.wi_0.kernel,
            hf_weight_key=f"layers.{index}.mlp.Wi.weight",
            hook_fn=lambda x, s: _split_wi(x, s, 0),
        )

        loader.port_weight(
            keras_variable=keras_layer.mlp.wi_1.kernel,
            hf_weight_key=f"layers.{index}.mlp.Wi.weight",
            hook_fn=lambda x, s: _split_wi(x, s, 1),
        )

        if getattr(keras_layer.mlp.wi_0, "bias", None) is not None:
            loader.port_weight(
                keras_variable=keras_layer.mlp.wi_0.bias,
                hf_weight_key=f"layers.{index}.mlp.Wi.bias",
                hook_fn=lambda x, s: _split_bias(x, s, 0),
            )

        if getattr(keras_layer.mlp.wi_1, "bias", None) is not None:
            loader.port_weight(
                keras_variable=keras_layer.mlp.wi_1.bias,
                hf_weight_key=f"layers.{index}.mlp.Wi.bias",
                hook_fn=lambda x, s: _split_bias(x, s, 1),
            )

        # MLP Wo
        loader.port_weight(
            keras_variable=keras_layer.mlp.wo.kernel,
            hf_weight_key=f"layers.{index}.mlp.Wo.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )

        if getattr(keras_layer.mlp.wo, "bias", None) is not None:
            loader.port_weight(
                keras_variable=keras_layer.mlp.wo.bias,
                hf_weight_key=f"layers.{index}.mlp.Wo.bias",
            )

    # Final LayerNorm. Always present on the backbone.
    loader.port_weight(
        keras_variable=_get_norm_variable(backbone.final_norm),
        hf_weight_key="final_norm.weight",
    )


def convert_head(task, loader, transformers_config):
    """Port Hugging Face ModernBERT MLM head weights into KerasHub."""
    del transformers_config

    loader.port_weight(
        keras_variable=task.mlm_head_dense.kernel,
        hf_weight_key="head.dense.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )

    loader.port_weight(
        keras_variable=task.mlm_head_norm.gamma,
        hf_weight_key="head.norm.weight",
    )

    loader.port_weight(
        keras_variable=task.mlm_head_decoder_bias,
        hf_weight_key="decoder.bias",
    )


def convert_tokenizer(cls, preset, **kwargs):
    """Convert a Hugging Face ModernBERT tokenizer."""
    tokenizer_json = load_json(preset, "tokenizer.json")
    tokenizer_model = tokenizer_json["model"]

    if tokenizer_model["type"] != "BPE":
        raise ValueError(
            "Expected a BPE tokenizer for ModernBERT, got "
            f"{tokenizer_model['type']!r}."
        )

    vocab = dict(tokenizer_model["vocab"])
    # Newer `tokenizers` versions serialize merges as `["a", "b"]` pairs
    # (e.g. `nomic-ai/modernbert-embed-base`), while older checkpoints such
    # as `answerdotai/ModernBERT-base` use `"a b"` strings. Normalize to the
    # string form expected by `BytePairTokenizer`.
    merges = [
        merge if isinstance(merge, str) else " ".join(merge)
        for merge in tokenizer_model["merges"]
    ]

    # Ordered rather than a `set`, so `unsplittable_tokens` (and therefore
    # the serialized config) is stable across runs under hash randomization.
    special_tokens = []
    for token in tokenizer_json.get("added_tokens", []):
        vocab[token["content"]] = token["id"]
        if token["content"] not in special_tokens:
            special_tokens.append(token["content"])

    kwargs.setdefault("unsplittable_tokens", special_tokens)
    kwargs.setdefault(
        "add_prefix_space",
        tokenizer_json.get("pre_tokenizer", {}).get("add_prefix_space", False),
    )

    return cls(vocabulary=vocab, merges=merges, **kwargs)


def _read_pooling_mode(pooling_config):
    """Read the pooling mode from a sentence-transformers pooling config.

    Accepts both formats sentence-transformers writes to
    `1_Pooling/config.json`: the `pooling_mode` string (or list of strings)
    written by recent versions, e.g. `"pooling_mode": "mean"`, and the
    boolean `pooling_mode_*` flags written by older ones. As in
    sentence-transformers, `pooling_mode` takes precedence over the flags,
    and a config with no mode enabled means mean pooling.

    Several modes are joined with `"+"` (e.g. `"cls+mean"`). Modes that
    `ModernBertTextEmbedder` does not support are returned by name, so
    building the embedder raises an error naming them.
    """
    if "pooling_mode" in pooling_config:
        modes = pooling_config["pooling_mode"]
        if isinstance(modes, str):
            modes = [modes]
    else:
        modes = [
            mode
            for flag, mode in _SENTENCE_TRANSFORMERS_POOLING_FLAGS.items()
            if pooling_config.get(flag, False)
        ]
    return "+".join(mode.lower() for mode in modes) or "mean"


def load_task_config(preset, transformers_config):
    """Detect sentence-transformers pipeline settings.

    Reads two sentence-transformers config files when available:

    - `modules.json`: determines whether L2 normalization is applied. For
      example, `nomic-ai/modernbert-embed-base` includes a `Normalize`
      module, while `Alibaba-NLP/gte-modernbert-base` does not.
    - `1_Pooling/config.json`: determines the pooling strategy (`"cls"`,
      `"mean"` or `"max"`), in either the string or the boolean-flag format.

    Returns kwargs suitable for `ModernBertTextEmbedder`. A pooling config
    that `ModernBertTextEmbedder` does not support (e.g. `"weightedmean"`,
    or several concatenated strategies) is passed through by name, so
    building the embedder raises an error naming the unsupported mode
    rather than silently falling back to a different pooling strategy.
    Other ModernBERT tasks do not take these kwargs, and the Keras
    functional model constructor drops them.
    """
    del transformers_config
    kwargs = {}

    if check_file_exists(preset, "modules.json"):
        modules = load_json(preset, "modules.json")
        kwargs["normalize"] = any(
            "Normalize" in module.get("type", "") for module in modules
        )

    pooling_config_file = "1_Pooling/config.json"
    if check_file_exists(preset, pooling_config_file):
        pooling_config = load_json(preset, pooling_config_file)
        kwargs["pooling_mode"] = _read_pooling_mode(pooling_config)

    return kwargs


def load_preprocessor_config(preset, transformers_config):
    """Read the sentence-transformers sequence length, if available.

    Sentence-transformers checkpoints define their maximum sequence length
    as `max_seq_length` in `sentence_bert_config.json`, which can differ
    from `max_position_embeddings` in `config.json`. For example,
    `nomic-ai/modernbert-embed-base` specifies `8192`.

    Without `max_seq_length` (e.g. `Alibaba-NLP/gte-modernbert-base` ships
    no `sentence_bert_config.json`), sentence-transformers uses the
    tokenizer's `model_max_length`, capped at `max_position_embeddings`.
    The same length is used here, so `hf://` loads and converted presets
    truncate exactly like sentence-transformers. Checkpoints that are not
    sentence-transformers models (no `modules.json`), such as
    `answerdotai/ModernBERT-base`, keep the preprocessor's default.
    """
    st_config_file = "sentence_bert_config.json"
    if check_file_exists(preset, st_config_file):
        st_config = load_json(preset, st_config_file)
        max_seq_length = st_config.get("max_seq_length")
        if max_seq_length is not None:
            return {"sequence_length": max_seq_length}

    max_position_embeddings = transformers_config.get("max_position_embeddings")
    if max_position_embeddings is None or not check_file_exists(
        preset, "modules.json"
    ):
        return {}
    sequence_length = max_position_embeddings
    if check_file_exists(preset, "tokenizer_config.json"):
        tokenizer_config = load_json(preset, "tokenizer_config.json")
        model_max_length = tokenizer_config.get("model_max_length")
        if model_max_length is not None:
            sequence_length = min(sequence_length, int(model_max_length))
    return {"sequence_length": sequence_length}
