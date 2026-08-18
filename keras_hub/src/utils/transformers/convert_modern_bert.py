import numpy as np

from keras_hub.src.models.modernbert.modern_bert_backbone import ModernBertBackbone
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = ModernBertBackbone


def convert_backbone_config(transformers_config):
    """Convert a Hugging Face ModernBERT config to KerasHub backbone params."""

    rope_parameters = transformers_config.get("rope_parameters", {})

    global_rope = rope_parameters.get("full_attention") or rope_parameters.get("global_attention") or {}

    rotary_max_wavelength = global_rope.get(
        "rope_theta",
        transformers_config.get("global_rope_theta", 160000.0),
    )

    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "dropout": transformers_config["attention_dropout"],
        "local_attention_window": transformers_config["local_attention"],
        "global_attn_every_n_layers": transformers_config["global_attn_every_n_layers"],
        "rotary_max_wavelength": float(rotary_max_wavelength),
        "layer_norm_epsilon": transformers_config["norm_eps"],
    }


def _split_wi(hf_tensor, keras_shape, index):
    """
    Split HF GeGLU input projection into gate and input components.
    HF ModernBERT layout: Chunk 0 = wi_0 (gate), Chunk 1 = wi_1 (input).
    """
    del keras_shape

    assert hf_tensor.ndim == 2
    assert hf_tensor.shape[0] % 2 == 0

    gate, value = np.split(hf_tensor, 2, axis=0)

    weight = gate if index == 0 else value

    return weight.T


def _split_bias(hf_tensor, keras_shape, index):
    """Split HF GeGLU bias into wi_0 and wi_1."""
    chunks = np.split(hf_tensor, 2, axis=0)
    return chunks[index]


def _get_norm_variable(norm_layer):
    """Extract the scale variable from a normalization layer."""
    if norm_layer is None:
        return None
    if hasattr(norm_layer, "gamma") and norm_layer.gamma is not None:
        return norm_layer.gamma
    return None


def convert_weights(backbone, loader, _):
    """Port HuggingFace ModernBERT weights into KerasHub backbone."""
    # Token Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="embeddings.tok_embeddings.weight",
    )

    # Embedding Norm
    if hasattr(backbone, "embedding_norm") and backbone.embedding_norm is not None:
        embedding_norm_var = _get_norm_variable(backbone.embedding_norm)
        if embedding_norm_var is not None:
            loader.port_weight(
                keras_variable=embedding_norm_var,
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

        # Attention Norm
        attn_norm_var = _get_norm_variable(keras_layer.attn_norm)
        if attn_norm_var is not None:
            loader.port_weight(
                keras_variable=attn_norm_var,
                hf_weight_key=f"layers.{index}.attn_norm.weight",
            )

        # MLP Norm
        mlp_norm_var = _get_norm_variable(keras_layer.mlp_norm)
        if mlp_norm_var is not None:
            loader.port_weight(
                keras_variable=mlp_norm_var,
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

    # Final LayerNorm
    if hasattr(backbone, "final_norm") and backbone.final_norm is not None:
        final_norm_var = _get_norm_variable(backbone.final_norm)
        if final_norm_var is not None:
            loader.port_weight(
                keras_variable=final_norm_var,
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

    tokenizer_json = load_json(
        preset,
        "tokenizer.json",
    )
    tokenizer_model = tokenizer_json["model"]

    if tokenizer_model["type"] != "BPE":
        raise ValueError(f"Expected a BPE tokenizer for ModernBERT, got {tokenizer_model['type']!r}.")

    return cls(
        vocabulary=tokenizer_model["vocab"],
        merges=tokenizer_model["merges"],
        sequence_length=kwargs.pop("sequence_length", None),
        add_prefix_space=tokenizer_json.get("pre_tokenizer", {}).get(
            "add_prefix_space",
            False,
        ),
    )
