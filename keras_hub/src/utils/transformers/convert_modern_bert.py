import numpy as np

from keras_hub.src.models.modernbert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.utils.preset_utils import HF_TOKENIZER_CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = ModernBertBackbone


def convert_backbone_config(transformers_config):
    """Convert a Hugging Face ModernBERT config to KerasHub backbone params."""

    rope_theta = transformers_config.get("rope_theta")
    if rope_theta is None:
        rope_config = transformers_config.get("rope_config", {})
        rope_theta = rope_config.get("rope_theta", 160000.0)

    return {
        "vocabulary_size": transformers_config.get("vocab_size", 50368),
        "hidden_dim": transformers_config.get("hidden_size", 768),
        "intermediate_dim": transformers_config.get("intermediate_size", 1152),
        "num_layers": transformers_config.get("num_hidden_layers", 22),
        "num_heads": transformers_config.get("num_attention_heads", 12),
        "dropout": transformers_config.get("attention_dropout", 0.0),
        "local_attention_window": transformers_config.get(
            "local_attention", 128
        ),
        "global_attn_every_n_layers": transformers_config.get(
            "global_attn_every_n_layers", 3
        ),
        "max_sequence_length": transformers_config.get(
            "max_position_embeddings", 8192
        ),
        "rotary_max_wavelength": float(rope_theta),
        "layer_norm_epsilon": transformers_config.get(
            "norm_eps", transformers_config.get("layer_norm_eps", 1e-5)
        ),
    }


def _port_wqkv(hf_tensor, keras_shape, num_heads):
    del keras_shape, num_heads

    return hf_tensor.T


def _port_wqkv_bias(hf_tensor, keras_shape, num_heads):
    del keras_shape, num_heads

    return hf_tensor


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
    """Extract weight/gamma/scale variable from Keras LayerNorm/RMSNorm."""
    if norm_layer is None:
        return None
    if hasattr(norm_layer, "gamma") and norm_layer.gamma is not None:
        return norm_layer.gamma
    if hasattr(norm_layer, "scale") and norm_layer.scale is not None:
        return norm_layer.scale
    if hasattr(norm_layer, "weights") and len(norm_layer.weights) > 0:
        return norm_layer.weights[0]
    return None


def convert_weights(backbone, loader, _):
    """Port HuggingFace ModernBERT weights into KerasHub backbone."""

    # Token Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="embeddings.tok_embeddings.weight",
    )

    # Embedding Norm
    if (
        hasattr(backbone, "embedding_norm")
        and backbone.embedding_norm is not None
    ):
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
                hook_fn=lambda x, s: _port_wqkv(x, s, backbone.num_heads),
            )
            if getattr(keras_layer.attn.qkv, "bias", None) is not None:
                loader.port_weight(
                    keras_variable=keras_layer.attn.qkv.bias,
                    hf_weight_key=f"layers.{index}.attn.Wqkv.bias",
                    hook_fn=lambda x, s: _port_wqkv_bias(
                        x, s, backbone.num_heads
                    ),
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


def convert_tokenizer(cls, preset, **kwargs):
    """Convert a Hugging Face ModernBERT tokenizer."""

    tokenizer_config = load_json(
        preset,
        HF_TOKENIZER_CONFIG_FILE,
    )

    return cls(
        vocabulary=get_file(preset, "vocab.json"),
        merges=get_file(preset, "merges.txt"),
        sequence_length=kwargs.pop("sequence_length", None),
        add_prefix_space=tokenizer_config.get(
            "add_prefix_space",
            tokenizer_config.get("pre_tokenizer", {}).get(
                "add_prefix_space", False
            ),
        ),
    )
