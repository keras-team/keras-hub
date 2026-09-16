import numpy as np

from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)

backbone_cls = MuseGlimmerBackbone


def convert_backbone_config(transformers_config):
    """Map the assistant's config.json -> `MuseGlimmerBackbone` kwargs."""
    rope_theta = transformers_config.get("rope_parameters", {}).get(
        "rope_theta"
    )
    if rope_theta is None:
        rope_theta = transformers_config.get("rope_theta", 500000.0)

    return {
        # Required by `MuseGlimmerBackbone.__init__` but unused here.
        "vocabulary_size": transformers_config.get("vocab_size", 1),
        "num_layers": transformers_config["num_hidden_layers"],
        "num_query_heads": transformers_config["num_attention_heads"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "head_dim": transformers_config["head_dim"],
        "sliding_window_size": transformers_config.get("sliding_window", 2048),
        "rope_max_wavelength": rope_theta,
        "rms_norm_eps": transformers_config.get("rms_norm_eps", 1e-5),
        "layer_types": transformers_config.get("layer_types"),
        "use_bidirectional_attention": True,
        "context_projection_layer_ids": transformers_config["target_layer_ids"],
        "use_external_embeddings": True,
        "enable_qk_scale_and_gate": False,
        "qk_norm_with_scale": True,
        "use_sandwich_norm": False,
        "use_centered_norm": False,
    }


def convert_task_config(transformers_config):
    """Map config.json -> `MuseGlimmerAssistantCausalLM` kwargs."""
    kwargs = {}
    if "block_size" in transformers_config:
        kwargs["block_size"] = transformers_config["block_size"]
    if "mask_token_id" in transformers_config:
        kwargs["mask_token_id"] = transformers_config["mask_token_id"]
    return kwargs


def convert_weights(backbone, loader, transformers_config):
    """Port `MuseGlimmerAssistantModel` backbone weights from HF."""

    for i in range(backbone.num_layers):
        layer = backbone.transformer_layers[i]
        attn = layer._self_attention_layer

        loader.port_weight(
            layer._input_layernorm.scale,
            f"layers.{i}.input_layernorm.weight",
        )
        # HF's `post_attention_layernorm` is the pre-MLP norm here.
        loader.port_weight(
            layer._pre_feedforward_layernorm.scale,
            f"layers.{i}.post_attention_layernorm.weight",
        )

        loader.port_weight(
            attn._query_dense.kernel,
            f"layers.{i}.self_attn.q_proj.weight",
            hook_fn=lambda x, shape: np.transpose(x, axes=(1, 0)).reshape(
                shape
            ),
        )
        loader.port_weight(
            attn._key_dense.kernel,
            f"layers.{i}.self_attn.k_proj.weight",
            hook_fn=lambda x, shape: np.transpose(x, axes=(1, 0)).reshape(
                shape
            ),
        )
        loader.port_weight(
            attn._value_dense.kernel,
            f"layers.{i}.self_attn.v_proj.weight",
            hook_fn=lambda x, shape: np.transpose(x, axes=(1, 0)).reshape(
                shape
            ),
        )
        loader.port_weight(
            attn._output_dense.kernel,
            f"layers.{i}.self_attn.o_proj.weight",
            hook_fn=lambda x, shape: np.transpose(x, axes=(1, 0)).reshape(
                shape
            ),
        )
        loader.port_weight(
            attn._query_norm.scale, f"layers.{i}.self_attn.q_norm.weight"
        )
        loader.port_weight(
            attn._key_norm.scale, f"layers.{i}.self_attn.k_norm.weight"
        )

        loader.port_weight(
            layer._feedforward_gate_dense.kernel,
            f"layers.{i}.mlp.gate_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
        )
        loader.port_weight(
            layer._feedforward_up_dense.kernel,
            f"layers.{i}.mlp.up_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
        )
        loader.port_weight(
            layer._feedforward_down_dense.kernel,
            f"layers.{i}.mlp.down_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
        )

    loader.port_weight(backbone.layer_norm.scale, "norm.weight")

    loader.port_weight(
        backbone.context_projection.dense.kernel,
        "encoder.fc.weight",
        hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
    )
    loader.port_weight(
        backbone.context_projection.norm.scale, "encoder.output_norm_enc.weight"
    )
