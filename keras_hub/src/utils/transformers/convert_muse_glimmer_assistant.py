import numpy as np

from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)

backbone_cls = MuseGlimmerBackbone


def _transpose(hf_tensor, _):
    return np.transpose(hf_tensor, axes=(1, 0))


def _multi_head_transpose(hf_tensor, keras_shape):
    """HF `nn.Linear` (out_features, in_features) -> a 3D EinsumDense kernel.

    Same reshape as `convert_muse_glimmer._multi_head_transpose`: the
    Q/K/V/output projections of `MuseGlimmerTextAttention` are 3D
    `EinsumDense` kernels; transposing to `(in_features, out_features)`
    leaves the `heads * head_dim` block contiguous and head-major, so a
    plain reshape into `keras_shape` completes the conversion.
    """
    return np.transpose(hf_tensor, axes=(1, 0)).reshape(keras_shape)


def convert_backbone_config(transformers_config):
    """Map the assistant's config.json -> `MuseGlimmerBackbone` kwargs.

    Always sets the four (arguably five) drafter opt-in flags: this
    checkpoint has no other use than as a DFlash drafter conditioned on
    the separate 30B target model's hidden states (see migration report
    Section 4).
    """
    rope_theta = transformers_config.get("rope_parameters", {}).get(
        "rope_theta"
    )
    if rope_theta is None:
        rope_theta = transformers_config.get("rope_theta", 500000.0)

    return {
        # No vocabulary in this checkpoint (`use_external_embeddings=True`
        # skips building `ReversibleEmbedding` entirely); the value is
        # unused but kept for `get_config()` round-tripping.
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
        "use_sandwich_norm": False,
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
    """Port `MuseGlimmerAssistantModel` backbone weights from HF.

    `q_norm`/`k_norm` are not ported: `MuseGlimmerTextAttention` always
    builds its QK-norm scaleless (`with_scale=False`), so there is no
    KerasHub weight to receive a learned HF scale, if one exists in the
    real checkpoint. `output_norm_enc` (context projection norm) is not
    ported for the same reason (`MuseGlimmerContextProjection.norm` is
    scaleless). Both are flagged for human verification against the real
    checkpoint before this preset is finalized (see migration report
    Section 5).
    """
    for i in range(backbone.num_layers):
        layer = backbone.transformer_layers[i]
        prefix = f"model.layers.{i}"
        attn = layer._self_attention_layer

        loader.port_weight(
            layer._input_layernorm.scale,
            f"{prefix}.input_layernorm.weight",
        )
        # HF's `post_attention_layernorm` plays the *pre-norm* role before
        # the MLP in this checkpoint's plain two-norm decoder layer (no
        # sandwich norms here, `use_sandwich_norm=False`), so it maps to
        # `_pre_feedforward_layernorm`, not `_post_attention_layernorm`
        # (which is not even built when `use_sandwich_norm=False`).
        loader.port_weight(
            layer._pre_feedforward_layernorm.scale,
            f"{prefix}.post_attention_layernorm.weight",
        )

        loader.port_weight(
            attn._query_dense.kernel,
            f"{prefix}.self_attn.q_proj.weight",
            hook_fn=_multi_head_transpose,
        )
        loader.port_weight(
            attn._key_dense.kernel,
            f"{prefix}.self_attn.k_proj.weight",
            hook_fn=_multi_head_transpose,
        )
        loader.port_weight(
            attn._value_dense.kernel,
            f"{prefix}.self_attn.v_proj.weight",
            hook_fn=_multi_head_transpose,
        )
        loader.port_weight(
            attn._output_dense.kernel,
            f"{prefix}.self_attn.o_proj.weight",
            hook_fn=_multi_head_transpose,
        )

        loader.port_weight(
            layer._feedforward_gate_dense.kernel,
            f"{prefix}.mlp.gate_proj.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            layer._feedforward_up_dense.kernel,
            f"{prefix}.mlp.up_proj.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            layer._feedforward_down_dense.kernel,
            f"{prefix}.mlp.down_proj.weight",
            hook_fn=_transpose,
        )

    loader.port_weight(backbone.layer_norm.scale, "model.norm.weight")

    loader.port_weight(
        backbone.context_projection.dense.kernel,
        "model.encoder.fc.weight",
        hook_fn=_transpose,
    )


def convert_head(model, loader, transformers_config):
    """No dedicated top-level projection layers to port.

    Unlike `Gemma4AssistantCausalLM`, `MuseGlimmerAssistantCausalLM` has no
    `pre_projection`/`post_projection` layers of its own — its backbone
    covers every weight in the checkpoint (see `convert_weights`).
    """
    del model, loader, transformers_config
