"""HF -> KerasHub weight converter for MuseGlimmer."""

import numpy as np

from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionEncoder,
)
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = MuseGlimmerBackbone


def _transpose(hf_tensor, _):
    return np.transpose(hf_tensor, axes=(1, 0))


def _multi_head_transpose(hf_tensor, keras_shape):
    """HF `nn.Linear` (out_features, in_features) -> a 3D EinsumDense kernel.

    Works for both `MuseGlimmerTextAttention`'s Q/K/V/gate projections
    (kernel shape `(in_dim, heads, head_dim)`) and its output projection
    (kernel shape `(heads, head_dim, out_dim)`): transposing to
    `(in_features, out_features)` always leaves the untouched dimension at
    one edge and the `heads * head_dim` block, contiguous and head-major,
    at the other — so a plain reshape into `keras_shape` splits or merges
    it correctly without further axis permutation.
    """
    return np.transpose(hf_tensor, axes=(1, 0)).reshape(keras_shape)


def convert_backbone_config(transformers_config):
    text_config = transformers_config["text_config"]
    vision_config = transformers_config.get("vision_config", None)

    num_layers = text_config["num_hidden_layers"]
    layer_types = text_config.get("layer_types", None)
    if layer_types is None:
        layer_types = [
            "full_attention"
            if (num_layers - 1 - i) % 4 == 0
            else "sliding_attention"
            for i in range(num_layers)
        ]

    rope_theta = text_config.get("rope_parameters", {}).get("rope_theta")
    if rope_theta is None:
        rope_theta = text_config["rope_theta"]

    vision_encoder = None
    if vision_config is not None:
        out_hidden_size = transformers_config.get(
            "out_hidden_size",
            vision_config["hidden_size"] * vision_config["merge_size"] ** 2,
        )
        vision_encoder = MuseGlimmerVisionEncoder(
            num_layers=vision_config["num_hidden_layers"],
            hidden_size=vision_config["hidden_size"],
            num_heads=vision_config["num_attention_heads"],
            intermediate_size=vision_config["intermediate_size"],
            patch_size=vision_config["patch_size"],
            patch_temporal=vision_config["patch_temporal"],
            merge_size=vision_config["merge_size"],
            pos_emb_height=vision_config["pos_emb_height"],
            pos_emb_width=vision_config["pos_emb_width"],
            rope_theta=vision_config.get("rope_parameters", {}).get(
                "rope_theta", 10000.0
            ),
            layer_norm_eps=vision_config["layer_norm_eps"],
            layer_types=vision_config.get("layer_types", None),
            out_hidden_size=out_hidden_size,
        )

    result = {
        "vocabulary_size": text_config["vocab_size"],
        "num_layers": num_layers,
        "num_query_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "hidden_dim": text_config["hidden_size"],
        "intermediate_dim": text_config["intermediate_size"],
        "head_dim": text_config["head_dim"],
        "sliding_window_size": text_config["sliding_window"],
        "rope_max_wavelength": rope_theta,
        "rms_norm_eps": text_config["rms_norm_eps"],
        "post_norm_eps": text_config["post_norm_eps"],
        "qk_scale_factor": text_config["qk_scale_factor"],
        "output_multiplier": text_config["output_multiplier"],
        "final_logit_softcapping": text_config["final_logit_softcapping"],
        "layer_types": layer_types,
    }
    if vision_encoder is not None:
        result["vision_encoder"] = vision_encoder
        result["projector_hidden_dim"] = transformers_config[
            "projector_hidden_size"
        ]
        result["projector_hidden_act"] = transformers_config.get(
            "projector_hidden_act", "gelu"
        )
    return result


def convert_weights(backbone, loader, transformers_config):
    # Text embedding + its scaleless norm.
    loader.port_weight(
        backbone.token_embedding.embeddings,
        "model.language_model.embed_tokens.weight",
    )
    # `embed_norm` is scaleless (`with_scale=False`) so there is no weight
    # to port for it — HF's `MuseGlimmerTextNormedEmbedding.embed_norm` is
    # likewise scaleless.

    # LM head (untied per the config/`_tied_weights_keys` disagreement
    # noted in the migration report; ported separately from the input
    # embedding).
    loader.port_weight(
        backbone.token_embedding.reverse_embeddings,
        "lm_head.weight",
        hook_fn=_transpose,
    )

    for i in range(backbone.num_layers):
        layer = backbone.transformer_layers[i]
        prefix = f"model.language_model.layers.{i}"
        attn = layer._self_attention_layer

        loader.port_weight(
            layer._input_layernorm.scale,
            f"{prefix}.input_layernorm.weight",
        )
        loader.port_weight(
            layer._post_attention_layernorm.scale,
            f"{prefix}.post_attention_layernorm.weight",
        )
        loader.port_weight(
            layer._pre_feedforward_layernorm.scale,
            f"{prefix}.pre_feedforward_layernorm.weight",
        )
        loader.port_weight(
            layer._post_feedforward_layernorm.scale,
            f"{prefix}.post_feedforward_layernorm.weight",
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
            attn._gate_dense.kernel,
            f"{prefix}.self_attn.gate_proj.weight",
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

    loader.port_weight(
        backbone.layer_norm.scale, "model.language_model.norm.weight"
    )

    if backbone.vision_encoder is not None:
        vis = backbone.vision_encoder
        vis_prefix = "model.vision_tower"

        loader.port_weight(
            vis.patch_embedder.patch_embedding.kernel,
            f"{vis_prefix}.patch_embedder.patch_embedding.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            vis.patch_embedder.position_embedding_table.embeddings,
            f"{vis_prefix}.patch_embedder.position_embedding_table.weight",
        )
        loader.port_weight(vis.ln_pre.gamma, f"{vis_prefix}.ln_pre.weight")
        loader.port_weight(vis.ln_pre.beta, f"{vis_prefix}.ln_pre.bias")
        loader.port_weight(vis.ln_post.gamma, f"{vis_prefix}.ln_post.weight")
        loader.port_weight(vis.ln_post.beta, f"{vis_prefix}.ln_post.bias")

        for i in range(vis.num_layers):
            blk = vis.blocks[i]
            blk_prefix = f"{vis_prefix}.layers.{i}"

            loader.port_weight(blk.norm1.gamma, f"{blk_prefix}.norm1.weight")
            loader.port_weight(blk.norm1.beta, f"{blk_prefix}.norm1.bias")
            loader.port_weight(blk.norm2.gamma, f"{blk_prefix}.norm2.weight")
            loader.port_weight(blk.norm2.beta, f"{blk_prefix}.norm2.bias")

            loader.port_weight(
                blk.attn.q_proj.kernel,
                f"{blk_prefix}.attn.q_proj.weight",
                hook_fn=_transpose,
            )
            loader.port_weight(
                blk.attn.q_proj.bias, f"{blk_prefix}.attn.q_proj.bias"
            )
            loader.port_weight(
                blk.attn.k_proj.kernel,
                f"{blk_prefix}.attn.k_proj.weight",
                hook_fn=_transpose,
            )
            loader.port_weight(
                blk.attn.k_proj.bias, f"{blk_prefix}.attn.k_proj.bias"
            )
            loader.port_weight(
                blk.attn.v_proj.kernel,
                f"{blk_prefix}.attn.v_proj.weight",
                hook_fn=_transpose,
            )
            loader.port_weight(
                blk.attn.v_proj.bias, f"{blk_prefix}.attn.v_proj.bias"
            )
            loader.port_weight(
                blk.attn.proj.kernel,
                f"{blk_prefix}.attn.proj.weight",
                hook_fn=_transpose,
            )
            loader.port_weight(
                blk.attn.proj.bias, f"{blk_prefix}.attn.proj.bias"
            )

            loader.port_weight(
                blk.mlp.fc1.kernel,
                f"{blk_prefix}.mlp.fc1.weight",
                hook_fn=_transpose,
            )
            loader.port_weight(blk.mlp.fc1.bias, f"{blk_prefix}.mlp.fc1.bias")
            loader.port_weight(
                blk.mlp.fc2.kernel,
                f"{blk_prefix}.mlp.fc2.weight",
                hook_fn=_transpose,
            )
            loader.port_weight(blk.mlp.fc2.bias, f"{blk_prefix}.mlp.fc2.bias")

        # Multimodal fusion: adapter (double-activation, no bias) + a
        # separate projection Linear + a final scaleless RMSNorm.
        loader.port_weight(
            backbone.vision_adapter_fc1.kernel,
            "model.vision_adapter.fc1.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            backbone.vision_adapter_fc2.kernel,
            "model.vision_adapter.fc2.weight",
            hook_fn=_transpose,
        )
        loader.port_weight(
            backbone.vision_projection.kernel,
            "model.vision_projection.weight",
            hook_fn=_transpose,
        )
        # `perception_emb_norm` is scaleless — no weight to port.

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = dict(tokenizer_config["model"]["vocab"])
    merges = tokenizer_config["model"]["merges"]
    if merges and isinstance(merges[0], list):
        merges = [" ".join(item) for item in merges]

    # `tokenizer.json` keeps the 2,048 Muse Glimmer special tokens in
    # `added_tokens`, rather than in `model.vocab`. They retain the IDs used
    # by the model and must be added before constructing the KerasHub
    # tokenizer.
    special_tokens = []
    for token in tokenizer_config.get("added_tokens", []):
        content = token["content"]
        vocab[content] = token["id"]
        if token.get("special", False):
            special_tokens.append(content)

    hf_tokenizer_config = load_json(preset, "tokenizer_config.json")
    for token_name in ("bos_token", "eos_token", "pad_token"):
        token = hf_tokenizer_config.get(token_name)
        if token is not None:
            kwargs.setdefault(token_name, token)
    kwargs.setdefault("unsplittable_tokens", special_tokens)

    config = load_json(preset, "config.json")
    kwargs.setdefault("image_token_id", config.get("image_token_id", 200092))
    kwargs.setdefault("video_token_id", config.get("video_token_id", 200091))

    return cls(vocabulary=vocab, merges=merges, **kwargs)
