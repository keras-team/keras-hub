"""Convert HuggingFace Qwen2.5-VL weights to KerasHub."""

from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_backbone import (
    Qwen2_5VLBackbone,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_image_converter import (
    Qwen2_5VLImageConverter,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_vision_encoder import (
    Qwen2_5VLVisionEncoder,
)
from keras_hub.src.utils.transformers.convert_qwen2_vl import (
    _port_merger_mlp_weights,
)
from keras_hub.src.utils.transformers.convert_qwen2_vl import _port_patch_embed
from keras_hub.src.utils.transformers.convert_qwen2_vl import _port_text_weights
from keras_hub.src.utils.transformers.convert_qwen2_vl import (
    _port_vision_attention_weights,
)
from keras_hub.src.utils.transformers.convert_qwen2_vl import _transpose_2d
from keras_hub.src.utils.transformers.convert_qwen2_vl import (
    convert_tokenizer as convert_tokenizer,
)
from keras_hub.src.utils.transformers.convert_qwen2_vl import (
    load_image_converter_config as load_image_converter_config,
)

backbone_cls = Qwen2_5VLBackbone


def convert_backbone_config(transformers_config):
    # Newer transformers nest text params under "text_config".
    tc = transformers_config.get("text_config", transformers_config)
    vision_config = transformers_config.get("vision_config", None)
    # transformers 5.x renames "rope_scaling" to "rope_parameters" and may
    # serialize it as null; neither key is guaranteed alongside the other.
    rope_params = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
    rope_theta = rope_params.get("rope_theta")
    if rope_theta is None:
        rope_theta = tc.get("rope_theta")
    if rope_theta is None:
        rope_theta = 1000000
    # M-RoPE section lives inside rope_parameters/rope_scaling in the HF
    # config; HF defaults to [16, 24, 24].
    mrope_section = rope_params.get("mrope_section", [16, 24, 24])
    # Outer level in transformers 5.x; inside text_config in 4.x.
    if "tie_word_embeddings" in transformers_config:
        tie_word_embeddings = transformers_config["tie_word_embeddings"]
    else:
        tie_word_embeddings = tc["tie_word_embeddings"]

    vision_encoder = None
    if vision_config is not None:
        vision_encoder = Qwen2_5VLVisionEncoder(
            patch_size=vision_config.get("patch_size", 14),
            temporal_patch_size=vision_config.get("temporal_patch_size", 2),
            in_channels=vision_config.get(
                "in_chans", vision_config.get("in_channels", 3)
            ),
            embed_dim=vision_config.get("hidden_size", 3584),
            out_dim=vision_config.get("out_hidden_size", 3584),
            depth=vision_config.get("depth", 32),
            num_heads=vision_config.get("num_heads", 16),
            intermediate_dim=vision_config.get("intermediate_size", 3420),
            spatial_merge_size=vision_config.get("spatial_merge_size", 2),
            window_size=vision_config.get("window_size", 112),
            fullatt_block_indexes=vision_config.get(
                "fullatt_block_indexes", [7, 15, 23, 31]
            ),
            theta=(vision_config.get("rope_parameters") or {}).get(
                "rope_theta", vision_config.get("rope_theta", 10000.0)
            ),
        )

    result = {
        "vocabulary_size": tc["vocab_size"],
        "num_layers": tc["num_hidden_layers"],
        "num_query_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "hidden_dim": tc["hidden_size"],
        "intermediate_dim": tc["intermediate_size"],
        "rope_max_wavelength": rope_theta,
        "mrope_section": mrope_section,
        "layer_norm_epsilon": tc["rms_norm_eps"],
        "tie_word_embeddings": tie_word_embeddings,
        "use_sliding_window_attention": tc.get("use_sliding_window", False),
        "dropout": tc["attention_dropout"],
    }
    # transformers 5.x serializes `sliding_window` as null when unused.
    # Omit it then so the backbone default applies.
    if tc.get("sliding_window") is not None:
        result["sliding_window_size"] = tc["sliding_window"]
    if vision_encoder is not None:
        result["vision_encoder"] = vision_encoder
    return result


def load_preprocessor_config(preset, transformers_config):
    """Extra kwargs for ``Qwen2_5VLCausalLMPreprocessor``.

    ``tokens_per_second`` lives in HF ``vision_config`` and scales video
    temporal M-RoPE positions.
    """
    vision_config = transformers_config.get("vision_config")
    if vision_config is None:
        return None
    # Sparse vision configs may omit it; HF's vision config defaults to 4.
    return {"tokens_per_second": vision_config.get("tokens_per_second", 4)}


def load_video_converter_config(preset, transformers_config):
    """Return kwargs for ``Qwen2_5VLVideoConverter``, or None if text-only.

    Qwen2.5-VL applies identical preprocessing to videos and images, and
    its checkpoints do not ship a ``video_preprocessor_config.json``, so
    this delegates to ``load_image_converter_config``.
    """
    image_kwargs = load_image_converter_config(preset, transformers_config)
    if image_kwargs is None:
        return None
    return {"image_converter": Qwen2_5VLImageConverter(**image_kwargs)}


def convert_weights(backbone, loader, transformers_config):
    # Text tower (embeddings, decoder layers, final norm) is identical
    # to Qwen2-VL.
    _port_text_weights(backbone, loader)

    # ── vision encoder ──────────────────────────────────────────────
    vision = backbone.vision_encoder
    if vision is None:
        return backbone

    _port_patch_embed(vision, loader)

    # Vision blocks — RMSNorm and gated SiLU MLP differ from Qwen2-VL's
    # LayerNorm/QuickGELU; the attention projections are shared.
    for i in range(vision.depth):
        block = vision.blocks[i]
        prefix = f"visual.blocks.{i}"

        # RMSNorm 1 (weight only — no bias, unlike Qwen2-VL's LayerNorm)
        loader.port_weight(
            keras_variable=block.norm1.scale,
            hf_weight_key=f"{prefix}.norm1.weight",
        )

        _port_vision_attention_weights(block, prefix, loader)

        # RMSNorm 2
        loader.port_weight(
            keras_variable=block.norm2.scale,
            hf_weight_key=f"{prefix}.norm2.weight",
        )

        # Gated SiLU MLP (bias=True, unlike the text MLP)
        loader.port_weight(
            keras_variable=block.mlp.gate_proj.kernel,
            hf_weight_key=f"{prefix}.mlp.gate_proj.weight",
            hook_fn=_transpose_2d,
        )
        loader.port_weight(
            keras_variable=block.mlp.gate_proj.bias,
            hf_weight_key=f"{prefix}.mlp.gate_proj.bias",
        )
        loader.port_weight(
            keras_variable=block.mlp.up_proj.kernel,
            hf_weight_key=f"{prefix}.mlp.up_proj.weight",
            hook_fn=_transpose_2d,
        )
        loader.port_weight(
            keras_variable=block.mlp.up_proj.bias,
            hf_weight_key=f"{prefix}.mlp.up_proj.bias",
        )
        loader.port_weight(
            keras_variable=block.mlp.down_proj.kernel,
            hf_weight_key=f"{prefix}.mlp.down_proj.weight",
            hook_fn=_transpose_2d,
        )
        loader.port_weight(
            keras_variable=block.mlp.down_proj.bias,
            hf_weight_key=f"{prefix}.mlp.down_proj.bias",
        )

    # Patch merger — RMSNorm ``ln_q`` (scale only), shared MLP port.
    merger = vision.merger
    loader.port_weight(
        keras_variable=merger.ln_q.scale,
        hf_weight_key="visual.merger.ln_q.weight",
    )
    _port_merger_mlp_weights(merger, loader)

    return backbone
