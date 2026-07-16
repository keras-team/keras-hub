import numpy as np

from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoder,
)
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers.convert_gemma4 import (
    _convert_decoder_block,
)
from keras_hub.src.utils.transformers.convert_gemma4 import (
    _convert_decoder_block_weights,
)
from keras_hub.src.utils.transformers.convert_gemma4 import (
    convert_tokenizer as target_convert_tokenizer,
)
from keras_hub.src.utils.transformers.convert_gemma4 import (
    load_image_converter_config as target_load_image_converter_config,
)
from keras_hub.src.utils.transformers.convert_gemma4 import (
    load_video_converter_config as target_load_video_converter_config,
)


def convert_tokenizer(cls, preset, **kwargs):
    return target_convert_tokenizer(cls, preset, **kwargs)


def load_image_converter_config(preset, transformers_config):
    return target_load_image_converter_config(preset, transformers_config)


def load_video_converter_config(preset, transformers_config):
    return target_load_video_converter_config(preset, transformers_config)


backbone_cls = Gemma4Backbone


def convert_backbone_config(transformers_config):
    """Map a DiffusionGemma Transformers config → Gemma4Backbone kwargs."""
    model_type = transformers_config.get("model_type", "diffusion_gemma")
    is_text_only = model_type == "diffusion_gemma_text"

    if is_text_only:
        text_cfg = transformers_config
        vision_encoder = None
        image_size = None
    else:
        # DiffusionGemma may nest text fields under "text_config" (like
        # standard Gemma4) or expose them at the top level — fall back to the
        # top-level dict if "text_config" is absent.
        text_cfg = transformers_config.get("text_config", transformers_config)
        image_size = 896

        if "vision_config" in transformers_config:
            vis_cfg = transformers_config["vision_config"]
            vision_encoder = Gemma4VisionEncoder(
                image_size=image_size,
                patch_size=vis_cfg["patch_size"],
                num_heads=vis_cfg["num_attention_heads"],
                hidden_dim=vis_cfg["hidden_size"],
                num_layers=vis_cfg["num_hidden_layers"],
                intermediate_dim=vis_cfg["intermediate_size"],
                head_dim=vis_cfg.get("head_dim", 64),
                num_key_value_heads=vis_cfg.get(
                    "num_key_value_heads", vis_cfg["num_attention_heads"]
                ),
                output_dim=text_cfg["hidden_size"],
                pool_size=vis_cfg.get("pooling_kernel_size", 3),
                position_embedding_size=vis_cfg.get(
                    "position_embedding_size", 10240
                ),
                rope_max_wavelength=vis_cfg.get("rope_parameters", {}).get(
                    "rope_theta", 100.0
                ),
                layer_norm_epsilon=vis_cfg.get("rms_norm_eps", 1e-6),
                use_clipped_linears=vis_cfg.get("use_clipped_linears", True),
                standardize=vis_cfg.get("standardize", False),
            )
        else:
            vision_encoder = None

    if (
        "layer_types" in text_cfg
        and text_cfg["layer_types"]
        and len(text_cfg["layer_types"]) > 1
    ):
        layer_types = text_cfg["layer_types"]
        try:
            first_idx = layer_types.index("full_attention")
            second_idx = layer_types.index("full_attention", first_idx + 1)
            sliding_window_pattern = second_idx - first_idx
        except ValueError:
            sliding_window_pattern = 6
    else:
        sliding_window_pattern = (
            text_cfg.get("_sliding_window_pattern")
            or text_cfg.get("sliding_window_pattern")
            or 6
        )

    rope_params = text_cfg.get("rope_parameters") or {}
    global_rope_partial_rotary_factor = rope_params.get(
        "full_attention", {}
    ).get("partial_rotary_factor")
    global_rope_theta = rope_params.get("full_attention", {}).get("rope_theta")
    local_rope_theta = rope_params.get("sliding_attention", {}).get(
        "rope_theta"
    )
    if global_rope_theta is None:
        global_rope_theta = text_cfg.get("rope_theta")
    if local_rope_theta is None:
        local_rope_theta = text_cfg.get("rope_theta")

    hf_bidir = text_cfg.get("use_bidirectional_attention")
    use_vision_bidirectional_attention = hf_bidir == "vision"

    # MoE: DiffusionGemma may omit enable_moe_block and signal MoE via
    # num_experts alone.
    enable_moe_block = text_cfg.get("enable_moe_block") or bool(
        text_cfg.get("num_experts", 0)
    )

    return {
        "vocabulary_size": text_cfg.get("vocab_size", 262144),
        "image_size": image_size,
        "num_layers": text_cfg["num_hidden_layers"],
        "num_query_heads": text_cfg.get("num_attention_heads", 8),
        "num_key_value_heads": text_cfg.get("num_key_value_heads", 1),
        "hidden_dim": text_cfg["hidden_size"],
        "intermediate_dim": text_cfg["intermediate_size"],
        "head_dim": text_cfg["head_dim"],
        "global_head_dim": text_cfg.get("global_head_dim", None),
        "attention_logit_soft_cap": text_cfg.get(
            "attn_logit_softcapping", None
        ),
        "final_logit_soft_cap": text_cfg.get("final_logit_softcapping", None),
        "use_sliding_window_attention": text_cfg.get("sliding_window", 0) > 0,
        "sliding_window_size": text_cfg.get("sliding_window", 512) or 512,
        "sliding_window_pattern": sliding_window_pattern,
        "layer_norm_epsilon": text_cfg.get("rms_norm_eps", 1e-6),
        "layer_types": text_cfg["layer_types"],
        "vision_encoder": vision_encoder,
        "audio_encoder": None,
        "num_kv_shared_layers": text_cfg.get("num_kv_shared_layers", 0),
        "num_global_key_value_heads": text_cfg.get(
            "num_global_key_value_heads", None
        ),
        "hidden_size_per_layer_input": (
            text_cfg.get("hidden_size_per_layer_input") or 0
        ),
        "vocab_size_per_layer_input": text_cfg.get(
            "vocab_size_per_layer_input", None
        ),
        "global_rope_partial_rotary_factor": global_rope_partial_rotary_factor,
        "global_rope_wavelength": global_rope_theta,
        "local_rope_wavelength": local_rope_theta,
        "use_double_wide_mlp": text_cfg.get("use_double_wide_mlp", False),
        "enable_moe_block": enable_moe_block,
        "num_experts": text_cfg.get("num_experts", None),
        "expert_intermediate_dim": (
            text_cfg.get("moe_intermediate_size")
            or text_cfg.get("expert_intermediate_size")
        ),
        "num_experts_per_token": text_cfg.get("top_k_experts") or 8,
        "use_vision_bidirectional_attention": (
            use_vision_bidirectional_attention
        ),
        # DiffusionGemma encoder and decoder passes use separate per-layer
        # scalars (HF buffers are not tied across the two passes).
        "has_encoder_layer_scalar": True,
        # Self-conditioning lives inside model.decoder in HF; mirror that by
        # keeping it in the backbone rather than the task.
        "has_diffusion_self_conditioning": True,
    }


def convert_task_config(transformers_config):
    """Map DiffusionGemma config keys → Gemma4BlockDiffusionLM kwargs."""
    kwargs = {}
    if "canvas_length" in transformers_config:
        kwargs["canvas_length"] = transformers_config["canvas_length"]
    if "max_denoising_steps" in transformers_config:
        kwargs["max_denoising_steps"] = transformers_config[
            "max_denoising_steps"
        ]
    if "t_min" in transformers_config:
        kwargs["t_min"] = transformers_config["t_min"]
    if "t_max" in transformers_config:
        kwargs["t_max"] = transformers_config["t_max"]
    return kwargs


def _convert_vision_encoder(vision_encoder, loader, transformers_config):
    """Port vision-encoder weights using DiffusionGemma HF path layout."""
    image_encoder = vision_encoder.get_layer("image_encoder")
    patch_embedder = image_encoder.patch_embedder

    vis_prefix = "model.encoder.vision_tower"

    loader.port_weight(
        keras_variable=patch_embedder.input_proj.kernel,
        hf_weight_key=f"{vis_prefix}.patch_embedder.input_proj.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )
    loader.port_weight(
        keras_variable=patch_embedder.position_embedding_table,
        hf_weight_key=f"{vis_prefix}.patch_embedder.position_embedding_table",
    )

    for i, block in enumerate(image_encoder.encoder_blocks):
        vis_layer_prefix = f"{vis_prefix}.encoder.layers.{i}"
        _convert_decoder_block_weights(block, vis_layer_prefix, loader)

    projector_prefix = "model.encoder.embed_vision"
    vision_output = vision_encoder.get_layer("vision_output_encoder")
    loader.port_weight(
        keras_variable=vision_output.vision_input_projection.kernel,
        hf_weight_key=f"{projector_prefix}.embedding_projection.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )

    vis_cfg = transformers_config.get("vision_config", {})
    if vis_cfg.get("standardize", False):
        loader.port_weight(
            keras_variable=vision_output.std_bias,
            hf_weight_key=f"{vis_prefix}.std_bias",
        )
        loader.port_weight(
            keras_variable=vision_output.std_scale,
            hf_weight_key=f"{vis_prefix}.std_scale",
        )


def convert_weights(backbone, loader, transformers_config):
    model_type = transformers_config.get("model_type", "diffusion_gemma")

    # Text-only variant: weights live directly under "model.*".
    # Full model: the decoder transformer is under "model.decoder.*".
    if model_type == "diffusion_gemma_text":
        text_prefix = "model"
    else:
        text_prefix = "model.decoder"

    def hf_key(suffix):
        return f"{text_prefix}.{suffix}"

    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key=hf_key("embed_tokens.weight"),
    )

    if backbone.hidden_size_per_layer_input > 0:
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "per_layer_token_embedding"
            ).embeddings,
            hf_weight_key=hf_key("embed_tokens_per_layer.weight"),
        )
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "per_layer_model_projection"
            ).kernel,
            hf_weight_key=hf_key("per_layer_model_projection.weight"),
            hook_fn=lambda x, _: np.transpose(x),
        )
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "per_layer_projection_norm"
            ).scale,
            hf_weight_key=hf_key("per_layer_projection_norm.weight"),
        )

    vision_encoder = backbone.vision_encoder
    if vision_encoder is not None:
        _convert_vision_encoder(vision_encoder, loader, transformers_config)

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")
        _convert_decoder_block(decoder_layer, i, loader, hf_key)

    # encoder_layer_scalar: DiffusionGemma stores separate per-layer scalars
    # for the encoder and decoder passes; the shared _convert_decoder_block
    # only ports the decoder copy (under model.decoder.*).
    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")
        loader.port_weight(
            keras_variable=decoder_layer.encoder_layer_scalar,
            hf_weight_key=(
                f"model.encoder.language_model.layers.{i}.layer_scalar"
            ),
            hook_fn=lambda x, _: np.squeeze(x),
        )

    if backbone.has_diffusion_self_conditioning:
        sc = backbone.diffusion_self_conditioning
        hf_sc_prefix = "model.decoder.self_conditioning"
        loader.port_weight(
            keras_variable=sc.pre_norm.scale,
            hf_weight_key=f"{hf_sc_prefix}.pre_norm.weight",
        )
        loader.port_weight(
            keras_variable=sc.gate_proj.kernel,
            hf_weight_key=f"{hf_sc_prefix}.gate_proj.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )
        loader.port_weight(
            keras_variable=sc.up_proj.kernel,
            hf_weight_key=f"{hf_sc_prefix}.up_proj.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )
        loader.port_weight(
            keras_variable=sc.down_proj.kernel,
            hf_weight_key=f"{hf_sc_prefix}.down_proj.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )
        # post_norm has no learnable scale (Gemma4VNorm) — no weight to port.

    loader.port_weight(
        keras_variable=backbone.get_layer("final_normalization").scale,
        hf_weight_key=hf_key("norm.weight"),
    )

    return backbone


def load_task_config(preset, transformers_config):
    """Read generation_config.json and return Gemma4BlockDiffusionLM kwargs."""
    if not check_file_exists(preset, "generation_config.json"):
        return {}
    gen_cfg = load_json(preset, "generation_config.json")
    kwargs = {}
    # HF may store canvas length as "block_length" or "canvas_length".
    canvas_length = gen_cfg.get("canvas_length") or gen_cfg.get("block_length")
    if canvas_length is not None:
        kwargs["canvas_length"] = canvas_length
    if "max_denoising_steps" in gen_cfg:
        kwargs["max_denoising_steps"] = gen_cfg["max_denoising_steps"]
    if "t_min" in gen_cfg:
        kwargs["t_min"] = gen_cfg["t_min"]
    if "t_max" in gen_cfg:
        kwargs["t_max"] = gen_cfg["t_max"]
    return kwargs


def load_preprocessor_config(preset, transformers_config):
    """Return extra Gemma4BlockDiffusionLMPreprocessor kwargs."""
    kwargs = {
        "add_start_token": False,
        "add_end_token": False,
    }
    if not check_file_exists(preset, "processor_config.json"):
        return kwargs
    processor_config = load_json(preset, "processor_config.json")
    if "image_processor" in processor_config:
        image_proc = processor_config["image_processor"]
        kwargs["num_vision_tokens_per_image"] = image_proc.get(
            "max_soft_tokens", 280
        )
        kwargs["sequence_length"] = 1024
    if "video_processor" in processor_config:
        video_proc = processor_config["video_processor"]
        kwargs["num_frames_per_video"] = video_proc["num_frames"]
        kwargs["num_vision_tokens_per_frame"] = video_proc["max_soft_tokens"]
        kwargs["video_fps"] = 24.0
        kwargs["sequence_length"] = 1024
    return kwargs
