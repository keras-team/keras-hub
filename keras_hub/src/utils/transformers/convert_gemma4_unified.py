import numpy as np

from keras_hub.src.models.gemma4.gemma4_audio_encoder import Gemma4AudioEncoder
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoder,
)
from keras_hub.src.models.gemma4_unified.gemma4_unified_audio_embedder import (
    Gemma4UnifiedAudioEmbedder,
)
from keras_hub.src.models.gemma4_unified.gemma4_unified_backbone import (
    Gemma4UnifiedBackbone,
)
from keras_hub.src.models.gemma4_unified.gemma4_unified_vision_embedder import (
    Gemma4UnifiedVisionEmbedder,
)
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers.convert_gemma4 import (
    _compute_scale_offset,
)
from keras_hub.src.utils.transformers.convert_gemma4 import (
    _convert_decoder_block,
)
from keras_hub.src.utils.transformers.convert_gemma4 import _resolve_prefix
from keras_hub.src.utils.transformers.convert_gemma4 import (  # noqa: F401
    convert_tokenizer,  # re-exported: the preset loader calls it on this module
)

backbone_cls = Gemma4UnifiedBackbone


def load_image_converter_config(preset, transformers_config):
    """Return kwargs for Gemma4ImageConverter, or None for text-only models."""
    if "vision_config" not in transformers_config:
        return None

    model_type = transformers_config["model_type"]
    is_unified = model_type.startswith("gemma4_unified")

    processor_config = load_json(preset, "processor_config.json")

    vision_config = transformers_config["vision_config"]
    pooling_kernel_size = vision_config["pooling_kernel_size"]

    if is_unified:
        # Unified models use a different image processor structure.
        # The processor_config may have image_processor nested or at top level.
        if "image_processor" in processor_config:
            image_processor = processor_config["image_processor"]
        else:
            image_processor = processor_config
        scale, offset = None, None
        if image_processor.get("do_rescale") or image_processor.get(
            "do_normalize"
        ):
            scale, offset = _compute_scale_offset(
                image_processor, "image_processor"
            )

        patch_size = vision_config["patch_size"]
        max_soft_tokens = vision_config["num_soft_tokens"]

        return {
            "image_size": None,
            "patch_size": patch_size,
            "max_soft_tokens": max_soft_tokens,
            "pooling_kernel_size": pooling_kernel_size,
            "scale": scale,
            "offset": offset,
        }
    else:
        image_processor = processor_config["image_processor"]
        scale, offset = _compute_scale_offset(
            image_processor, "image_processor"
        )

        # image_size is not present in the HF vision_config; 896 is the fixed
        # positional-embedding size used across all Gemma4 vision checkpoints
        # (matches the hardcoded value in convert_backbone_config).
        image_size = 896
        patch_size = vision_config["patch_size"]
        # max_soft_tokens lives in the image_processor section of
        # processor_config; vision_soft_tokens_per_image in the model
        # config is the authoritative cap.
        max_soft_tokens = image_processor["max_soft_tokens"]

        return {
            "image_size": (image_size, image_size),
            "patch_size": patch_size,
            "max_soft_tokens": max_soft_tokens,
            "pooling_kernel_size": pooling_kernel_size,
            "scale": scale,
            "offset": offset,
        }


def load_audio_converter_config(preset, transformers_config):
    """Return Gemma4AudioConverter kwargs, or None for text/vision models."""
    if not transformers_config.get("audio_config"):
        return None

    model_type = transformers_config["model_type"]
    is_unified = model_type.startswith("gemma4_unified")

    if is_unified:
        # Unified models chunk raw waveform into fixed-length frames
        # (no mel spectrograms, no conformer encoder).
        processor_config = load_json(preset, "processor_config.json")
        feature_extractor = processor_config["feature_extractor"]
        return {
            "audio_samples_per_token": feature_extractor[
                "audio_samples_per_token"
            ],
            "sampling_rate": feature_extractor["sampling_rate"],
        }

    processor_config = load_json(preset, "processor_config.json")
    feature_extractor = processor_config["feature_extractor"]

    # Map HF keys to KerasHub keys
    return {
        "num_mels": feature_extractor["feature_size"],
        "num_fft_bins": feature_extractor["fft_length"],
        "stride": feature_extractor["hop_length"],
        "sampling_rate": feature_extractor["sampling_rate"],
        "frame_length": feature_extractor["frame_length"],
        "max_frequency": feature_extractor["max_frequency"],
        "min_frequency": feature_extractor["min_frequency"],
        "mel_floor": feature_extractor["mel_floor"],
    }


def load_video_converter_config(preset, transformers_config):
    """Return Gemma4VideoConverter kwargs, or None if not a video model."""
    processor_config = load_json(preset, "processor_config.json")
    if "video_processor" not in processor_config:
        return None

    video_proc = processor_config["video_processor"]
    scale, offset = _compute_scale_offset(video_proc, "video_processor")

    result = {
        "num_frames": video_proc["num_frames"],
        "max_soft_tokens": video_proc["max_soft_tokens"],
        "patch_size": video_proc["patch_size"],
        "pooling_kernel_size": video_proc["pooling_kernel_size"],
        "scale": scale,
        "offset": offset,
    }

    # For unified models the video converter must use the unified image
    # converter so that per-frame patches are merged to the model patch
    # dimension (e.g. 48²×3 = 6912 instead of 16²×3 = 768).
    model_type = transformers_config.get("model_type", "gemma4")
    if model_type.startswith("gemma4_unified"):
        from keras_hub.src.models.gemma4_unified.gemma4_unified_image_converter import (  # noqa: E501
            Gemma4UnifiedImageConverter,
        )

        result["image_converter"] = Gemma4UnifiedImageConverter(
            patch_size=video_proc["patch_size"],
            max_soft_tokens=video_proc["max_soft_tokens"],
            pooling_kernel_size=video_proc["pooling_kernel_size"],
            scale=scale,
            offset=offset,
        )

    return result


def load_preprocessor_config(preset, transformers_config):
    """Return extra Gemma4CausalLMPreprocessor kwargs from processor_config."""
    processor_config = load_json(preset, "processor_config.json")

    result = {}

    # Audio feature size: needed for correct dummy audio_mel shapes.
    feature_extractor = processor_config.get("feature_extractor")
    if feature_extractor is not None:
        model_type = transformers_config.get("model_type", "gemma4")
        if model_type.startswith("gemma4_unified"):
            result["audio_input_feat_size"] = feature_extractor[
                "audio_samples_per_token"
            ]
        else:
            result["audio_input_feat_size"] = feature_extractor["feature_size"]

    if "video_processor" not in processor_config:
        return result

    video_proc = processor_config["video_processor"]
    # do_sample_frames=True means the processor samples num_frames from the
    # full video. KH's VideoConverter always samples; if this flag is ever
    # False, frames are expected to be pre-sampled by the caller and num_frames
    # is ignored at runtime (the converter still linspaces over whatever it
    # receives, but total_frames == num_frames so the result is identical).
    result.update(
        {
            "num_frames_per_video": video_proc["num_frames"],
            "num_vision_tokens_per_frame": video_proc["max_soft_tokens"],
            # video_fps is not stored in HF configs;
            # 24.0 matches the HF default.
            "video_fps": 24.0,
        }
    )
    return result


def convert_backbone_config(transformers_config):
    """Map a Transformers config dict → Gemma4Backbone keyword arguments."""
    model_type = transformers_config.get("model_type", "gemma4")
    is_text_only = model_type in ("gemma4_text", "gemma4_unified_text")

    if is_text_only:
        text_cfg = transformers_config
        vision_encoder = None
        audio_encoder = None
        image_size = None
    else:
        text_cfg = transformers_config["text_config"]

        if True:
            # --- Gemma4 Unified (encoder-free) ---
            image_size = None

            # Vision: lightweight projection (no ViT tower).
            if "vision_config" in transformers_config:
                vis_cfg = transformers_config["vision_config"]
                vision_encoder = Gemma4UnifiedVisionEmbedder(
                    hidden_dim=vis_cfg["mm_embed_dim"],
                    model_patch_size=vis_cfg["model_patch_size"],
                    mm_posemb_size=vis_cfg["mm_posemb_size"],
                    num_soft_tokens=vis_cfg["num_soft_tokens"],
                    pooling_kernel_size=vis_cfg["pooling_kernel_size"],
                    patch_size=vis_cfg["patch_size"],
                    layer_norm_epsilon=vis_cfg.get("rms_norm_eps", 1e-6),
                )
            else:
                vision_encoder = None

            # Audio: lightweight projection (no conformer).
            if "audio_config" in transformers_config:
                aud_cfg = transformers_config["audio_config"]
                audio_encoder = Gemma4UnifiedAudioEmbedder(
                    hidden_dim=text_cfg["hidden_size"],
                    audio_embed_dim=aud_cfg["audio_embed_dim"],
                    layer_norm_epsilon=aud_cfg.get("rms_norm_eps", 1e-6),
                )
            else:
                audio_encoder = None
        else:
            # --- Standard Gemma4 (tower-based) ---
            image_size = 896

            # Vision encoder.
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
                        "num_key_value_heads",
                        vis_cfg["num_attention_heads"],
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
                    use_clipped_linears=vis_cfg.get(
                        "use_clipped_linears", True
                    ),
                    standardize=vis_cfg.get("standardize", False),
                )
            else:
                vision_encoder = None

            # Audio encoder — key may be present but null.
            if transformers_config.get("audio_config") is not None:
                aud_cfg = transformers_config["audio_config"]
                audio_encoder = Gemma4AudioEncoder(
                    hidden_size=aud_cfg["hidden_size"],
                    num_heads=aud_cfg["num_attention_heads"],
                    num_layers=aud_cfg["num_hidden_layers"],
                    chunk_size=aud_cfg["attention_chunk_size"],
                    context_left=aud_cfg["attention_context_left"],
                    context_right=aud_cfg["attention_context_right"],
                    logit_cap=aud_cfg["attention_logit_cap"],
                    invalid_logit_value=aud_cfg[
                        "attention_invalid_logits_value"
                    ],
                    conv_kernel_size=aud_cfg["conv_kernel_size"],
                    residual_weight=aud_cfg["residual_weight"],
                    gradient_clipping=aud_cfg["gradient_clipping"],
                    sscp_conv_channels=tuple(
                        aud_cfg["subsampling_conv_channels"]
                    ),
                    output_proj_dims=aud_cfg["output_proj_dims"],
                    output_dim=text_cfg["hidden_size"],
                    norm_eps=aud_cfg["rms_norm_eps"],
                    sscp_norm_eps=aud_cfg["rms_norm_eps"],
                )
            else:
                audio_encoder = None

    # Config stores the pattern as "_sliding_window_pattern" (underscore
    # prefix).
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

    global_head_dim = text_cfg.get("global_head_dim", None)

    # Partial RoPE factor for global (full) attention layers.
    # Stored under rope_parameters["full_attention"]["partial_rotary_factor"].
    rope_params = text_cfg.get("rope_parameters", {}) or {}
    global_rope_partial_rotary_factor = rope_params.get(
        "full_attention", {}
    ).get("partial_rotary_factor")
    global_rope_theta = rope_params.get("full_attention", {}).get("rope_theta")

    attention_k_eq_v = text_cfg.get("attention_k_eq_v", False)
    if not attention_k_eq_v:
        raise ValueError(
            "Backbone only implements shared k==v on global layers."
        )

    # See HF _compute_proportional_rope_parameters.
    rope_type = rope_params.get("full_attention", {}).get(
        "rope_type", "default"
    )
    if rope_type not in ("proportional", "default"):
        raise ValueError(f"Unsupported rope_type: {rope_type}.")
    local_rope_theta = rope_params.get("sliding_attention", {}).get(
        "rope_theta"
    )

    # If it's missing in `full_attention`, safely fall back to top-level cfg.
    if global_rope_theta is None:
        global_rope_theta = text_cfg.get("rope_theta")

    # HF `use_bidirectional_attention` controls vision-token attention only:
    #   null     → purely causal for all tokens (E2B, E4B).
    #   "vision" → vision tokens attend bidirectionally within an image
    #              while text tokens remain causal (26B, 31B and larger).
    # This maps to KH `use_vision_bidirectional_attention` only.
    # KH `use_bidirectional_attention` (full bidirec for ALL tokens) is a
    # separate concept used exclusively for embedding models and is never
    # set from this HF field.
    hf_bidir = text_cfg.get("use_bidirectional_attention")
    use_vision_bidirectional_attention = hf_bidir == "vision"

    return {
        "vocabulary_size": text_cfg.get("vocab_size", 262144),
        "image_size": image_size,
        "num_layers": text_cfg["num_hidden_layers"],
        "num_query_heads": text_cfg.get("num_attention_heads", 8),
        "num_key_value_heads": text_cfg.get("num_key_value_heads", 1),
        "hidden_dim": text_cfg["hidden_size"],
        "intermediate_dim": text_cfg["intermediate_size"],
        "head_dim": text_cfg["head_dim"],
        "global_head_dim": global_head_dim,
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
        "audio_encoder": audio_encoder,
        "num_kv_shared_layers": text_cfg.get("num_kv_shared_layers", 0),
        "num_global_key_value_heads": text_cfg.get(
            "num_global_key_value_heads", None
        ),
        "hidden_size_per_layer_input": text_cfg.get(
            "hidden_size_per_layer_input"
        )
        or 0,
        "vocab_size_per_layer_input": text_cfg.get(
            "vocab_size_per_layer_input", None
        ),
        "global_rope_partial_rotary_factor": global_rope_partial_rotary_factor,
        "global_rope_wavelength": global_rope_theta,
        "local_rope_wavelength": local_rope_theta,
        "use_double_wide_mlp": text_cfg.get("use_double_wide_mlp", False),
        "enable_moe_block": text_cfg.get("enable_moe_block", False),
        "num_experts": text_cfg.get("num_experts", None),
        "expert_intermediate_dim": (
            text_cfg.get("moe_intermediate_size")
            or text_cfg.get("expert_intermediate_size")
        ),
        "num_experts_per_token": text_cfg.get("top_k_experts") or 8,
        "use_vision_bidirectional_attention": (
            use_vision_bidirectional_attention
        ),
    }


def convert_weights(backbone, loader, transformers_config):
    model_type = transformers_config.get("model_type", "gemma4")
    is_text_only = model_type in ("gemma4_text", "gemma4_unified_text")

    if is_text_only:
        text_prefix = _resolve_prefix(loader, ["model", "language_model", ""])
    else:
        text_prefix = _resolve_prefix(
            loader,
            ["model.language_model", "language_model"],
        )

    def hf_key(suffix):
        if text_prefix:
            return f"{text_prefix}.{suffix}"
        return suffix

    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key=hf_key("embed_tokens.weight"),
    )

    # Per-layer token conditioning (E4B / E2B models).

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
        _convert_unified_vision_embedder(
            vision_encoder, loader, transformers_config
        )

    audio_encoder = backbone.audio_encoder
    if audio_encoder is not None:
        _convert_unified_audio_embedder(
            audio_encoder, loader, transformers_config
        )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")
        _convert_decoder_block(decoder_layer, i, loader, hf_key)

    loader.port_weight(
        keras_variable=backbone.get_layer("final_normalization").scale,
        hf_weight_key=hf_key("norm.weight"),
    )

    return backbone


def _convert_unified_vision_embedder(
    vision_embedder, loader, transformers_config
):
    """Port unified vision-embedder weights from HF.

    The HF checkpoint stores vision embedder weights under two prefixes:
      - `model.vision_embedder` for the patch embedding + pos embedding
        layers (patch_ln1, patch_dense, patch_ln2, pos_embedding, pos_norm).
      - `model.embed_vision` for the multimodal projection
        (embedding_projection).
    The `embedding_pre_projection_norm` is parameter-free (VNorm).
    """
    vis_prefix = "model.vision_embedder"
    proj_prefix = "model.embed_vision"

    # --- patch_ln1 ---
    loader.port_weight(
        keras_variable=vision_embedder.get_layer("patch_ln1").gamma,
        hf_weight_key=f"{vis_prefix}.patch_ln1.weight",
    )
    loader.port_weight(
        keras_variable=vision_embedder.get_layer("patch_ln1").beta,
        hf_weight_key=f"{vis_prefix}.patch_ln1.bias",
    )

    # --- patch_dense (Linear with bias) ---
    loader.port_weight(
        keras_variable=vision_embedder.get_layer("patch_dense").kernel,
        hf_weight_key=f"{vis_prefix}.patch_dense.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )
    loader.port_weight(
        keras_variable=vision_embedder.get_layer("patch_dense").bias,
        hf_weight_key=f"{vis_prefix}.patch_dense.bias",
    )

    # --- patch_ln2 ---
    loader.port_weight(
        keras_variable=vision_embedder.get_layer("patch_ln2").gamma,
        hf_weight_key=f"{vis_prefix}.patch_ln2.weight",
    )
    loader.port_weight(
        keras_variable=vision_embedder.get_layer("patch_ln2").beta,
        hf_weight_key=f"{vis_prefix}.patch_ln2.bias",
    )

    # --- factorized pos_embedding: HF shape (mm_posemb_size, 2, hidden) ---
    # Split axis-0 (X) and axis-1 (Y) into our two Embedding layers.
    loader.port_weight(
        keras_variable=vision_embedder.get_layer("pos_embedding_x").embeddings,
        hf_weight_key=f"{vis_prefix}.pos_embedding",
        hook_fn=lambda x, _: x[:, 0, :],
    )
    loader.port_weight(
        keras_variable=vision_embedder.get_layer("pos_embedding_y").embeddings,
        hf_weight_key=f"{vis_prefix}.pos_embedding",
        hook_fn=lambda x, _: x[:, 1, :],
    )

    # --- pos_norm ---
    loader.port_weight(
        keras_variable=vision_embedder.get_layer("pos_norm").gamma,
        hf_weight_key=f"{vis_prefix}.pos_norm.weight",
    )
    loader.port_weight(
        keras_variable=vision_embedder.get_layer("pos_norm").beta,
        hf_weight_key=f"{vis_prefix}.pos_norm.bias",
    )

    # --- embedding_projection (multimodal embedder: RMSNorm → Linear) ---
    # embedding_pre_projection_norm: parameter-free (Gemma4VNorm).
    loader.port_weight(
        keras_variable=vision_embedder.get_layer("embedding_projection").kernel,
        hf_weight_key=f"{proj_prefix}.embedding_projection.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )


def _convert_unified_audio_embedder(
    audio_embedder, loader, transformers_config
):
    """Port unified audio-embedder weights from HF.

    The unified model uses `model.embed_audio` which is a
    Gemma4UnifiedMultimodalEmbedder with:
      - embedding_pre_projection_norm: parameter-free VNorm (no weights)
      - embedding_projection.weight  (Dense: [text_hidden, audio_embed_dim])
    """
    aud_prefix = "model.embed_audio"

    loader.port_weight(
        keras_variable=audio_embedder.embedding_projection.kernel,
        hf_weight_key=f"{aud_prefix}.embedding_projection.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )
    # embedding_pre_projection_norm: parameter-free (Gemma4VNorm).
