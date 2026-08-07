import numpy as np

from keras_hub.src.models.diffusion_gemma.diffusion_gemma_backbone import (
    DiffusionGemmaBackbone,
)
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoder,
)
from keras_hub.src.samplers.entropy_bound_sampler import EntropyBoundSampler
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers.convert_gemma4 import (
    _convert_decoder_block_weights,
)
from keras_hub.src.utils.transformers.convert_gemma4 import (
    convert_tokenizer as target_convert_tokenizer,
)
from keras_hub.src.utils.transformers.convert_gemma4 import (
    load_image_converter_config as target_load_image_converter_config,
)


def convert_tokenizer(cls, preset, **kwargs):
    return target_convert_tokenizer(cls, preset, **kwargs)


def load_image_converter_config(preset, transformers_config):
    return target_load_image_converter_config(preset, transformers_config)


backbone_cls = DiffusionGemmaBackbone


def convert_backbone_config(transformers_config):
    """Map a DiffusionGemma Transformers config → DiffusionGemmaBackbone
    kwargs."""
    model_type = transformers_config.get("model_type", "diffusion_gemma")
    is_text_only = model_type == "diffusion_gemma_text"

    if is_text_only:
        text_cfg = transformers_config
        vision_encoder = None
        image_size = None
    else:
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
        "num_global_key_value_heads": text_cfg.get(
            "num_global_key_value_heads", None
        ),
        "global_rope_partial_rotary_factor": global_rope_partial_rotary_factor,
        "global_rope_wavelength": global_rope_theta,
        "local_rope_wavelength": local_rope_theta,
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
    }


def convert_task_config(transformers_config):
    """Map DiffusionGemma config keys → DiffusionGemmaBlockDiffusionLM
    kwargs."""
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


def _convert_decoder_block(decoder_layer, layer_idx, loader, hf_key_fn):
    """Port a single DiffusionGemmaTransformerLayer from HF."""
    layer_prefix = f"layers.{layer_idx}"

    def layer_key(attr):
        return hf_key_fn(f"{layer_prefix}.{attr}")

    # Layer norms
    loader.port_weight(
        keras_variable=decoder_layer.pre_attention_norm.scale,
        hf_weight_key=layer_key("input_layernorm.weight"),
    )
    loader.port_weight(
        keras_variable=decoder_layer.post_attention_norm.scale,
        hf_weight_key=layer_key("post_attention_layernorm.weight"),
    )
    loader.port_weight(
        keras_variable=decoder_layer.pre_ffw_norm.scale,
        hf_weight_key=layer_key("pre_feedforward_layernorm.weight"),
    )
    loader.port_weight(
        keras_variable=decoder_layer.post_ffw_norm.scale,
        hf_weight_key=layer_key("post_feedforward_layernorm.weight"),
    )

    # Attention Q / K / V / O + Q-norm / K-norm
    loader.port_weight(
        keras_variable=decoder_layer.attention.query_dense.kernel,
        hf_weight_key=layer_key("self_attn.q_proj.weight"),
        # HF: [num_q_heads * head_dim, hidden]
        # → Keras: [num_q_heads, hidden, head_dim]
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor,
                (keras_shape[0], keras_shape[2], keras_shape[1]),
            ),
            axes=(0, 2, 1),
        ),
    )
    loader.port_weight(
        keras_variable=decoder_layer.attention.query_norm.scale,
        hf_weight_key=layer_key("self_attn.q_norm.weight"),
    )
    loader.port_weight(
        keras_variable=decoder_layer.attention.key_dense.kernel,
        hf_weight_key=layer_key("self_attn.k_proj.weight"),
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor,
                (keras_shape[0], keras_shape[2], keras_shape[1]),
            ),
            axes=(0, 2, 1),
        ),
    )
    loader.port_weight(
        keras_variable=decoder_layer.attention.key_norm.scale,
        hf_weight_key=layer_key("self_attn.k_norm.weight"),
    )
    # v_proj is absent on global-attention layers when
    # attention_k_eq_v=True: value reuses the key projection, so
    # value_dense=None.
    if decoder_layer.attention.value_dense is not None:
        loader.port_weight(
            keras_variable=decoder_layer.attention.value_dense.kernel,
            hf_weight_key=layer_key("self_attn.v_proj.weight"),
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
    # v_norm (Gemma4VNorm) is parameter-free — no weight to port.
    loader.port_weight(
        keras_variable=decoder_layer.attention.output_dense.kernel,
        hf_weight_key=layer_key("self_attn.o_proj.weight"),
        # HF: [hidden, num_q_heads * head_dim]
        # → Keras: [num_q_heads, head_dim, hidden]
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor,
                (keras_shape[2], keras_shape[0], keras_shape[1]),
            ),
            axes=(1, 2, 0),
        ),
    )

    loader.port_weight(
        keras_variable=decoder_layer.gating_ffw.kernel,
        hf_weight_key=layer_key("mlp.gate_proj.weight"),
        hook_fn=lambda x, _: np.transpose(x),
    )
    loader.port_weight(
        keras_variable=decoder_layer.gating_ffw_2.kernel,
        hf_weight_key=layer_key("mlp.up_proj.weight"),
        hook_fn=lambda x, _: np.transpose(x),
    )
    loader.port_weight(
        keras_variable=decoder_layer.ffw_linear.kernel,
        hf_weight_key=layer_key("mlp.down_proj.weight"),
        hook_fn=lambda x, _: np.transpose(x),
    )

    # MoE block (parallel dense + expert paths).
    if decoder_layer.enable_moe_block:
        # Extra norms.
        loader.port_weight(
            keras_variable=decoder_layer.post_ffw_norm_dense.scale,
            hf_weight_key=layer_key("post_feedforward_layernorm_1.weight"),
        )
        loader.port_weight(
            keras_variable=decoder_layer.pre_ffw_norm_moe.scale,
            hf_weight_key=layer_key("pre_feedforward_layernorm_2.weight"),
        )
        loader.port_weight(
            keras_variable=decoder_layer.post_ffw_norm_moe_path.scale,
            hf_weight_key=layer_key("post_feedforward_layernorm_2.weight"),
        )
        # Router: per-dim scale + projection (rms_norm has no learnable
        # weights).
        loader.port_weight(
            keras_variable=decoder_layer.moe_router.per_dim_scale,
            hf_weight_key=layer_key("router.scale"),
        )
        loader.port_weight(
            keras_variable=decoder_layer.moe_router.proj.kernel,
            hf_weight_key=layer_key("router.proj.weight"),
            hook_fn=lambda x, _: np.transpose(x),
        )
        # Expert bank: HF `gate_up_proj` is [E, 2*I, H], `down_proj` is [E, H,
        # I].
        # Keras Hub `gate` / `up` are [E, H, I], `down` is [E, I, H].
        I = decoder_layer.expert_intermediate_dim
        loader.port_weight(
            keras_variable=decoder_layer.moe_expert_bank.gate_proj,
            hf_weight_key=layer_key("experts.gate_up_proj"),
            hook_fn=lambda x, _: np.transpose(x[:, :I, :], axes=(0, 2, 1)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.moe_expert_bank.up_proj,
            hf_weight_key=layer_key("experts.gate_up_proj"),
            hook_fn=lambda x, _: np.transpose(x[:, I:, :], axes=(0, 2, 1)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.moe_expert_bank.down_proj,
            hf_weight_key=layer_key("experts.down_proj"),
            hook_fn=lambda x, _: np.transpose(x, axes=(0, 2, 1)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.moe_expert_bank.per_expert_scale,
            hf_weight_key=layer_key("router.per_expert_scale"),
        )

    # layer_scalar — present on all text decoder layers (HF Buffer).
    loader.port_weight(
        keras_variable=decoder_layer.layer_scalar,
        hf_weight_key=layer_key("layer_scalar"),
        hook_fn=lambda x, _: np.squeeze(x),
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

    vision_encoder = backbone.vision_encoder
    if vision_encoder is not None:
        _convert_vision_encoder(vision_encoder, loader, transformers_config)

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")
        _convert_decoder_block(decoder_layer, i, loader, hf_key)

    # Port encoder-pass per-layer scalars.
    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")
        loader.port_weight(
            keras_variable=decoder_layer.encoder_layer_scalar,
            hf_weight_key=(
                f"model.encoder.language_model.layers.{i}.layer_scalar"
            ),
            hook_fn=lambda x, _: np.squeeze(x),
        )

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
    """Read generation_config.json and return DiffusionGemmaBlockDiffusionLM
    kwargs."""
    if not check_file_exists(preset, "generation_config.json"):
        return {}
    gen_cfg = load_json(preset, "generation_config.json")
    kwargs = {}

    sampler_config = gen_cfg.get("sampler_config", {})
    sampler_keys = {
        "confidence_threshold",
        "sampler_config",
        "stability_threshold",
    }
    if sampler_keys.intersection(gen_cfg):
        kwargs["sampler"] = EntropyBoundSampler(
            entropy_bound=sampler_config.get("entropy_bound", 0.1),
            confidence_threshold=gen_cfg.get("confidence_threshold", 0.005),
            stability_threshold=gen_cfg.get("stability_threshold", 1),
        )
    if "max_denoising_steps" in gen_cfg:
        kwargs["max_denoising_steps"] = gen_cfg["max_denoising_steps"]
    if "t_min" in gen_cfg:
        kwargs["t_min"] = gen_cfg["t_min"]
    if "t_max" in gen_cfg:
        kwargs["t_max"] = gen_cfg["t_max"]
    if "eos_token_id" in gen_cfg:
        stop_token_ids = gen_cfg["eos_token_id"]
        if not isinstance(stop_token_ids, list):
            stop_token_ids = [stop_token_ids]
        kwargs["stop_token_ids"] = tuple(stop_token_ids)
    if "pad_token_id" in gen_cfg:
        kwargs["pad_token_id"] = gen_cfg["pad_token_id"]
    return kwargs


def load_preprocessor_config(preset, transformers_config):
    """Return extra DiffusionGemmaBlockDiffusionLMPreprocessor kwargs."""
    return {
        "add_start_token": False,
        "add_end_token": False,
    }
