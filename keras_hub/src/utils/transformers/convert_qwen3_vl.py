import numpy as np

from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_vision_encoder import (
    Qwen3VLVisionEncoder,
)
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers.convert_qwen3_5 import (
    _compute_scale_offset,
)
from keras_hub.src.utils.transformers.convert_qwen3_5 import (
    _transpose_and_reshape,
)
from keras_hub.src.utils.transformers.convert_qwen3_5 import convert_tokenizer  # noqa: F401

backbone_cls = Qwen3VLBackbone

_VISION_PREFIX = "model.visual"
_TEXT_PREFIX = "model.language_model"

# PIL resampling codes used by HF processor configs.
_RESAMPLE = {0: "nearest", 2: "bilinear", 3: "bicubic"}


def _converter_config(processor_config, vision_config):
    """Kwargs shared by the image and video converters."""
    scale, offset = _compute_scale_offset(
        processor_config["image_mean"],
        processor_config["image_std"],
        processor_config.get("rescale_factor", 1.0 / 255),
    )
    resample = processor_config.get("resample", 3)
    if resample not in _RESAMPLE:
        raise ValueError(
            f"Unsupported `resample={resample}`. Supported values are "
            f"{sorted(_RESAMPLE)}."
        )
    size = processor_config["size"]
    return {
        "patch_size": processor_config.get(
            "patch_size", vision_config["patch_size"]
        ),
        "temporal_patch_size": processor_config.get(
            "temporal_patch_size", vision_config["temporal_patch_size"]
        ),
        "spatial_merge_size": processor_config.get(
            "merge_size", vision_config["spatial_merge_size"]
        ),
        "min_pixels": size["shortest_edge"],
        "max_pixels": size["longest_edge"],
        "scale": scale,
        "offset": offset,
        "interpolation": _RESAMPLE[resample],
        "antialias": True,
    }


def load_image_converter_config(preset, transformers_config):
    """Return kwargs for `Qwen3VLImageConverter`."""
    return _converter_config(
        load_json(preset, "preprocessor_config.json"),
        transformers_config["vision_config"],
    )


def load_video_converter_config(preset, transformers_config):
    """Return kwargs for `Qwen3VLVideoConverter`."""
    video_config = load_json(preset, "video_preprocessor_config.json")
    config = _converter_config(
        video_config, transformers_config["vision_config"]
    )
    for key in ("fps", "min_frames", "max_frames"):
        if video_config.get(key) is not None:
            config[key] = video_config[key]
    return config


def _transpose(x, _):
    return np.transpose(x)


def _patch_embed_kernel(x, _):
    # HF Conv3d (out, in, t, h, w) -> Keras (t, h, w, in, out).
    return np.transpose(x, (2, 3, 4, 1, 0))


def _rope_parameters(text_config):
    # transformers 4.x stores `rope_scaling`, 5.x stores `rope_parameters`.
    if "rope_parameters" in text_config:
        return text_config["rope_parameters"]
    return text_config["rope_scaling"]


def convert_vision_encoder_config(transformers_config):
    """Return the `Qwen3VLVisionEncoder` kwargs for an HF config."""
    vision_config = transformers_config["vision_config"]
    if vision_config["hidden_act"] != "gelu_pytorch_tanh":
        raise ValueError(
            "Only `hidden_act='gelu_pytorch_tanh'` is supported for the "
            f"vision tower. Received: {vision_config['hidden_act']}"
        )
    return {
        "depth": vision_config["depth"],
        "hidden_size": vision_config["hidden_size"],
        "num_heads": vision_config["num_heads"],
        "intermediate_size": vision_config["intermediate_size"],
        "out_hidden_size": vision_config["out_hidden_size"],
        "patch_size": vision_config["patch_size"],
        "temporal_patch_size": vision_config["temporal_patch_size"],
        "in_channels": vision_config["in_channels"],
        "spatial_merge_size": vision_config["spatial_merge_size"],
        "num_position_embeddings": vision_config["num_position_embeddings"],
        "deepstack_visual_indexes": vision_config["deepstack_visual_indexes"],
    }


def convert_text_config(transformers_config):
    """Return the `Qwen3VLBackbone` kwargs, without the vision encoder."""
    text_config = transformers_config["text_config"]
    if text_config["hidden_act"] != "silu":
        raise ValueError(
            "Only `hidden_act='silu'` is supported for the text model. "
            f"Received: {text_config['hidden_act']}"
        )
    if text_config["attention_bias"]:
        raise ValueError("`attention_bias=True` is not supported.")

    rope_params = _rope_parameters(text_config)
    if rope_params.get("mrope_interleaved") is False:
        raise ValueError(
            "Only interleaved M-RoPE is supported. Received: "
            "mrope_interleaved=False"
        )
    rope_type = rope_params["rope_type"]
    if rope_type != "default":
        raise ValueError(
            f"Only `rope_type='default'` is supported. Received: {rope_type}"
        )
    rope_theta = rope_params.get("rope_theta")
    if rope_theta is None:
        rope_theta = text_config["rope_theta"]
    return {
        "vocabulary_size": text_config["vocab_size"],
        "num_layers": text_config["num_hidden_layers"],
        "num_query_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "head_dim": text_config["head_dim"],
        "hidden_dim": text_config["hidden_size"],
        "intermediate_dim": text_config["intermediate_size"],
        "rope_max_wavelength": rope_theta,
        "layer_norm_epsilon": text_config["rms_norm_eps"],
        "tie_word_embeddings": transformers_config["tie_word_embeddings"],
        "mrope_section": rope_params["mrope_section"],
        "dropout": text_config["attention_dropout"],
    }


def convert_backbone_config(
    transformers_config, cls=None, dtype=None, **kwargs
):
    vision_config = transformers_config["vision_config"]
    text_config = transformers_config["text_config"]
    if vision_config["out_hidden_size"] != text_config["hidden_size"]:
        raise ValueError(
            "`vision_config.out_hidden_size` must equal "
            "`text_config.hidden_size`. Received: "
            f"{vision_config['out_hidden_size']} and "
            f"{text_config['hidden_size']}"
        )
    config = convert_text_config(transformers_config)
    # The loader passes `dtype` here instead of to the backbone.
    config["dtype"] = dtype
    config["vision_encoder"] = Qwen3VLVisionEncoder(
        **convert_vision_encoder_config(transformers_config), dtype=dtype
    )
    return config


def expected_hf_keys(
    num_layers, vision_depth, num_deepstack, tie_word_embeddings
):
    """Return every HF safetensors key a Qwen3-VL checkpoint must contain.

    Pass `vision_depth=None` for a backbone without a vision encoder.
    """
    keys = {
        f"{_TEXT_PREFIX}.embed_tokens.weight",
        f"{_TEXT_PREFIX}.norm.weight",
    }
    if not tie_word_embeddings:
        keys.add("lm_head.weight")
    for i in range(num_layers):
        p = f"{_TEXT_PREFIX}.layers.{i}"
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            keys.add(f"{p}.{name}.weight")
    if vision_depth is None:
        return keys

    v = _VISION_PREFIX
    keys.update(
        {
            f"{v}.patch_embed.proj.weight",
            f"{v}.patch_embed.proj.bias",
            f"{v}.pos_embed.weight",
        }
    )
    for i in range(vision_depth):
        for name in (
            "norm1",
            "norm2",
            "attn.qkv",
            "attn.proj",
            "mlp.linear_fc1",
            "mlp.linear_fc2",
        ):
            keys.add(f"{v}.blocks.{i}.{name}.weight")
            keys.add(f"{v}.blocks.{i}.{name}.bias")
    mergers = [f"{v}.merger"] + [
        f"{v}.deepstack_merger_list.{i}" for i in range(num_deepstack)
    ]
    for m in mergers:
        for name in ("norm", "linear_fc1", "linear_fc2"):
            keys.add(f"{m}.{name}.weight")
            keys.add(f"{m}.{name}.bias")
    return keys


def _checkpoint_keys(loader):
    if loader.safetensor_config is not None:
        return set(loader.safetensor_config["weight_map"])
    # Porting opened the single safetensors file, which cached its keys.
    return set().union(*loader.safetensor_keys.values())


def _port_linear(loader, ported, dense, hf_prefix):
    for var, suffix, hook in (
        (dense.kernel, "weight", _transpose),
        (dense.bias, "bias", None),
    ):
        key = f"{hf_prefix}.{suffix}"
        loader.port_weight(var, key, hook_fn=hook)
        ported.add(key)


def _port_norm(loader, ported, norm, hf_prefix):
    for var, suffix in ((norm.gamma, "weight"), (norm.beta, "bias")):
        key = f"{hf_prefix}.{suffix}"
        loader.port_weight(var, key)
        ported.add(key)


def _port_merger(loader, ported, merger, hf_prefix):
    _port_norm(loader, ported, merger.norm, f"{hf_prefix}.norm")
    _port_linear(loader, ported, merger.linear_fc1, f"{hf_prefix}.linear_fc1")
    _port_linear(loader, ported, merger.linear_fc2, f"{hf_prefix}.linear_fc2")


def convert_weights(backbone, loader, transformers_config):
    ported = set()
    convert_text_weights(backbone, loader, ported)
    if backbone.vision_encoder is not None:
        convert_vision_weights(backbone.vision_encoder, loader, ported)
    _check_all_keys_accounted_for(backbone, loader, ported)
    return backbone


def convert_text_weights(backbone, loader, ported=None):
    """Port the text decoder weights. The vision encoder is left untouched.

    Works on a text-only backbone too. Every ported HF key is added to
    `ported` when it is given.
    """
    ported = set() if ported is None else ported

    def port(var, key, hook_fn=None):
        loader.port_weight(var, key, hook_fn=hook_fn)
        ported.add(key)

    embedding = backbone.get_layer("token_embedding")
    port(embedding.embeddings, f"{_TEXT_PREFIX}.embed_tokens.weight")
    if not backbone.tie_word_embeddings:
        port(embedding.reverse_embeddings, "lm_head.weight", hook_fn=_transpose)

    for i in range(backbone.num_layers):
        layer = backbone.get_layer(f"transformer_layer_{i}")
        attn = layer._self_attention_layer
        p = f"{_TEXT_PREFIX}.layers.{i}"
        port(
            layer._self_attention_layernorm.scale,
            f"{p}.input_layernorm.weight",
        )
        port(
            layer._feedforward_layernorm.scale,
            f"{p}.post_attention_layernorm.weight",
        )
        for dense, name in (
            (attn._query_dense, "q_proj"),
            (attn._key_dense, "k_proj"),
            (attn._value_dense, "v_proj"),
            (attn._output_dense, "o_proj"),
        ):
            port(
                dense.kernel,
                f"{p}.self_attn.{name}.weight",
                hook_fn=_transpose_and_reshape,
            )
        port(
            attn._query_norm.scale,
            f"{p}.self_attn.q_norm.weight",
        )
        port(
            attn._key_norm.scale,
            f"{p}.self_attn.k_norm.weight",
        )
        for dense, name in (
            (layer._feedforward_gate_dense, "gate_proj"),
            (layer._feedforward_intermediate_dense, "up_proj"),
            (layer._feedforward_output_dense, "down_proj"),
        ):
            port(dense.kernel, f"{p}.mlp.{name}.weight", hook_fn=_transpose)

    port(
        backbone.get_layer("sequence_output_layernorm").scale,
        f"{_TEXT_PREFIX}.norm.weight",
    )
    return ported


def convert_vision_weights(vision_encoder, loader, ported=None):
    """Port the weights of a `Qwen3VLVisionEncoder`.

    Every ported HF key is added to `ported` when it is given.
    """
    ported = set() if ported is None else ported

    def port(var, key, hook_fn=None):
        loader.port_weight(var, key, hook_fn=hook_fn)
        ported.add(key)

    vis = vision_encoder
    v = _VISION_PREFIX
    port(
        vis.patch_embed.proj.kernel,
        f"{v}.patch_embed.proj.weight",
        hook_fn=_patch_embed_kernel,
    )
    port(vis.patch_embed.proj.bias, f"{v}.patch_embed.proj.bias")
    port(vis.pos_embed.embeddings, f"{v}.pos_embed.weight")
    for i, block in enumerate(vis.blocks):
        b = f"{v}.blocks.{i}"
        _port_norm(loader, ported, block.norm1, f"{b}.norm1")
        _port_norm(loader, ported, block.norm2, f"{b}.norm2")
        _port_linear(loader, ported, block.attn.qkv, f"{b}.attn.qkv")
        _port_linear(loader, ported, block.attn.proj, f"{b}.attn.proj")
        _port_linear(loader, ported, block.mlp.fc1, f"{b}.mlp.linear_fc1")
        _port_linear(loader, ported, block.mlp.fc2, f"{b}.mlp.linear_fc2")
    _port_merger(loader, ported, vis.merger, f"{v}.merger")
    for i, merger in enumerate(vis.deepstack_merger_list):
        _port_merger(loader, ported, merger, f"{v}.deepstack_merger_list.{i}")
    return ported


def _check_all_keys_accounted_for(backbone, loader, ported):
    vision_encoder = backbone.vision_encoder
    expected = expected_hf_keys(
        backbone.num_layers,
        vision_encoder.depth if vision_encoder is not None else None,
        len(vision_encoder.deepstack_visual_indexes)
        if vision_encoder is not None
        else 0,
        backbone.tie_word_embeddings,
    )
    if ported != expected:
        raise ValueError(
            "Ported keys differ from the expected Qwen3-VL key set. "
            f"Missing: {sorted(expected - ported)}, "
            f"unexpected: {sorted(ported - expected)}"
        )
    # A tied checkpoint may still carry a redundant `lm_head.weight`.
    skipped = {"lm_head.weight"} if backbone.tie_word_embeddings else set()
    unaccounted = _checkpoint_keys(loader) - ported - skipped
    if vision_encoder is None:
        # A text-only backbone leaves the vision tower out on purpose.
        unaccounted = {
            k for k in unaccounted if not k.startswith(f"{_VISION_PREFIX}.")
        }
    if unaccounted:
        raise ValueError(
            "HF checkpoint keys neither ported nor skipped: "
            f"{sorted(unaccounted)}"
        )
