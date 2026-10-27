import numpy as np

from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionEncoder,
)
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = MuseGlimmerBackbone


def _get_normalization(processor_config):
    """Return the mean and std in pixel units, as the HF processor fuses them.

    The HF processor multiplies the float32 mean and std by
    `1 / rescale_factor`. Then it computes `(x - mean) / std` in float32.
    """
    to_pixel_units = np.float32(1.0 / processor_config["rescale_factor"])
    image_mean = [
        float(np.float32(value) * to_pixel_units)
        for value in processor_config["image_mean"]
    ]
    image_std = [
        float(np.float32(value) * to_pixel_units)
        for value in processor_config["image_std"]
    ]
    return image_mean, image_std


def load_image_converter_config(preset, transformers_config):
    """Load image converter settings from the HuggingFace processor config."""
    if "vision_config" not in transformers_config:
        return None

    processor_config = load_json(preset, "processor_config.json")[
        "image_processor"
    ]
    image_mean, image_std = _get_normalization(processor_config)
    return {
        "patch_size": processor_config["patch_size"],
        "patch_temporal": processor_config["temporal_patch_size"],
        "merge_size": processor_config["merge_size"],
        "max_image_tokens": processor_config["max_image_tokens"],
        "image_mean": image_mean,
        "image_std": image_std,
        "interpolation": "lanczos3",
        "antialias": True,
    }


def load_video_converter_config(preset, transformers_config):
    """Load video converter settings from the HuggingFace processor config."""
    if "vision_config" not in transformers_config:
        return None

    processor_config = load_json(preset, "processor_config.json")[
        "video_processor"
    ]
    image_mean, image_std = _get_normalization(processor_config)
    return {
        "patch_size": processor_config["patch_size"],
        "patch_temporal": processor_config["temporal_patch_size"],
        "merge_size": processor_config["merge_size"],
        "fps": processor_config["fps"],
        "num_frames": processor_config["num_frames"],
        "max_video_frame_tokens": processor_config["max_video_frame_tokens"],
        "image_mean": image_mean,
        "image_std": image_std,
        "interpolation": "lanczos3",
        "antialias": True,
    }


def _check_supported_options(text_config, vision_config, layer_types):
    """Raise for Hugging Face options that `MuseGlimmerBackbone` cannot run.

    The Keras layers use bias-free attention, a SiLU MLP, default RoPE, and
    RoPE on exactly the sliding-attention layers. Any other setting would
    load without an error and then produce different logits.
    """
    rope_type = (text_config.get("rope_parameters") or {}).get(
        "rope_type", "default"
    )
    if rope_type != "default":
        raise ValueError(
            f"`rope_type={rope_type!r}` is not supported. "
            "`MuseGlimmerBackbone` only implements the default RoPE."
        )
    if text_config.get("attention_bias", False):
        raise ValueError(
            "`attention_bias=True` is not supported. The attention "
            "projections of `MuseGlimmerBackbone` have no bias."
        )
    activation = text_config.get("hidden_activation", "silu")
    if activation != "silu":
        raise ValueError(
            f"`hidden_activation={activation!r}` is not supported. "
            "`MuseGlimmerBackbone` always uses `'silu'`."
        )
    # Hugging Face turns RoPE off on the layers where `layer_rope_theta` is
    # 0, independent of `layer_types`. Any nonzero value turns it on, because
    # all layers share one rotary embedding.
    num_layers = len(layer_types)
    layer_rope_theta = text_config.get("layer_rope_theta")
    if layer_rope_theta is None:
        layer_rope_theta = [
            0 if (num_layers - 1 - i) % 4 == 0 else 1 for i in range(num_layers)
        ]
    expected = [layer_type != "full_attention" for layer_type in layer_types]
    if [bool(theta) for theta in layer_rope_theta] != expected:
        raise ValueError(
            "`layer_rope_theta` must be 0 on every `'full_attention'` layer "
            "and nonzero on every other layer. Received: "
            f"layer_types={layer_types}, layer_rope_theta={layer_rope_theta}."
        )
    if vision_config is not None:
        hidden_act = vision_config.get("hidden_act", "gelu")
        if hidden_act != "gelu":
            raise ValueError(
                "`vision_config.hidden_act` must be `'gelu'`. Received: "
                f"{hidden_act!r}."
            )


def convert_backbone_config(transformers_config, cls, **kwargs):
    text_config = transformers_config["text_config"]
    vision_config = transformers_config.get("vision_config", None)
    dtype = kwargs.pop("dtype", None)

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

    _check_supported_options(text_config, vision_config, layer_types)

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
            dtype=dtype,
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
        "dtype": dtype,
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

    # LM head. Hugging Face reads `tie_word_embeddings` from the top-level
    # config. A tied checkpoint has no `lm_head.weight` tensor, so the
    # untied Keras head is filled from the input embedding instead.
    tied = transformers_config.get("tie_word_embeddings", False)
    loader.port_weight(
        backbone.token_embedding.reverse_embeddings,
        "model.language_model.embed_tokens.weight"
        if tied
        else "lm_head.weight",
        hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
    )

    for i in range(backbone.num_layers):
        layer = backbone.transformer_layers[i]
        attn = layer._self_attention_layer

        loader.port_weight(
            layer._input_layernorm.scale,
            f"model.language_model.layers.{i}.input_layernorm.weight",
        )
        loader.port_weight(
            layer._post_attention_layernorm.scale,
            f"model.language_model.layers.{i}.post_attention_layernorm.weight",
        )
        loader.port_weight(
            layer._pre_feedforward_layernorm.scale,
            f"model.language_model.layers.{i}.pre_feedforward_layernorm.weight",
        )
        loader.port_weight(
            layer._post_feedforward_layernorm.scale,
            f"model.language_model.layers.{i}"
            ".post_feedforward_layernorm.weight",
        )

        loader.port_weight(
            attn._query_dense.kernel,
            f"model.language_model.layers.{i}.self_attn.q_proj.weight",
            hook_fn=lambda x, shape: np.transpose(x, axes=(1, 0)).reshape(
                shape
            ),
        )
        loader.port_weight(
            attn._key_dense.kernel,
            f"model.language_model.layers.{i}.self_attn.k_proj.weight",
            hook_fn=lambda x, shape: np.transpose(x, axes=(1, 0)).reshape(
                shape
            ),
        )
        loader.port_weight(
            attn._value_dense.kernel,
            f"model.language_model.layers.{i}.self_attn.v_proj.weight",
            hook_fn=lambda x, shape: np.transpose(x, axes=(1, 0)).reshape(
                shape
            ),
        )
        loader.port_weight(
            attn._gate_dense.kernel,
            f"model.language_model.layers.{i}.self_attn.gate_proj.weight",
            hook_fn=lambda x, shape: np.transpose(x, axes=(1, 0)).reshape(
                shape
            ),
        )
        loader.port_weight(
            attn._output_dense.kernel,
            f"model.language_model.layers.{i}.self_attn.o_proj.weight",
            hook_fn=lambda x, shape: np.transpose(x, axes=(1, 0)).reshape(
                shape
            ),
        )

        loader.port_weight(
            layer._feedforward_gate_dense.kernel,
            f"model.language_model.layers.{i}.mlp.gate_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
        )
        loader.port_weight(
            layer._feedforward_up_dense.kernel,
            f"model.language_model.layers.{i}.mlp.up_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
        )
        loader.port_weight(
            layer._feedforward_down_dense.kernel,
            f"model.language_model.layers.{i}.mlp.down_proj.weight",
            hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
        )

    loader.port_weight(
        backbone.layer_norm.scale, "model.language_model.norm.weight"
    )

    if backbone.vision_encoder is not None:
        vis = backbone.vision_encoder

        loader.port_weight(
            vis.patch_embedder.patch_embedding.kernel,
            "model.vision_tower.patch_embedder.patch_embedding.weight",
            hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
        )
        loader.port_weight(
            vis.patch_embedder.position_embedding_table.embeddings,
            "model.vision_tower.patch_embedder.position_embedding_table.weight",
        )
        loader.port_weight(vis.ln_pre.gamma, "model.vision_tower.ln_pre.weight")
        loader.port_weight(vis.ln_pre.beta, "model.vision_tower.ln_pre.bias")
        loader.port_weight(
            vis.ln_post.gamma, "model.vision_tower.ln_post.weight"
        )
        loader.port_weight(vis.ln_post.beta, "model.vision_tower.ln_post.bias")

        for i in range(vis.num_layers):
            blk = vis.blocks[i]
            blk_prefix = f"model.vision_tower.layers.{i}"

            loader.port_weight(blk.norm1.gamma, f"{blk_prefix}.norm1.weight")
            loader.port_weight(blk.norm1.beta, f"{blk_prefix}.norm1.bias")
            loader.port_weight(blk.norm2.gamma, f"{blk_prefix}.norm2.weight")
            loader.port_weight(blk.norm2.beta, f"{blk_prefix}.norm2.bias")

            loader.port_weight(
                blk.attn.q_proj.kernel,
                f"{blk_prefix}.attn.q_proj.weight",
                hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
            )
            loader.port_weight(
                blk.attn.q_proj.bias, f"{blk_prefix}.attn.q_proj.bias"
            )
            loader.port_weight(
                blk.attn.k_proj.kernel,
                f"{blk_prefix}.attn.k_proj.weight",
                hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
            )
            loader.port_weight(
                blk.attn.k_proj.bias, f"{blk_prefix}.attn.k_proj.bias"
            )
            loader.port_weight(
                blk.attn.v_proj.kernel,
                f"{blk_prefix}.attn.v_proj.weight",
                hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
            )
            loader.port_weight(
                blk.attn.v_proj.bias, f"{blk_prefix}.attn.v_proj.bias"
            )
            loader.port_weight(
                blk.attn.proj.kernel,
                f"{blk_prefix}.attn.proj.weight",
                hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
            )
            loader.port_weight(
                blk.attn.proj.bias, f"{blk_prefix}.attn.proj.bias"
            )

            loader.port_weight(
                blk.mlp.fc1.kernel,
                f"{blk_prefix}.mlp.fc1.weight",
                hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
            )
            loader.port_weight(blk.mlp.fc1.bias, f"{blk_prefix}.mlp.fc1.bias")
            loader.port_weight(
                blk.mlp.fc2.kernel,
                f"{blk_prefix}.mlp.fc2.weight",
                hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
            )
            loader.port_weight(blk.mlp.fc2.bias, f"{blk_prefix}.mlp.fc2.bias")

        # Multimodal fusion: adapter (double-activation, no bias) + a
        # separate projection Linear + a final scaleless RMSNorm.
        loader.port_weight(
            backbone.vision_adapter_fc1.kernel,
            "model.vision_adapter.fc1.weight",
            hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
        )
        loader.port_weight(
            backbone.vision_adapter_fc2.kernel,
            "model.vision_adapter.fc2.weight",
            hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
        )
        loader.port_weight(
            backbone.vision_projection.kernel,
            "model.vision_projection.weight",
            hook_fn=lambda x, _: np.transpose(x, axes=(1, 0)),
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
