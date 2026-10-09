import json
import os

import numpy as np

from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers import convert_gemma4

backbone_cls = EmbeddingGemma2Backbone


def convert_backbone_config(transformers_config):
    text_cfg = transformers_config["text_config"]
    image_size = 896
    vision_encoder = None
    if "vision_config" in transformers_config:
        vision_encoder = _build_vision_encoder(
            transformers_config["vision_config"], text_cfg, image_size
        )
    audio_encoder = None
    if "audio_config" in transformers_config:
        audio_encoder = _build_audio_encoder(
            transformers_config["audio_config"], text_cfg
        )
    per_layer_config = text_cfg.get("per_layer_config", {})
    global_head_dim = 512
    num_global_kv = 1
    # HF keys `per_layer_config` by layer index; the backbone takes a single
    # global head_dim / KV-head count, so every entry must agree.
    global_entries = [
        v for v in per_layer_config.values() if isinstance(v, dict)
    ]
    if global_entries:
        first = global_entries[0]
        for entry in global_entries[1:]:
            if entry.get("head_dim") != first.get("head_dim") or entry.get(
                "num_key_value_heads"
            ) != first.get("num_key_value_heads"):
                raise ValueError(
                    "`per_layer_config` entries disagree on `head_dim` or "
                    f"`num_key_value_heads`: {per_layer_config}"
                )
        global_head_dim = first.get("head_dim", global_head_dim)
        num_global_kv = first.get("num_key_value_heads", num_global_kv)
    rope = text_cfg["rope_parameters"]
    return {
        "vocabulary_size": text_cfg["vocab_size"],
        "num_layers": text_cfg["num_hidden_layers"],
        "num_query_heads": text_cfg["num_attention_heads"],
        "num_key_value_heads": text_cfg["num_key_value_heads"],
        "hidden_dim": text_cfg["hidden_size"],
        "intermediate_dim": text_cfg["intermediate_size"],
        "head_dim": text_cfg["head_dim"],
        "global_head_dim": global_head_dim,
        "num_global_key_value_heads": num_global_kv,
        "sliding_window_size": text_cfg["sliding_window"],
        "layer_types": text_cfg["layer_types"],
        "hidden_size_per_layer_input": text_cfg["hidden_size_per_layer_input"],
        "embedding_dim": text_cfg["embedding_dim"],
        "layer_norm_epsilon": text_cfg["rms_norm_eps"],
        "dropout": text_cfg["attention_dropout"],
        "vision_encoder": vision_encoder,
        "audio_encoder": audio_encoder,
        "image_size": image_size,
        "local_rope_wavelength": rope["sliding_attention"]["rope_theta"],
        "global_rope_wavelength": rope["full_attention"]["rope_theta"],
        "global_rope_partial_rotary_factor": rope["full_attention"].get(
            "partial_rotary_factor", 1.0
        ),
        "num_audio_tokens_per_clip": transformers_config.get(
            "audio_soft_tokens_per_image"
        ),
        "use_bidirectional_attention": True,
    }


def convert_weights(backbone, loader, transformers_config):
    p = loader.port_weight

    p(
        backbone.get_layer("token_embedding").embeddings,
        "language_model.embed_tokens.weight",
    )
    if backbone.hidden_size_per_layer_input > 0:
        p(
            keras_variable=backbone.get_layer(
                "per_layer_model_projection"
            ).kernel,
            hf_weight_key="language_model.ple.per_layer_model_projection.weight",
            hook_fn=lambda x, _: x.T,
        )
        p(
            backbone.get_layer("per_layer_projection_norm").scale,
            "language_model.ple.per_layer_projection_norm.weight",
        )
    for i in range(backbone.num_layers):
        layer = backbone.get_layer(f"decoder_block_{i}")
        _convert_decoder_block(
            p,
            layer,
            i,
            hf_key_fn=lambda attr: f"language_model.{attr}",
            layer_prefix_str="layers.",
            ple_prefix="ple_block.",
        )
    p(
        backbone.get_layer("final_normalization").scale,
        "language_model.norm.weight",
    )
    p(
        keras_variable=backbone.get_layer("embedding_projection").kernel,
        hf_weight_key="language_model.embedding_projection.weight",
        hook_fn=lambda x, _: x.T,
    )
    if backbone.vision_encoder is not None:
        _convert_vision_encoder(backbone.vision_encoder, p, transformers_config)
    if backbone.audio_encoder is not None:
        _convert_audio_encoder(backbone.audio_encoder, p, transformers_config)


def _compute_scale_offset(proc_cfg, section_name):
    """Compute KH scale/offset from HF do_rescale/do_normalize flags.

    KH applies `output = input * scale + offset`; HF rescales then normalizes,
    so `scale = rescale_factor / std` and `offset = -mean / std`.
    """
    if not proc_cfg["do_resize"]:
        raise ValueError(
            f"{section_name}: do_resize=False is not supported; the image "
            "converter always applies aspect-ratio-preserving resize."
        )
    do_rescale = proc_cfg["do_rescale"]
    do_normalize = proc_cfg["do_normalize"]
    if not do_rescale and not do_normalize:
        return None, None
    rescale = proc_cfg["rescale_factor"] if do_rescale else 1.0
    mean = proc_cfg["image_mean"] if do_normalize else [0.0, 0.0, 0.0]
    std = proc_cfg["image_std"] if do_normalize else [1.0, 1.0, 1.0]
    scale = [rescale / s for s in std]
    offset = [-m / s for m, s in zip(mean, std)]
    return scale, offset


def load_image_converter_config(preset, transformers_config):
    if "vision_config" not in transformers_config:
        return None
    vis_cfg = transformers_config["vision_config"]
    image_processor = load_json(preset, "processor_config.json")[
        "image_processor"
    ]
    scale, offset = _compute_scale_offset(image_processor, "image_processor")
    # `image_size` is not stored in the HF config; 896 is the fixed value used
    # by every Gemma 4 family vision checkpoint.
    return {
        "image_size": (896, 896),
        "patch_size": vis_cfg["patch_size"],
        "pooling_kernel_size": vis_cfg["pooling_kernel_size"],
        "max_soft_tokens": image_processor["max_soft_tokens"],
        "scale": scale,
        "offset": offset,
    }


def load_audio_converter_config(preset, transformers_config):
    if "audio_config" not in transformers_config:
        return None
    fe = load_json(preset, "processor_config.json")["feature_extractor"]
    return {
        "num_mels": fe["feature_size"],
        "num_fft_bins": fe["fft_length"],
        "stride": fe["hop_length"],
        "sampling_rate": fe["sampling_rate"],
        "frame_length": fe["frame_length"],
        "max_frequency": fe["max_frequency"],
        "min_frequency": fe["min_frequency"],
        "mel_floor": fe["mel_floor"],
    }


def load_video_converter_config(preset, transformers_config):
    processor_config = load_json(preset, "processor_config.json")
    if "video_processor" not in processor_config:
        return None
    video_proc = processor_config["video_processor"]
    scale, offset = _compute_scale_offset(video_proc, "video_processor")
    return {
        "patch_size": video_proc["patch_size"],
        "pooling_kernel_size": video_proc["pooling_kernel_size"],
        "max_soft_tokens": video_proc["max_soft_tokens"],
        "max_frames": video_proc["max_frames"],
        "scale": scale,
        "offset": offset,
    }


def load_preprocessor_config(preset, transformers_config):
    return {
        "sequence_length": 512,
    }


def load_task_config(preset, transformers_config):
    pooling_mode = "mean"
    normalize = True
    pooling_path = os.path.join(preset, "1_Pooling", "config.json")
    if os.path.exists(pooling_path):
        with open(pooling_path, "r") as f:
            pool_config = json.load(f)
            pooling_mode = pool_config["pooling_mode"]
    return {
        "pooling_mode": pooling_mode,
        "normalize": normalize,
    }


def _build_vision_encoder(vis_cfg, text_cfg, image_size):
    from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
        Gemma4VisionEncoder,
    )

    # Same field mapping as `convert_gemma4`; `pool_size` in particular must
    # follow HF `pooling_kernel_size`, otherwise the encoder emits a different
    # number of soft tokens than the preprocessor placed in the text.
    return Gemma4VisionEncoder(
        patch_size=vis_cfg["patch_size"],
        hidden_dim=vis_cfg["hidden_size"],
        num_layers=vis_cfg["num_hidden_layers"],
        num_heads=vis_cfg["num_attention_heads"],
        intermediate_dim=vis_cfg["intermediate_size"],
        image_size=image_size,
        head_dim=vis_cfg.get("head_dim", 64),
        num_key_value_heads=vis_cfg.get(
            "num_key_value_heads", vis_cfg["num_attention_heads"]
        ),
        output_dim=text_cfg["hidden_size"],
        pool_size=vis_cfg.get("pooling_kernel_size", 3),
        position_embedding_size=vis_cfg.get("position_embedding_size", 10240),
        rope_max_wavelength=vis_cfg.get("rope_parameters", {}).get(
            "rope_theta", 100.0
        ),
        layer_norm_epsilon=vis_cfg.get("rms_norm_eps", 1e-6),
        use_clipped_linears=vis_cfg.get("use_clipped_linears", True),
        standardize=vis_cfg.get("standardize", False),
    )


def _build_audio_encoder(aud_cfg, text_cfg):
    from keras_hub.src.models.gemma4.gemma4_audio_encoder import (
        Gemma4AudioEncoder,
    )

    # HF `Gemma4AudioConfig` carries no mel-bin count. The HF subsampler sizes
    # `input_proj_linear` as `(conv_channels[0] // 4) * conv_channels[1]`
    # (modeling_gemma4.py, `Gemma4AudioSubSampleConvProjection.__init__`),
    # i.e. it assumes the mel dimension equals the first conv's channel count.
    # Derive `input_feat_size` the same way so the Keras projection matches
    # the checkpoint for any config, not only the 128-bin production one.
    sscp_conv_channels = tuple(aud_cfg["subsampling_conv_channels"])
    return Gemma4AudioEncoder(
        input_feat_size=sscp_conv_channels[0],
        hidden_size=aud_cfg["hidden_size"],
        num_heads=aud_cfg["num_attention_heads"],
        num_layers=aud_cfg["num_hidden_layers"],
        chunk_size=aud_cfg["attention_chunk_size"],
        context_left=aud_cfg["attention_context_left"],
        context_right=aud_cfg["attention_context_right"],
        logit_cap=aud_cfg["attention_logit_cap"],
        invalid_logit_value=aud_cfg["attention_invalid_logits_value"],
        conv_kernel_size=aud_cfg["conv_kernel_size"],
        residual_weight=aud_cfg["residual_weight"],
        gradient_clipping=aud_cfg["gradient_clipping"],
        sscp_conv_channels=sscp_conv_channels,
        output_proj_dims=aud_cfg["output_proj_dims"],
        output_dim=text_cfg["hidden_size"],
        norm_eps=aud_cfg["rms_norm_eps"],
        sscp_norm_eps=aud_cfg["rms_norm_eps"],
    )


def _convert_decoder_block(
    p,
    decoder_layer,
    layer_idx,
    hf_key_fn,
    layer_prefix_str="layers.",
    ple_prefix="ple_block.",
    is_vision=False,
):
    layer_prefix = f"{layer_prefix_str}{layer_idx}"

    def layer_key(attr):
        return hf_key_fn(f"{layer_prefix}.{attr}")

    kv_layer_key = layer_key
    lin = ".linear" if is_vision else ""
    p(
        decoder_layer.pre_attention_norm.scale,
        layer_key("input_layernorm.weight"),
    )
    p(
        decoder_layer.post_attention_norm.scale,
        layer_key("post_attention_layernorm.weight"),
    )
    p(
        decoder_layer.pre_ffw_norm.scale,
        layer_key("pre_feedforward_layernorm.weight"),
    )
    p(
        decoder_layer.post_ffw_norm.scale,
        layer_key("post_feedforward_layernorm.weight"),
    )
    attn = decoder_layer.attention
    p(
        attn.query_dense.dense.kernel if is_vision else attn.query_dense.kernel,
        layer_key(f"self_attn.q_proj{lin}.weight"),
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor, (keras_shape[0], keras_shape[2], keras_shape[1])
            ),
            axes=(0, 2, 1),
        ),
    )
    p(attn.query_norm.scale, layer_key("self_attn.q_norm.weight"))
    p(
        attn.key_dense.dense.kernel if is_vision else attn.key_dense.kernel,
        kv_layer_key(f"self_attn.k_proj{lin}.weight"),
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor, (keras_shape[0], keras_shape[2], keras_shape[1])
            ),
            axes=(0, 2, 1),
        ),
    )
    p(attn.key_norm.scale, kv_layer_key("self_attn.k_norm.weight"))
    p(
        attn.value_dense.dense.kernel if is_vision else attn.value_dense.kernel,
        kv_layer_key(f"self_attn.v_proj{lin}.weight"),
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor, (keras_shape[0], keras_shape[2], keras_shape[1])
            ),
            axes=(0, 2, 1),
        ),
    )
    p(
        attn.output_dense.dense.kernel
        if is_vision
        else attn.output_dense.kernel,
        layer_key(f"self_attn.o_proj{lin}.weight"),
        hook_fn=lambda hf_tensor, keras_shape: np.transpose(
            np.reshape(
                hf_tensor, (keras_shape[2], keras_shape[0], keras_shape[1])
            ),
            axes=(1, 2, 0),
        ),
    )
    p(
        decoder_layer.gating_ffw.dense.kernel
        if is_vision
        else decoder_layer.gating_ffw.kernel,
        layer_key(f"mlp.gate_proj{lin}.weight"),
        hook_fn=lambda x, _: x.T,
    )
    p(
        decoder_layer.gating_ffw_2.dense.kernel
        if is_vision
        else decoder_layer.gating_ffw_2.kernel,
        layer_key(f"mlp.up_proj{lin}.weight"),
        hook_fn=lambda x, _: x.T,
    )
    p(
        decoder_layer.ffw_linear.dense.kernel
        if is_vision
        else decoder_layer.ffw_linear.kernel,
        layer_key(f"mlp.down_proj{lin}.weight"),
        hook_fn=lambda x, _: x.T,
    )
    if not is_vision:
        if decoder_layer.hidden_size_per_layer_input > 0:
            p(
                keras_variable=decoder_layer.per_layer_input_gate.kernel,
                hf_weight_key=hf_key_fn(
                    f"{layer_prefix}.{ple_prefix}per_layer_input_gate.weight"
                ),
                hook_fn=lambda x, _: x.T,
            )
            p(
                keras_variable=decoder_layer.per_layer_up_proj.kernel,
                hf_weight_key=hf_key_fn(
                    f"{layer_prefix}.{ple_prefix}per_layer_projection.weight"
                ),
                hook_fn=lambda x, _: x.T,
            )
            p(
                decoder_layer.post_per_layer_input_norm.scale,
                hf_key_fn(
                    f"{layer_prefix}.{ple_prefix}post_per_layer_input_norm.weight"
                ),
            )
        p(
            decoder_layer.layer_scalar,
            layer_key("layer_scalar"),
            hook_fn=lambda x, _: np.squeeze(x),
        )


def _port_clips(p, keras_layer, hf_name):
    if not getattr(keras_layer, "use_clipped_linears", False):
        return
    for w in ["input_min", "input_max", "output_min", "output_max"]:
        p(getattr(keras_layer, w), f"{hf_name}.{w}")


def _convert_vision_encoder(vision_encoder, p, transformers_config):
    image_encoder = vision_encoder.get_layer("image_encoder")
    patch_embedder = image_encoder.patch_embedder
    vis_prefix = "vision_tower"
    p(
        keras_variable=patch_embedder.input_proj.kernel,
        hf_weight_key=f"{vis_prefix}.patch_embedder.input_proj.weight",
        hook_fn=lambda x, _: x.T,
    )
    p(
        patch_embedder.position_embedding_table,
        f"{vis_prefix}.patch_embedder.position_embedding_table",
    )
    for i, block in enumerate(image_encoder.encoder_blocks):
        _convert_decoder_block(
            p,
            block,
            i,
            hf_key_fn=lambda attr: f"{vis_prefix}.encoder.{attr}",
            is_vision=True,
        )
        attn = block.attention
        blk = f"{vis_prefix}.encoder.layers.{i}"
        _port_clips(p, attn.query_dense, f"{blk}.self_attn.q_proj")
        _port_clips(p, attn.key_dense, f"{blk}.self_attn.k_proj")
        _port_clips(p, attn.value_dense, f"{blk}.self_attn.v_proj")
        _port_clips(p, attn.output_dense, f"{blk}.self_attn.o_proj")
        _port_clips(p, block.gating_ffw, f"{blk}.mlp.gate_proj")
        _port_clips(p, block.gating_ffw_2, f"{blk}.mlp.up_proj")
        _port_clips(p, block.ffw_linear, f"{blk}.mlp.down_proj")
    projector_prefix = "embed_vision"
    vision_output = vision_encoder.get_layer("vision_output_encoder")
    p(
        keras_variable=vision_output.vision_input_projection.kernel,
        hf_weight_key=f"{projector_prefix}.embedding_projection.weight",
        hook_fn=lambda x, _: x.T,
    )


def _convert_audio_encoder(audio_encoder, p, transformers_config):
    aud_prefix = "audio_tower"
    sscp = audio_encoder.subsample_conv_projection
    for conv_block, hf_attr in [
        (sscp.conv_0, "layer0"),
        (sscp.conv_1, "layer1"),
    ]:
        hf_conv_pfx = f"{aud_prefix}.subsample_conv_projection.{hf_attr}"
        p(
            keras_variable=conv_block.conv.kernel,
            hf_weight_key=f"{hf_conv_pfx}.conv.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        p(conv_block.norm.gamma, f"{hf_conv_pfx}.norm.weight")
    p(
        keras_variable=sscp.input_proj.kernel,
        hf_weight_key=f"{aud_prefix}.subsample_conv_projection.input_proj_linear.weight",
        hook_fn=lambda x, _: x.T,
    )
    for i, block in enumerate(audio_encoder.conformer_blocks):
        hf_blk = f"{aud_prefix}.layers.{i}"
        for hf_ffw_name, keras_ffw in [
            ("feed_forward1", block.ffw_start),
            ("feed_forward2", block.ffw_end),
        ]:
            p(
                keras_variable=keras_ffw.ffw_1.dense.kernel,
                hf_weight_key=f"{hf_blk}.{hf_ffw_name}.ffw_layer_1.linear.weight",
                hook_fn=lambda x, _: x.T,
            )
            p(
                keras_variable=keras_ffw.ffw_2.dense.kernel,
                hf_weight_key=f"{hf_blk}.{hf_ffw_name}.ffw_layer_2.linear.weight",
                hook_fn=lambda x, _: x.T,
            )
            ffw_pfx = f"{hf_blk}.{hf_ffw_name}"
            _port_clips(p, keras_ffw.ffw_1, f"{ffw_pfx}.ffw_layer_1")
            _port_clips(p, keras_ffw.ffw_2, f"{ffw_pfx}.ffw_layer_2")
        attn = block.attention.attn
        hf_attn = f"{hf_blk}.self_attn"
        for proj_name, keras_dense in [
            ("q_proj", attn.q_proj),
            ("k_proj", attn.k_proj),
            ("v_proj", attn.v_proj),
        ]:
            p(
                keras_variable=keras_dense.dense.kernel,
                hf_weight_key=f"{hf_attn}.{proj_name}.linear.weight",
                hook_fn=lambda x, _: x.T,
            )
            _port_clips(p, keras_dense, f"{hf_attn}.{proj_name}")
        p(attn.per_dim_scale, f"{hf_attn}.per_dim_scale")
        p(
            keras_variable=attn.rpe.pos_proj,
            hf_weight_key=f"{hf_attn}.relative_k_proj.weight",
            hook_fn=lambda x, _: x.T,
        )
        p(
            keras_variable=block.attention.out_proj.dense.kernel,
            hf_weight_key=f"{hf_blk}.self_attn.post.linear.weight",
            hook_fn=lambda x, _: x.T,
        )
        _port_clips(p, block.attention.out_proj, f"{hf_attn}.post")
        lconv = block.lconv
        hf_lconv = f"{hf_blk}.lconv1d"
        p(
            keras_variable=lconv.linear_start.dense.kernel,
            hf_weight_key=f"{hf_lconv}.linear_start.linear.weight",
            hook_fn=lambda x, _: x.T,
        )
        _port_clips(p, lconv.linear_start, f"{hf_lconv}.linear_start")
        p(
            keras_variable=lconv.depthwise_conv.kernel,
            hf_weight_key=f"{hf_lconv}.depthwise_conv1d.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 0, 1)),
        )
        p(
            keras_variable=lconv.linear_end.dense.kernel,
            hf_weight_key=f"{hf_lconv}.linear_end.linear.weight",
            hook_fn=lambda x, _: x.T,
        )
        _port_clips(p, lconv.linear_end, f"{hf_lconv}.linear_end")
        for hf_ffw_name, keras_ffw in [
            ("feed_forward1", block.ffw_start),
            ("feed_forward2", block.ffw_end),
        ]:
            p(
                keras_ffw.pre_norm.scale,
                f"{hf_blk}.{hf_ffw_name}.pre_layer_norm.weight",
            )
            p(
                keras_ffw.post_norm.scale,
                f"{hf_blk}.{hf_ffw_name}.post_layer_norm.weight",
            )
        p(
            block.attention.pre_attn_norm.scale,
            f"{hf_blk}.norm_pre_attn.weight",
        )
        p(block.attention.post_norm.scale, f"{hf_blk}.norm_post_attn.weight")
        p(lconv.pre_norm.scale, f"{hf_lconv}.pre_layer_norm.weight")
        p(lconv.conv_norm.scale, f"{hf_lconv}.conv_norm.weight")
        p(block.norm.scale, f"{hf_blk}.norm_out.weight")
    if audio_encoder.output_proj is not None:
        p(
            keras_variable=audio_encoder.output_proj.kernel,
            hf_weight_key=f"{aud_prefix}.output_proj.weight",
            hook_fn=lambda x, _: x.T,
        )
        p(audio_encoder.output_proj.bias, f"{aud_prefix}.output_proj.bias")
    p(
        keras_variable=audio_encoder.audio_output_projection.kernel,
        hf_weight_key="embed_audio.embedding_projection.weight",
        hook_fn=lambda x, _: x.T,
    )


def convert_tokenizer(cls, preset, **kwargs):
    # EmbeddingGemma2 ships the Gemma4 tokenizer unchanged; reuse the Gemma4
    # converter so added_tokens (<|image|>, <|audio|>, ...) are emitted as
    # USER_DEFINED pieces and tokenize atomically, exactly as in HF.
    return convert_gemma4.convert_tokenizer(cls, preset, **kwargs)
