"""KerasHub → HuggingFace export utilities for Gemma4.

This module provides three public helpers that are consumed by
`hf_exporter.py`:

  * ``get_gemma4_config``       – build an HF-compatible ``config.json`` dict.
  * ``get_gemma4_weights_map``  – return a ``{hf_key: keras_tensor}`` dict.
  * ``get_gemma4_tokenizer_config`` – build an HF ``tokenizer_config.json``.

Multimodal (vision + audio) models are fully supported.  Text-only models
(``vision_encoder=None`` and ``audio_encoder=None``) produce a flat
``gemma4_text`` config and put weights under ``model.*`` rather than
``model.language_model.*``.
"""

from keras import ops

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _build_text_config(backbone):
    """Build the flat text-config dict from backbone attributes."""
    # Rope parameters section
    rope_parameters = {
        "full_attention": {
            "rope_theta": backbone.global_rope_wavelength or 1_000_000.0,
            "partial_rotary_factor": backbone.global_rope_partial_rotary_factor,
        },
        "sliding_attention": {
            "rope_theta": backbone.local_rope_wavelength or 10_000.0,
        },
    }

    # HF uses "vision" string for vision-only bidirectional attention.
    use_bidirectional_attention = (
        "vision" if backbone.use_vision_bidirectional_attention else None
    )

    text_cfg = {
        "model_type": "gemma4_text",
        "vocab_size": backbone.vocabulary_size,
        "num_hidden_layers": backbone.num_layers,
        "num_attention_heads": backbone.num_query_heads,
        "num_key_value_heads": backbone.num_key_value_heads,
        "hidden_size": backbone.hidden_dim,
        "intermediate_size": backbone.intermediate_dim,
        "head_dim": backbone.head_dim,
        "global_head_dim": backbone.global_head_dim or backbone.head_dim,
        "rms_norm_eps": backbone.layer_norm_epsilon,
        "attention_bias": False,
        "attention_dropout": backbone.dropout,
        "hidden_activation": "gelu_pytorch_tanh",
        "sliding_window": backbone.sliding_window_size,
        "_sliding_window_pattern": backbone.sliding_window_pattern,
        "use_cache": True,
        "layer_types": backbone.layer_types,
        "attn_logit_softcapping": backbone.attention_logit_soft_cap,
        "final_logit_softcapping": backbone.final_logit_soft_cap,
        "num_kv_shared_layers": backbone.num_kv_shared_layers,
        "num_global_key_value_heads": (
            backbone.num_global_key_value_heads or backbone.num_key_value_heads
        ),
        "hidden_size_per_layer_input": backbone.hidden_size_per_layer_input,
        "vocab_size_per_layer_input": (
            backbone.vocab_size_per_layer_input or backbone.vocabulary_size
        ),
        "use_double_wide_mlp": backbone.use_double_wide_mlp,
        "enable_moe_block": backbone.enable_moe_block,
        "num_experts": backbone.num_experts,
        "moe_intermediate_size": backbone.expert_intermediate_dim,
        "top_k_experts": backbone.num_experts_per_token,
        "use_bidirectional_attention": use_bidirectional_attention,
        "rope_parameters": rope_parameters,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 1,
    }
    return text_cfg


def _build_vision_config(vision_encoder):
    """Build the ``vision_config`` dict from a Gemma4VisionEncoder."""
    return {
        "model_type": "gemma4_vision_model",
        "num_hidden_layers": vision_encoder.num_layers,
        "num_attention_heads": vision_encoder.num_heads,
        "num_key_value_heads": vision_encoder.num_key_value_heads,
        "hidden_size": vision_encoder.hidden_dim,
        "intermediate_size": vision_encoder.intermediate_dim,
        "head_dim": vision_encoder.head_dim,
        "patch_size": vision_encoder.patch_size,
        "pooling_kernel_size": vision_encoder.pool_size,
        "position_embedding_size": vision_encoder.position_embedding_size,
        "rms_norm_eps": vision_encoder.layer_norm_epsilon,
        "use_clipped_linears": vision_encoder.use_clipped_linears,
        "standardize": vision_encoder.standardize,
        "rope_parameters": {
            "rope_theta": vision_encoder.rope_max_wavelength,
        },
    }


def _build_audio_config(audio_encoder):
    """Build the ``audio_config`` dict from a Gemma4AudioEncoder."""
    return {
        "model_type": "gemma4_audio_model",
        "hidden_size": audio_encoder.hidden_size,
        "num_attention_heads": audio_encoder.num_heads,
        "num_hidden_layers": audio_encoder.num_layers,
        "attention_chunk_size": audio_encoder.chunk_size,
        "attention_context_left": audio_encoder.context_left,
        "attention_context_right": audio_encoder.context_right,
        "attention_logit_cap": audio_encoder.logit_cap,
        "attention_invalid_logits_value": audio_encoder.invalid_logit_value,
        "conv_kernel_size": audio_encoder.conv_kernel_size,
        "residual_weight": audio_encoder.residual_weight,
        "gradient_clipping": audio_encoder.gradient_clipping,
        "subsampling_conv_channels": list(audio_encoder.sscp_conv_channels),
        "output_proj_dims": audio_encoder.output_proj_dims,
        "rms_norm_eps": audio_encoder.norm_eps,
    }


def get_gemma4_config(backbone, include_lm_head=False):
    """Build an HF-compatible config dict for the given Gemma4 backbone.

    Args:
        backbone: A ``keras_hub.models.Gemma4Backbone`` instance.
        include_lm_head: bool. When ``True`` the architectures field
            reflects a CausalLM model.

    Returns:
        dict suitable for writing to ``config.json``.
    """
    has_vision = backbone.vision_encoder is not None
    has_audio = backbone.audio_encoder is not None
    is_text_only = not has_vision and not has_audio

    text_cfg = _build_text_config(backbone)

    if is_text_only:
        # Flat text-only config.
        arch = "Gemma4TextForCausalLM" if include_lm_head else "Gemma4TextModel"
        hf_config = dict(text_cfg)
        hf_config["architectures"] = [arch]
        hf_config["torch_dtype"] = backbone.dtype_policy.compute_dtype
        return hf_config

    # Multimodal config — text goes under "text_config".
    arch = (
        "Gemma4ForConditionalGeneration" if include_lm_head else "Gemma4Model"
    )
    hf_config = {
        "architectures": [arch],
        "model_type": "gemma4",
        "torch_dtype": backbone.dtype_policy.compute_dtype,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "text_config": text_cfg,
    }

    if has_vision:
        hf_config["vision_config"] = _build_vision_config(
            backbone.vision_encoder
        )

    if has_audio:
        hf_config["audio_config"] = _build_audio_config(backbone.audio_encoder)

    return hf_config


# ---------------------------------------------------------------------------
# Weight-map helpers
# ---------------------------------------------------------------------------


def _convert_qkv_kernel(kernel, hidden_dim):
    """Convert a Q/K/V EinsumDense kernel (n, d, h) → HF format (n*h, d).

    Keras stores:  (num_heads, hidden_dim, head_dim)  [from btd,ndh->btnh]
    HF stores:     (num_heads * head_dim, hidden_dim)
    """
    # (n, d, h) → (n, h, d) → (n*h, d)
    kernel = ops.transpose(kernel, axes=(0, 2, 1))
    kernel = ops.reshape(kernel, (-1, hidden_dim))
    return kernel


def _convert_output_kernel(kernel):
    """Convert an output EinsumDense kernel (n, h, d) → HF format (d, n*h).

    Keras stores:  (num_heads, head_dim, hidden_dim)  [from btnh,nhd->btd]
    HF stores:     (hidden_dim, num_heads * head_dim)
    """
    # (n, h, d) → (d, n, h) → (d, n*h)
    hidden_dim = kernel.shape[-1]
    kernel = ops.transpose(kernel, axes=(2, 0, 1))
    kernel = ops.reshape(kernel, (hidden_dim, -1))
    return kernel


def _add_text_decoder_block(weights_dict, block, layer_idx, prefix):
    """Populate weights_dict with one text decoder block's weights.

    Args:
        weights_dict: dict to update in place.
        block: ``Gemma4TextDecoderBlock`` instance.
        layer_idx: int layer index.
        prefix: HF key prefix (e.g. ``"model.language_model."``).
    """
    lp = f"{prefix}layers.{layer_idx}"

    # --- Layer norms ---
    weights_dict[f"{lp}.input_layernorm.weight"] = (
        block.pre_attention_norm.scale
    )
    weights_dict[f"{lp}.post_attention_layernorm.weight"] = (
        block.post_attention_norm.scale
    )
    weights_dict[f"{lp}.pre_feedforward_layernorm.weight"] = (
        block.pre_ffw_norm.scale
    )
    weights_dict[f"{lp}.post_feedforward_layernorm.weight"] = (
        block.post_ffw_norm.scale
    )

    # --- Attention ---
    attn = block.attention
    hidden_dim = block.hidden_dim

    # Q projection (always present).
    weights_dict[f"{lp}.self_attn.q_proj.weight"] = _convert_qkv_kernel(
        attn.query_dense.kernel, hidden_dim
    )
    weights_dict[f"{lp}.self_attn.q_norm.weight"] = attn.query_norm.scale

    # K / V projections (absent on KV-shared layers).
    if not attn.is_kv_shared_layer:
        weights_dict[f"{lp}.self_attn.k_proj.weight"] = _convert_qkv_kernel(
            attn.key_dense.kernel, hidden_dim
        )
        weights_dict[f"{lp}.self_attn.k_norm.weight"] = attn.key_norm.scale

        # V proj absent when attention_k_eq_v=True (global layers in MoE
        # models like 26B-A4B and 31B that reuse K for V).
        if attn.value_dense is not None:
            weights_dict[f"{lp}.self_attn.v_proj.weight"] = _convert_qkv_kernel(
                attn.value_dense.kernel, hidden_dim
            )
        # v_norm (Gemma4VNorm) has no learnable scale — skip.

    # O projection.
    weights_dict[f"{lp}.self_attn.o_proj.weight"] = _convert_output_kernel(
        attn.output_dense.kernel
    )

    # --- MLP ---
    weights_dict[f"{lp}.mlp.gate_proj.weight"] = ops.transpose(
        block.gating_ffw.kernel
    )
    weights_dict[f"{lp}.mlp.up_proj.weight"] = ops.transpose(
        block.gating_ffw_2.kernel
    )
    weights_dict[f"{lp}.mlp.down_proj.weight"] = ops.transpose(
        block.ffw_linear.kernel
    )

    # --- Per-layer input conditioning (E2B / E4B) ---
    if block.hidden_size_per_layer_input > 0:
        weights_dict[f"{lp}.per_layer_input_gate.weight"] = ops.transpose(
            block.per_layer_input_gate.kernel
        )
        weights_dict[f"{lp}.per_layer_projection.weight"] = ops.transpose(
            block.per_layer_up_proj.kernel
        )
        weights_dict[f"{lp}.post_per_layer_input_norm.weight"] = (
            block.post_per_layer_input_norm.scale
        )

    # --- MoE block (26B-A4B architecture) ---
    if block.enable_moe_block:
        # Extra norms.
        weights_dict[f"{lp}.post_feedforward_layernorm_1.weight"] = (
            block.post_ffw_norm_dense.scale
        )
        weights_dict[f"{lp}.pre_feedforward_layernorm_2.weight"] = (
            block.pre_ffw_norm_moe.scale
        )
        weights_dict[f"{lp}.post_feedforward_layernorm_2.weight"] = (
            block.post_ffw_norm_moe_path.scale
        )

        # Router.
        weights_dict[f"{lp}.router.scale"] = block.moe_router.per_dim_scale
        weights_dict[f"{lp}.router.proj.weight"] = ops.transpose(
            block.moe_router.proj.kernel
        )
        weights_dict[f"{lp}.router.per_expert_scale"] = (
            block.moe_expert_bank.per_expert_scale
        )

        # Expert bank: HF stores gate and up parts concatenated in
        # ``experts.gate_up_proj`` of shape (E, 2*I, H).
        # KH stores them separately as (E, H, I).
        # KH → HF: transpose each to (E, I, H), then concat on axis=1.
        gate_hf = ops.transpose(
            block.moe_expert_bank.gate_proj, axes=(0, 2, 1)
        )  # (E, I, H)
        up_hf = ops.transpose(
            block.moe_expert_bank.up_proj, axes=(0, 2, 1)
        )  # (E, I, H)
        weights_dict[f"{lp}.experts.gate_up_proj"] = ops.concatenate(
            [gate_hf, up_hf], axis=1
        )  # (E, 2*I, H)

        # down_proj: KH (E, I, H) → HF (E, H, I).
        weights_dict[f"{lp}.experts.down_proj"] = ops.transpose(
            block.moe_expert_bank.down_proj, axes=(0, 2, 1)
        )

    # --- Layer scalar (non-trainable buffer, shape () in KH, (1,) in HF) ---
    weights_dict[f"{lp}.layer_scalar"] = ops.expand_dims(
        block.layer_scalar, axis=0
    )


def _add_vision_block(weights_dict, block, layer_prefix):
    """Add one Gemma4VisionDecoderBlock's weights to weights_dict.

    Vision blocks use ``Gemma4ClippableEinsumDense`` (with ``.dense.kernel``)
    and ``Gemma4ClippableDense`` (with ``.dense.kernel``).  The HF key uses
    ``.linear.weight`` rather than just ``.weight``.

    Clipping scalars (input_min / max, output_min / max) are included when
    ``use_clipped_linears=True``.
    """
    lp = layer_prefix
    hidden_dim = block.hidden_dim

    def _add_clip_weights(keras_layer, hf_name):
        """Add clip scalars if the layer uses clipped linears."""
        if not getattr(keras_layer, "use_clipped_linears", False):
            return
        for w in ("input_min", "input_max", "output_min", "output_max"):
            weights_dict[f"{hf_name}.{w}"] = getattr(keras_layer, w)

    # Layer norms.
    weights_dict[f"{lp}.input_layernorm.weight"] = (
        block.pre_attention_norm.scale
    )
    weights_dict[f"{lp}.post_attention_layernorm.weight"] = (
        block.post_attention_norm.scale
    )
    weights_dict[f"{lp}.pre_feedforward_layernorm.weight"] = (
        block.pre_ffw_norm.scale
    )
    weights_dict[f"{lp}.post_feedforward_layernorm.weight"] = (
        block.post_ffw_norm.scale
    )

    # Attention — Gemma4VisionAttention uses ClippableEinsumDense.
    attn = block.attention
    q_kernel = _convert_qkv_kernel(attn.query_dense.dense.kernel, hidden_dim)
    weights_dict[f"{lp}.self_attn.q_proj.linear.weight"] = q_kernel
    _add_clip_weights(attn.query_dense, f"{lp}.self_attn.q_proj")

    weights_dict[f"{lp}.self_attn.q_norm.weight"] = attn.query_norm.scale

    k_kernel = _convert_qkv_kernel(attn.key_dense.dense.kernel, hidden_dim)
    weights_dict[f"{lp}.self_attn.k_proj.linear.weight"] = k_kernel
    _add_clip_weights(attn.key_dense, f"{lp}.self_attn.k_proj")

    weights_dict[f"{lp}.self_attn.k_norm.weight"] = attn.key_norm.scale

    v_kernel = _convert_qkv_kernel(attn.value_dense.dense.kernel, hidden_dim)
    weights_dict[f"{lp}.self_attn.v_proj.linear.weight"] = v_kernel
    _add_clip_weights(attn.value_dense, f"{lp}.self_attn.v_proj")
    # v_norm (Gemma4VNorm) is parameter-free — skip.

    o_kernel = _convert_output_kernel(attn.output_dense.dense.kernel)
    weights_dict[f"{lp}.self_attn.o_proj.linear.weight"] = o_kernel
    _add_clip_weights(attn.output_dense, f"{lp}.self_attn.o_proj")

    # MLP — ClippableEinsumDense wraps EinsumDense; also uses .dense.kernel.
    weights_dict[f"{lp}.mlp.gate_proj.linear.weight"] = ops.transpose(
        block.gating_ffw.dense.kernel
    )
    _add_clip_weights(block.gating_ffw, f"{lp}.mlp.gate_proj")

    weights_dict[f"{lp}.mlp.up_proj.linear.weight"] = ops.transpose(
        block.gating_ffw_2.dense.kernel
    )
    _add_clip_weights(block.gating_ffw_2, f"{lp}.mlp.up_proj")

    weights_dict[f"{lp}.mlp.down_proj.linear.weight"] = ops.transpose(
        block.ffw_linear.dense.kernel
    )
    _add_clip_weights(block.ffw_linear, f"{lp}.mlp.down_proj")


def _add_vision_encoder_weights(weights_dict, vision_encoder):
    """Add all vision-encoder weights to weights_dict."""
    vis_prefix = "model.vision_tower"
    image_encoder = vision_encoder.get_layer("image_encoder")
    patch_embedder = image_encoder.patch_embedder

    # Patch embedder.
    weights_dict[f"{vis_prefix}.patch_embedder.input_proj.weight"] = (
        ops.transpose(patch_embedder.input_proj.kernel)
    )
    weights_dict[f"{vis_prefix}.patch_embedder.position_embedding_table"] = (
        patch_embedder.position_embedding_table
    )

    # Transformer blocks.
    for i, block in enumerate(image_encoder.encoder_blocks):
        vis_layer_prefix = f"{vis_prefix}.encoder.layers.{i}"
        _add_vision_block(weights_dict, block, vis_layer_prefix)

    # Output projection.
    vision_output = vision_encoder.get_layer("vision_output_encoder")
    weights_dict["model.embed_vision.embedding_projection.weight"] = (
        ops.transpose(vision_output.vision_input_projection.kernel)
    )

    # Optional standardization weights.
    if vision_encoder.standardize:
        weights_dict[f"{vis_prefix}.std_bias"] = vision_output.std_bias
        weights_dict[f"{vis_prefix}.std_scale"] = vision_output.std_scale


def _add_audio_encoder_weights(weights_dict, audio_encoder):
    """Add all audio-encoder weights to weights_dict."""
    aud_prefix = "model.audio_tower"
    sscp = audio_encoder.subsample_conv_projection

    def _add_clip_weights(keras_layer, hf_name):
        if not getattr(keras_layer, "use_clipped_linears", False):
            return
        for w in ("input_min", "input_max", "output_min", "output_max"):
            weights_dict[f"{hf_name}.{w}"] = getattr(keras_layer, w)

    # --- SubSample Convolution Projection ---
    for conv_block, hf_attr in [
        (sscp.conv_0, "layer0"),
        (sscp.conv_1, "layer1"),
    ]:
        hf_conv_pfx = f"{aud_prefix}.subsample_conv_projection.{hf_attr}"
        # Keras Conv2D (channels_last): (kT, kF, C_in, C_out)
        # HF (PyTorch Conv2D):          (C_out, C_in, kT, kF)
        weights_dict[f"{hf_conv_pfx}.conv.weight"] = ops.transpose(
            conv_block.conv.kernel, axes=(3, 2, 0, 1)
        )
        weights_dict[f"{hf_conv_pfx}.norm.weight"] = conv_block.norm.gamma

    # Input projection: (proj_in, hidden) → (hidden, proj_in)
    weights_dict[
        f"{aud_prefix}.subsample_conv_projection.input_proj_linear.weight"
    ] = ops.transpose(sscp.input_proj.kernel)

    # --- Conformer blocks ---
    for i, block in enumerate(audio_encoder.conformer_blocks):
        hf_blk = f"{aud_prefix}.layers.{i}"

        # Feed-forward sub-blocks (Macaron FFW 1 and 2).
        for hf_ffw_name, keras_ffw in [
            ("feed_forward1", block.ffw_start),
            ("feed_forward2", block.ffw_end),
        ]:
            hf_ffw_pfx = f"{hf_blk}.{hf_ffw_name}"
            weights_dict[f"{hf_ffw_pfx}.ffw_layer_1.linear.weight"] = (
                ops.transpose(keras_ffw.ffw_1.dense.kernel)
            )
            _add_clip_weights(keras_ffw.ffw_1, f"{hf_ffw_pfx}.ffw_layer_1")
            weights_dict[f"{hf_ffw_pfx}.ffw_layer_2.linear.weight"] = (
                ops.transpose(keras_ffw.ffw_2.dense.kernel)
            )
            _add_clip_weights(keras_ffw.ffw_2, f"{hf_ffw_pfx}.ffw_layer_2")
            weights_dict[f"{hf_ffw_pfx}.pre_layer_norm.weight"] = (
                keras_ffw.pre_norm.scale
            )
            weights_dict[f"{hf_ffw_pfx}.post_layer_norm.weight"] = (
                keras_ffw.post_norm.scale
            )

        # Attention sub-block.
        attn = block.attention.attn
        hf_attn = f"{hf_blk}.self_attn"
        for proj_name, keras_dense in [
            ("q_proj", attn.q_proj),
            ("k_proj", attn.k_proj),
            ("v_proj", attn.v_proj),
        ]:
            weights_dict[f"{hf_attn}.{proj_name}.linear.weight"] = (
                ops.transpose(keras_dense.dense.kernel)
            )
            _add_clip_weights(keras_dense, f"{hf_attn}.{proj_name}")

        weights_dict[f"{hf_attn}.per_dim_scale"] = attn.per_dim_scale
        # Relative position projection: (hidden, N*H) → (N*H, hidden)
        weights_dict[f"{hf_attn}.relative_k_proj.weight"] = ops.transpose(
            attn.rpe.pos_proj
        )
        # Output projection.
        weights_dict[f"{hf_blk}.self_attn.post.linear.weight"] = ops.transpose(
            block.attention.out_proj.dense.kernel
        )
        _add_clip_weights(block.attention.out_proj, f"{hf_blk}.self_attn.post")

        # LightConv1D sub-block.
        lconv = block.lconv
        hf_lconv = f"{hf_blk}.lconv1d"
        weights_dict[f"{hf_lconv}.linear_start.linear.weight"] = ops.transpose(
            lconv.linear_start.dense.kernel
        )
        _add_clip_weights(lconv.linear_start, f"{hf_lconv}.linear_start")

        # Keras DepthwiseConv1D (channels_last): (ksize, C_in, depth_mult=1)
        # HF (PyTorch, groups=C):                (C_in, 1, ksize)
        weights_dict[f"{hf_lconv}.depthwise_conv1d.weight"] = ops.transpose(
            lconv.depthwise_conv.kernel, axes=(1, 2, 0)
        )

        weights_dict[f"{hf_lconv}.linear_end.linear.weight"] = ops.transpose(
            lconv.linear_end.dense.kernel
        )
        _add_clip_weights(lconv.linear_end, f"{hf_lconv}.linear_end")

        # Norms.
        weights_dict[f"{hf_blk}.norm_pre_attn.weight"] = (
            block.attention.pre_attn_norm.scale
        )
        weights_dict[f"{hf_blk}.norm_post_attn.weight"] = (
            block.attention.post_norm.scale
        )
        weights_dict[f"{hf_lconv}.pre_layer_norm.weight"] = lconv.pre_norm.scale
        weights_dict[f"{hf_lconv}.conv_norm.weight"] = lconv.conv_norm.scale
        weights_dict[f"{hf_blk}.norm_out.weight"] = block.norm.scale

    # --- Optional intermediate output projection ---
    if audio_encoder.output_proj is not None:
        weights_dict[f"{aud_prefix}.output_proj.weight"] = ops.transpose(
            audio_encoder.output_proj.kernel
        )
        weights_dict[f"{aud_prefix}.output_proj.bias"] = (
            audio_encoder.output_proj.bias
        )

    # --- Audio-to-text projection ---
    weights_dict["model.embed_audio.embedding_projection.weight"] = (
        ops.transpose(audio_encoder.audio_output_projection.kernel)
    )


def get_gemma4_weights_map(backbone, include_lm_head=False):
    """Build a ``{hf_key: keras_tensor}`` weight map for the Gemma4 backbone.

    Handles:
    * Text-only models (text prefix ``model.``)
    * Multimodal models (text prefix ``model.language_model.``)
    * Vision encoder weights
    * Audio encoder weights
    * Per-layer token conditioning (E2B / E4B)
    * MoE blocks (26B-A4B)
    * KV-shared layers
    * ``layer_scalar`` non-trainable buffers

    Args:
        backbone: A ``keras_hub.models.Gemma4Backbone`` instance.
        include_lm_head: bool. When ``True``, include ``lm_head.weight`` if
            the embedding layer does not tie weights.

    Returns:
        dict mapping HF weight keys to Keras tensors.
    """
    weights_dict = {}

    has_vision = backbone.vision_encoder is not None
    has_audio = backbone.audio_encoder is not None
    is_text_only = not has_vision and not has_audio

    # Prefix for language-model text weights.
    text_prefix = "model." if is_text_only else "model.language_model."

    # --- Token embeddings ---
    token_embedding_layer = backbone.get_layer("token_embedding")
    weights_dict[f"{text_prefix}embed_tokens.weight"] = (
        token_embedding_layer.embeddings
    )

    # --- Per-layer token conditioning (E2B / E4B) ---
    if backbone.hidden_size_per_layer_input > 0:
        weights_dict[f"{text_prefix}embed_tokens_per_layer.weight"] = (
            backbone.get_layer("per_layer_token_embedding").embeddings
        )
        weights_dict[f"{text_prefix}per_layer_model_projection.weight"] = (
            ops.transpose(
                backbone.get_layer("per_layer_model_projection").kernel
            )
        )
        weights_dict[f"{text_prefix}per_layer_projection_norm.weight"] = (
            backbone.get_layer("per_layer_projection_norm").scale
        )

    # --- Vision encoder ---
    if has_vision:
        _add_vision_encoder_weights(weights_dict, backbone.vision_encoder)

    # --- Audio encoder ---
    if has_audio:
        _add_audio_encoder_weights(weights_dict, backbone.audio_encoder)

    # --- Text decoder blocks ---
    for i in range(backbone.num_layers):
        block = backbone.get_layer(f"decoder_block_{i}")
        _add_text_decoder_block(weights_dict, block, i, text_prefix)

    # --- Final layer norm ---
    weights_dict[f"{text_prefix}norm.weight"] = backbone.get_layer(
        "final_normalization"
    ).scale

    # --- LM head (optional, only when weights are not tied) ---
    if include_lm_head and not token_embedding_layer.tie_weights:
        weights_dict["lm_head.weight"] = ops.transpose(
            token_embedding_layer.reverse_embeddings
        )

    return weights_dict


# ---------------------------------------------------------------------------
# Tokenizer config helper
# ---------------------------------------------------------------------------


def get_gemma4_tokenizer_config(tokenizer):
    """Build a Gemma4-compatible ``tokenizer_config.json`` dict.

    Args:
        tokenizer: A ``keras_hub.models.Gemma4Tokenizer`` instance.

    Returns:
        dict suitable for writing to ``tokenizer_config.json``.
    """
    tokenizer_config = {
        "tokenizer_class": "GemmaTokenizer",
        "clean_up_tokenization_spaces": False,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "add_bos_token": True,
        "add_eos_token": False,
        "model_max_length": 1000000000000000019884624838656,
    }

    # Special tokens registered by the tokenizer.
    special_tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<mask>",
        "[multimodal]",
        "<img>",
    ]
    # Gemma4-specific multimodal tokens.
    if getattr(tokenizer, "has_vision_tokens", False):
        special_tokens += ["<|image>", "<|image|>", "<image|>"]
    if getattr(tokenizer, "has_audio_tokens", False):
        special_tokens += ["<|audio>", "<|audio|>", "<audio|>"]
    if getattr(tokenizer, "has_video_tokens", False):
        special_tokens += ["<|video>", "<|video|>", "<video|>"]

    added_tokens_decoder = {}
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            added_tokens_decoder[str(token_id)] = {
                "content": token,
                "special": True,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
    tokenizer_config["added_tokens_decoder"] = added_tokens_decoder
    return tokenizer_config
