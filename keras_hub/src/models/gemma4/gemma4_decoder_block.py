import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.gemma4.gemma4_attention import Gemma4TextAttention
from keras_hub.src.models.gemma4.gemma4_attention import Gemma4VisionAttention
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4ClippableEinsumDense
from keras_hub.src.models.gemma4.gemma4_layers import RMSNormalization
from keras_hub.src.models.gemma4.gemma4_moe import Gemma4MoEBlock
from keras_hub.src.models.gemma4.gemma4_moe import Gemma4Router


class Gemma4TextDecoderBlock(keras.layers.Layer):
    """Transformer decoder layer for Gemma4.

    Gemma4 has several differences from Gemma3:

    1. Four normalizations per block (pre + post for both attention and FFW),
       which are always active (unlike Gemma3 where they are configurable).
    2. Q/K/V normalization always applied in attention.
    3. `scaling = 1.0` in attention (Q/K norms replace explicit scaling).
    4. All text decoder layers have a non-trainable `layer_scalar` (init 1.0)
       that the output is multiplied by (vision encoder blocks do not).
    5. Default sliding_window_size is 512 (vs 1024 in Gemma3).
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=512,
        layer_norm_epsilon=1e-6,
        rope_wavelength=10_000.0,
        rope_scaling_factor=1.0,
        rope_partial_rotary_factor=1.0,
        use_bidirectional_attention=False,
        use_vision_bidirectional_attention=False,
        is_global_attention=False,
        global_head_dim=None,
        dropout=0,
        is_kv_shared_layer=False,
        kv_shared_layer_index=None,
        hidden_size_per_layer_input=0,
        attention_k_eq_v=False,
        num_global_key_value_heads=None,
        use_double_wide_mlp=False,
        enable_moe_block=False,
        num_experts=None,
        expert_intermediate_dim=None,
        num_experts_per_token=8,
        is_text_layer=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.logit_soft_cap = logit_soft_cap
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rope_wavelength = rope_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_partial_rotary_factor = rope_partial_rotary_factor
        self.use_bidirectional_attention = use_bidirectional_attention
        self.use_vision_bidirectional_attention = (
            use_vision_bidirectional_attention
        )
        self.is_global_attention = is_global_attention
        # For global attention layers, head_dim may be larger than local.
        self.global_head_dim = global_head_dim
        self.dropout = dropout
        self.is_kv_shared_layer = is_kv_shared_layer
        self.kv_shared_layer_index = kv_shared_layer_index
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.attention_k_eq_v = attention_k_eq_v
        self.num_global_key_value_heads = num_global_key_value_heads
        self.use_double_wide_mlp = use_double_wide_mlp
        self.enable_moe_block = enable_moe_block
        self.num_experts = num_experts
        self.expert_intermediate_dim = expert_intermediate_dim
        self.num_experts_per_token = num_experts_per_token
        self.is_text_layer = is_text_layer
        # KV-shared layers optionally use a wider MLP (double intermediate).
        self.actual_intermediate_dim = (
            intermediate_dim * 2
            if (use_double_wide_mlp and is_kv_shared_layer)
            else intermediate_dim
        )

        # Pre-attention normalization.
        self.pre_attention_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_attention_norm",
        )
        # Post-attention normalization (always present in Gemma4).
        self.post_attention_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_norm",
        )

        effective_head_dim = (
            global_head_dim
            if is_global_attention and global_head_dim is not None
            else head_dim
        )
        attention_cls = (
            Gemma4TextAttention if self.is_text_layer else Gemma4VisionAttention
        )
        self.attention = attention_cls(
            head_dim=effective_head_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            logit_soft_cap=logit_soft_cap,
            use_sliding_window_attention=use_sliding_window_attention,
            sliding_window_size=sliding_window_size,
            layer_norm_epsilon=layer_norm_epsilon,
            rope_wavelength=rope_wavelength,
            rope_scaling_factor=rope_scaling_factor,
            rope_partial_rotary_factor=rope_partial_rotary_factor,
            use_bidirectional_attention=use_bidirectional_attention,
            is_global_attention=is_global_attention,
            is_kv_shared_layer=is_kv_shared_layer,
            attention_k_eq_v=attention_k_eq_v,
            num_global_key_value_heads=num_global_key_value_heads,
            dropout=dropout,
            dtype=self.dtype_policy,
            name="attention",
        )

        if self.dropout > 0:
            self.attention_dropout = keras.layers.Dropout(rate=dropout)
            self.feedforward_dropout = keras.layers.Dropout(rate=dropout)

        # Pre-FFW normalization.
        self.pre_ffw_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_ffw_norm",
        )
        # Post-FFW normalization (always present in Gemma4).
        self.post_ffw_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_ffw_norm",
        )

        # Feed-forward network uses standard gated GELU activation.
        # `actual_intermediate_dim` may be 2x for KV-shared layers when
        # `use_double_wide_mlp=True` (E2B architecture).
        self.gating_ffw = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, self.actual_intermediate_dim),
            dtype=self.dtype_policy,
            name="ffw_gating",
        )
        self.gating_ffw_2 = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, self.actual_intermediate_dim),
            dtype=self.dtype_policy,
            name="ffw_gating_2",
        )
        self.ffw_linear = keras.layers.EinsumDense(
            equation="btf,fd->btd",
            output_shape=(None, self.hidden_dim),
            dtype=self.dtype_policy,
            name="ffw_linear",
        )

        # MoE blocks (26b-a4b architecture).  When enabled, EVERY decoder layer
        # runs a dense MLP in parallel with a sparse MoE block; the two outputs
        # are summed BEFORE the combined post-FFW norm.
        if enable_moe_block:
            assert num_experts is not None, (
                "`num_experts` must be set when `enable_moe_block=True`."
            )
            assert expert_intermediate_dim is not None, (
                "`expert_intermediate_dim` must be set when "
                "`enable_moe_block=True`."
            )
            # Separate pre-norm for the MoE path (pre_feedforward_layernorm_2).
            self.pre_ffw_norm_moe = RMSNormalization(
                epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="pre_ffw_norm_moe",
            )
            # A separate dense-path normalization (post FFW) exists natively in
            # the HF checkpoint for MoE blocks.
            self.post_ffw_norm_dense = RMSNormalization(
                epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="post_ffw_norm_dense",
            )
            # Post-norms for dense and MoE paths individually
            # (post_feedforward_layernorm_1 / _2).
            self.post_ffw_norm_moe_path = RMSNormalization(
                epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="post_ffw_norm_moe_path",
            )
            # Router: takes raw hidden_states, computes dispatch weights.
            self.moe_router = Gemma4Router(
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="moe_router",
            )
            # Expert bank: does the actual per-expert computation.
            self.moe_expert_bank = Gemma4MoEBlock(
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                expert_intermediate_dim=expert_intermediate_dim,
                dtype=self.dtype_policy,
                name="moe_expert_bank",
            )

        # Per-layer input gate (E4B).  Applies a token-conditioned residual
        # to the layer output: residual + norm(proj_up(GELU(gate(x)) * emb)).
        if hidden_size_per_layer_input > 0:
            self.per_layer_input_gate = keras.layers.Dense(
                hidden_size_per_layer_input,
                use_bias=False,
                dtype=self.dtype_policy,
                name="per_layer_input_gate",
            )
            self.per_layer_up_proj = keras.layers.Dense(
                hidden_dim,
                use_bias=False,
                dtype=self.dtype_policy,
                name="per_layer_up_proj",
            )
            self.post_per_layer_input_norm = RMSNormalization(
                epsilon=layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="post_per_layer_input_norm",
            )

    def build(self, input_shape):
        self.pre_attention_norm.build(input_shape)
        self.attention.build(input_shape)

        attn_out_shape, cache_shape = self.attention.compute_output_shape(
            input_shape
        )
        self.post_attention_norm.build(attn_out_shape)

        self.pre_ffw_norm.build(input_shape)
        self.gating_ffw.build(input_shape)
        self.gating_ffw_2.build(input_shape)

        ffn_shape = self.gating_ffw.compute_output_shape(input_shape)
        self.ffw_linear.build(ffn_shape)

        ffw_out_shape = self.ffw_linear.compute_output_shape(ffn_shape)
        self.post_ffw_norm.build(ffw_out_shape)

        # MoE extra layers (26b-a4b).
        if self.enable_moe_block:
            self.pre_ffw_norm_moe.build(input_shape)
            self.post_ffw_norm_dense.build(input_shape)
            self.post_ffw_norm_moe_path.build(input_shape)
            self.moe_router.build(input_shape)
            self.moe_expert_bank.build(input_shape)

        # Per-layer input gate (E4B).
        if self.hidden_size_per_layer_input > 0:
            self.per_layer_input_gate.build(input_shape)
            gate_out_shape = self.per_layer_input_gate.compute_output_shape(
                input_shape
            )
            self.per_layer_up_proj.build(gate_out_shape)
            up_out_shape = self.per_layer_up_proj.compute_output_shape(
                gate_out_shape
            )
            self.post_per_layer_input_norm.build(up_out_shape)

        # Text decoder layers have a layer_scalar (Buffer, non-trainable,
        # initialised to 1.0 — matches HF nn.parameter.Buffer behaviour).
        # Vision encoder blocks do NOT get this weight.
        if self.is_text_layer:
            self.layer_scalar = self.add_weight(
                name="layer_scalar",
                shape=(),
                initializer="ones",
                trainable=False,
            )

        self.built = True

    def _compute_image_bidirectional_attention_mask(self, vision_mask):
        """Allow image tokens to attend to each other within the same image."""
        bidirectional_mask = vision_mask

        # Left pad with 0.
        padded_mask = ops.cast(
            ops.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0),
            dtype="int32",
        )

        boundary = ops.cast(
            ops.greater(padded_mask[..., 1:], padded_mask[..., :-1]),
            dtype="int32",
        )
        numbered_boundary = ops.cumsum(boundary, -1)
        indices = ops.multiply(bidirectional_mask, numbered_boundary)

        indices_expanded_1 = ops.expand_dims(indices, 1)
        indices_expanded_2 = ops.expand_dims(indices, -1)

        mask = ops.logical_and(
            ops.equal(indices_expanded_1, indices_expanded_2),
            indices_expanded_2,
        )
        return mask

    def _compute_attention_mask(
        self,
        x,
        padding_mask,
        vision_mask,
        cache,
        cache_update_index,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            inputs=x, padding_mask=padding_mask, attention_mask=None
        )

        batch_size = ops.shape(x)[0]
        input_length = output_length = ops.shape(x)[1]
        if cache is not None:
            input_length = ops.shape(cache)[2]

        if self.use_bidirectional_attention:
            # For embedding models with bidirectional attention.
            # When there is no padding, return None (attend to everything).
            if decoder_mask is None:
                return None
            mask_1 = decoder_mask
            mask_2 = ops.transpose(mask_1, (0, 2, 1))
            return mask_1 * mask_2

        causal_mask = compute_causal_mask(
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            cache_index=cache_update_index,
        )

        # For local (sliding-window) layers, restrict the causal mask to the
        # sliding window BEFORE OR-ing with the vision bidirec mask.
        # This matches HF's ordering: (causal AND sliding) OR vision_bidirec.
        # Doing the AND after the OR would incorrectly block vision tokens that
        # are in the same image but more than sliding_window_size apart.
        if self.use_sliding_window_attention and not self.is_global_attention:
            causal_mask = self.attention._mask_sliding_window(
                causal_mask,
                cache_update_index=cache_update_index,
            )

        # Image tokens attend bidirectionally within the same image, but ONLY
        # for local (sliding-window) layers — matching HF's behaviour where
        # the `or_mask_function` is applied only to `sliding_attention` masks
        # and NOT to `full_attention` (global) masks.
        # (e.g. HF 2B has use_bidirectional_attention=None → purely causal.)
        if (
            vision_mask is not None
            and self.use_vision_bidirectional_attention
            and not self.is_global_attention
        ):
            bidirectional_image_mask = (
                self._compute_image_bidirectional_attention_mask(vision_mask)
            )
            causal_mask = ops.logical_or(causal_mask, bidirectional_image_mask)

        # Respect the padding mask.
        if decoder_mask is not None:
            causal_mask = ops.minimum(decoder_mask, causal_mask)

        return causal_mask

    def call(
        self,
        x,
        padding_mask=None,
        vision_mask=None,
        cache=None,
        cache_update_index=0,
        cache_update_mask=None,
        per_layer_input=None,
        shared_kv=None,
    ):
        # Clamp float16 to avoid overflow.
        is_float16 = keras.backend.standardize_dtype(x.dtype) == "float16"
        if is_float16:
            x = ops.clip(x, -65504, 65504)

        # === Attention sub-block ===
        residual = x
        normalized_x = self.pre_attention_norm(x)
        attention_mask = self._compute_attention_mask(
            normalized_x, padding_mask, vision_mask, cache, cache_update_index
        )
        if cache is not None:
            attention, new_cache = self.attention(
                normalized_x,
                attention_mask=attention_mask,
                cache=cache,
                cache_update_index=cache_update_index,
                cache_update_mask=cache_update_mask,
                shared_kv=shared_kv,
            )
        else:
            attention, new_cache = self.attention(
                normalized_x,
                attention_mask=attention_mask,
                shared_kv=shared_kv,
            )

        # Post-attention norm (always applied in Gemma4).
        attention = self.post_attention_norm(attention)

        if self.dropout:
            attention = self.attention_dropout(attention)

        if is_float16:
            x = ops.cast(
                ops.clip(
                    ops.add(
                        ops.cast(residual, "float32"),
                        ops.cast(attention, "float32"),
                    ),
                    -65504,
                    65504,
                ),
                "float16",
            )
        else:
            x = residual + attention

        # === Feed-forward sub-block ===
        residual = x

        if self.enable_moe_block:
            # === Parallel Dense + MoE paths (26b-a4b architecture) ===
            # Dense path: pre_ffw_norm → dense MLP → post_ffw_norm_dense
            # HOTFIX: use direct matmul (same as the non-MoE path) to bypass
            # EinsumDense graph tracer bugs.
            normalized_x = self.pre_ffw_norm(x)
            x1 = ops.matmul(normalized_x, self.gating_ffw.kernel)
            x2 = ops.matmul(normalized_x, self.gating_ffw_2.kernel)
            dense_out = keras.activations.gelu(x1, approximate=True) * x2
            dense_out = ops.matmul(dense_out, self.ffw_linear.kernel)
            dense_out = self.post_ffw_norm_dense(dense_out)

            # MoE path: router uses raw x; expert bank uses pre_ffw_norm_moe(x)
            dispatch_weights = self.moe_router(x)  # [T, E]
            moe_in = self.pre_ffw_norm_moe(x)
            # Flatten for expert bank.
            shape = ops.shape(moe_in)
            moe_in_flat = ops.reshape(moe_in, (-1, shape[-1]))  # [T, H]
            expert_out = self.moe_expert_bank(moe_in_flat)  # [E, T, H]
            # Weighted sum: dispatch_weights [T, E] → [E, T] for broadcasting.
            dw = ops.transpose(
                ops.cast(dispatch_weights, expert_out.dtype), (1, 0)
            )  # [E, T]
            moe_out = ops.sum(expert_out * dw[:, :, None], axis=0)  # [T, H]
            moe_out = ops.reshape(moe_out, shape)  # [B, S, H]
            moe_out = self.post_ffw_norm_moe_path(moe_out)

            # Sum both paths, then shared post-FFW norm.
            x = dense_out + moe_out
        else:
            # === Standard dense FFW path ===
            # HOTFIX: Replacing EinsumDense with direct matmul to bypass graph
            # tracer bugs.
            normalized_x = self.pre_ffw_norm(x)
            x1 = ops.matmul(normalized_x, self.gating_ffw.kernel)
            x2 = ops.matmul(normalized_x, self.gating_ffw_2.kernel)

            x = keras.activations.gelu(x1, approximate=True) * x2
            x = ops.matmul(x, self.ffw_linear.kernel)

        # Post-FFW norm (shared; always applied in Gemma4).
        x = self.post_ffw_norm(x)

        if self.dropout:
            x = self.feedforward_dropout(x)

        if is_float16:
            x = ops.cast(
                ops.clip(
                    ops.add(
                        ops.cast(residual, "float32"), ops.cast(x, "float32")
                    ),
                    -65504,
                    65504,
                ),
                "float16",
            )
        else:
            x = residual + x

        # Per-layer input gate (E4B): gated residual conditioned on per-token
        # per-layer embedding. Applied AFTER the FFW residual and BEFORE the
        # layer_scalar (matching HF Gemma4DecoderLayer forward() order).
        if self.hidden_size_per_layer_input > 0 and per_layer_input is not None:
            residual = x
            gated = self.per_layer_input_gate(x)
            gated = keras.activations.gelu(gated, approximate=True)
            gated = gated * ops.cast(per_layer_input, gated.dtype)
            gated = self.per_layer_up_proj(gated)
            gated = self.post_per_layer_input_norm(gated)
            x = residual + gated

        # Text decoder layers scale the output by their layer_scalar.
        # Applied AFTER the per-layer input gate (matching HF order).
        if self.is_text_layer:
            x = x * ops.cast(self.layer_scalar, x.dtype)

        return x, new_cache

    def compute_output_shape(self, input_shape):
        attn_out_shape, cache_shape = self.attention.compute_output_shape(
            input_shape
        )
        return input_shape, cache_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "logit_soft_cap": self.logit_soft_cap,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "rope_wavelength": self.rope_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rope_partial_rotary_factor": self.rope_partial_rotary_factor,
                "use_bidirectional_attention": self.use_bidirectional_attention,
                "use_vision_bidirectional_attention": (
                    self.use_vision_bidirectional_attention
                ),
                "is_global_attention": self.is_global_attention,
                "global_head_dim": self.global_head_dim,
                "is_kv_shared_layer": self.is_kv_shared_layer,
                "kv_shared_layer_index": self.kv_shared_layer_index,
                "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
                "attention_k_eq_v": self.attention_k_eq_v,
                "num_global_key_value_heads": self.num_global_key_value_heads,
                "use_double_wide_mlp": self.use_double_wide_mlp,
                "enable_moe_block": self.enable_moe_block,
                "num_experts": self.num_experts,
                "expert_intermediate_dim": self.expert_intermediate_dim,
                "num_experts_per_token": self.num_experts_per_token,
                "is_text_layer": self.is_text_layer,
            }
        )
        return config


class Gemma4VisionDecoderBlock(keras.layers.Layer):
    """Vision decoder block for Gemma4.

    This operates strictly on images and disables MoE arrays, explicit layer
    scalars, and global sliding windows.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        layer_norm_epsilon=1e-6,
        rope_wavelength=10_000.0,
        dropout=0,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rope_wavelength = rope_wavelength
        self.dropout = dropout
        self.use_clipped_linears = use_clipped_linears

        self.pre_attention_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_attention_norm",
        )
        self.post_attention_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_norm",
        )

        self.attention = Gemma4VisionAttention(
            head_dim=head_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            layer_norm_epsilon=layer_norm_epsilon,
            rope_wavelength=rope_wavelength,
            dropout=dropout,
            use_clipped_linears=use_clipped_linears,
            dtype=self.dtype_policy,
            name="attention",
        )

        if self.dropout > 0:
            self.attention_dropout = keras.layers.Dropout(rate=dropout)
            self.feedforward_dropout = keras.layers.Dropout(rate=dropout)

        self.pre_ffw_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_ffw_norm",
        )
        self.post_ffw_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_ffw_norm",
        )

        self.gating_ffw = Gemma4ClippableEinsumDense(
            equation="btd,df->btf",
            output_shape=(None, intermediate_dim),
            use_clipped_linears=use_clipped_linears,
            dtype=self.dtype_policy,
            name="ffw_gating",
        )
        self.gating_ffw_2 = Gemma4ClippableEinsumDense(
            equation="btd,df->btf",
            output_shape=(None, intermediate_dim),
            use_clipped_linears=use_clipped_linears,
            dtype=self.dtype_policy,
            name="ffw_gating_2",
        )
        self.ffw_linear = Gemma4ClippableEinsumDense(
            equation="btf,fd->btd",
            output_shape=(None, hidden_dim),
            use_clipped_linears=use_clipped_linears,
            dtype=self.dtype_policy,
            name="ffw_linear",
        )

    def build(self, input_shape):
        self.pre_attention_norm.build(input_shape)
        self.attention.build(input_shape)

        attn_out_shape, cache_shape = self.attention.compute_output_shape(
            input_shape
        )
        self.post_attention_norm.build(attn_out_shape)

        self.pre_ffw_norm.build(input_shape)
        self.gating_ffw.build(input_shape)
        self.gating_ffw_2.build(input_shape)

        ffn_shape = self.gating_ffw.compute_output_shape(input_shape)
        self.ffw_linear.build(ffn_shape)

        ffw_out_shape = self.ffw_linear.compute_output_shape(ffn_shape)
        self.post_ffw_norm.build(ffw_out_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape, None

    def call(self, x, position_ids=None):
        # Clamp float16 to avoid overflow.
        is_float16 = keras.backend.standardize_dtype(x.dtype) == "float16"
        if is_float16:
            x = ops.clip(x, -65504, 65504)

        # === Attention sub-block ===
        residual = x
        normalized_x = self.pre_attention_norm(x)

        # Calculate mask where both x and y positions are NOT -1
        attention_mask = None
        if position_ids is not None:
            # position_ids shape is (B, Tokens, 2)
            attention_mask = ops.any(ops.not_equal(position_ids, -1), axis=-1)

        attention, _ = self.attention(
            normalized_x,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # Post-attention norm (always applied in Gemma4).
        attention = self.post_attention_norm(attention)

        if self.dropout:
            attention = self.attention_dropout(attention)

        if is_float16:
            x = ops.cast(
                ops.clip(
                    ops.add(
                        ops.cast(residual, "float32"),
                        ops.cast(attention, "float32"),
                    ),
                    -65504,
                    65504,
                ),
                "float16",
            )
        else:
            x = residual + attention

        # === Feed-forward sub-block ===
        residual = x

        # === Standard dense FFW path ===
        normalized_x = self.pre_ffw_norm(x)
        x1 = self.gating_ffw(normalized_x)
        x2 = self.gating_ffw_2(normalized_x)

        x = keras.activations.gelu(x1, approximate=True) * x2
        x = self.ffw_linear(x)

        # Post-FFW norm (shared; always applied in Gemma4).
        x = self.post_ffw_norm(x)

        if self.dropout:
            x = self.feedforward_dropout(x)

        if is_float16:
            x = ops.cast(
                ops.clip(
                    ops.add(
                        ops.cast(residual, "float32"), ops.cast(x, "float32")
                    ),
                    -65504,
                    65504,
                ),
                "float16",
            )
        else:
            x = residual + x

        return x, None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "rope_wavelength": self.rope_wavelength,
            }
        )
        return config
