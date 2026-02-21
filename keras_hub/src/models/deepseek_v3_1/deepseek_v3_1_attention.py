"""Multi-head Latent Attention (MLA) implementation for DeepSeek V3.1."""

import keras
from keras import ops


class DeepSeekV3_1Attention(keras.layers.Layer):
    """Multi-head Latent Attention (MLA) layer for DeepSeek V3.1.

    Implements the MLA architecture from the DeepSeek-V3 paper (Section 2.1.1).
    Key features:
    - Low-rank compression of K/V via down-projection to c_kv latent
    - Decoupled RoPE keys (k_rope) kept separate from content keys
    - During inference, only (c_kv, k_rope) need to be cached — not full K/V
    - W_UK and W_UV absorbed as raw weights (not Dense layers) for the
      MLA absorption trick: q_nope @ W_UK directly scores against c_kv

    YaRN (Yet another RoPE extensioN) is used for long-context scaling,
    applied only to the RoPE components as per Section 4.3.
    """

    def __init__(
        self,
        hidden_dim,
        num_query_heads,
        num_key_value_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        yarn_beta_fast=32,
        yarn_beta_slow=1,
        yarn_mscale=1.0,
        yarn_mscale_all_dim=0.0,
        attention_dropout=0.0,
        kernel_initializer="glorot_uniform",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.attention_dropout = attention_dropout
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow
        self.yarn_mscale = yarn_mscale
        self.yarn_mscale_all_dim = yarn_mscale_all_dim

        # Query low-rank compression (eq. 6-8)
        self.q_down_proj = keras.layers.Dense(
            q_lora_rank, dtype=dtype, name="q_down_proj"
        )
        self.q_up_nope_proj = keras.layers.Dense(
            num_query_heads * qk_nope_head_dim, dtype=dtype, name="q_up_nope_proj"
        )
        self.q_up_rope_proj = keras.layers.Dense(
            num_query_heads * qk_rope_head_dim, dtype=dtype, name="q_up_rope_proj"
        )

        # KV low-rank compression (eq. 1-5)
        self.kv_down_proj = keras.layers.Dense(
            kv_lora_rank, dtype=dtype, name="kv_down_proj"
        )
        self.k_rope_proj = keras.layers.Dense(
            qk_rope_head_dim, dtype=dtype, name="k_rope_proj"
        )

        # W_UK and W_UV: stored as raw weights for the MLA absorption trick.
        # q_nope @ W_UK^T computes content attention scores against c_kv directly,
        # avoiding materializing full per-head K tensors during inference.
        _weight_dtype = (
            self.dtype
            if not isinstance(self.dtype, keras.mixed_precision.Policy)
            else self.dtype.compute_dtype
        )
        self.w_uk = self.add_weight(
            shape=(num_query_heads * qk_nope_head_dim, kv_lora_rank),
            initializer=kernel_initializer,
            name="w_uk",
            dtype=_weight_dtype,
        )
        self.w_uv = self.add_weight(
            shape=(num_key_value_heads * v_head_dim, kv_lora_rank),
            initializer=kernel_initializer,
            name="w_uv",
            dtype=_weight_dtype,
        )

        self.output_proj = keras.layers.Dense(
            hidden_dim, dtype=dtype, name="output_proj"
        )
        self.dropout = keras.layers.Dropout(attention_dropout, dtype=dtype)

    def _get_yarn_inv_freq(self, dim, dtype):
        """Compute YaRN-scaled inverse frequencies for RoPE.

        YaRN (Section 4.3) extends RoPE to longer contexts by applying
        different scaling factors to different frequency bands:
        - High-frequency dimensions (short wavelength): interpolated less
        - Low-frequency dimensions (long wavelength): interpolated more

        The ramp function smoothly interpolates between no scaling (ramp=0)
        for high-frequency dims and full scaling (ramp=1) for low-frequency dims.

        FIX: Corrected the ramp direction — low-frequency (large wavelength)
        dimensions get more scaling, high-frequency ones get less.
        """
        # Base inverse frequencies: shape (dim//2,)
        freqs = 1.0 / (
            self.rope_max_wavelength ** (ops.arange(0, dim, 2, dtype="float32") / dim)
        )

        if self.rope_scaling_factor <= 1.0:
            return ops.cast(freqs, dtype), self.yarn_mscale

        # Wavelengths for each frequency dimension
        # freqs are inverse frequencies (rad/token), wavelengths = 2*pi / freq
        wavelengths = 2.0 * 3.14159265358979 / freqs  # shape (dim//2,)

        beta_fast = float(self.yarn_beta_fast)  # threshold for "high frequency"
        beta_slow = float(self.yarn_beta_slow)  # threshold for "low frequency"

        # Ramp: 0 for high-freq dims (short wavelength < beta_slow * ctx),
        #        1 for low-freq dims  (long wavelength > beta_fast * ctx)
        # FIX: ramp goes from 0 (no extra scaling) to 1 (full scaling).
        # High-freq (small wavelength relative to old_context_len) → ramp near 0
        # Low-freq  (large wavelength relative to old_context_len) → ramp near 1
        old_context_len = 4096.0
        ramp = (wavelengths / old_context_len - beta_slow) / (beta_fast - beta_slow)
        ramp = ops.clip(ramp, 0.0, 1.0)

        # Interpolation: blend between no scaling (factor=1) and full scaling
        # (factor=rope_scaling_factor) based on ramp
        # High-freq: factor ~ 1.0 (little or no scaling)
        # Low-freq:  factor ~ rope_scaling_factor (full scaling)
        scale_factor = (1.0 - ramp) * 1.0 + ramp * self.rope_scaling_factor
        scaled_freqs = freqs / scale_factor

        # Magnitude scaling factor for attention scores
        mscale = self.yarn_mscale
        if self.yarn_mscale_all_dim != 0.0:
            mscale = self.yarn_mscale_all_dim

        return ops.cast(scaled_freqs, dtype), mscale

    def _apply_rotary_embeddings(self, x, start_index, dtype):
        """Apply Rotary Position Embeddings (RoPE) to input tensor x.

        Args:
            x: Tensor of shape (batch, heads, seq_len, rope_head_dim)
            start_index: Position offset for KV cache decoding
            dtype: Target dtype for output
        Returns:
            Rotated tensor of same shape as x
        """
        seq_len = ops.shape(x)[-2]
        dim = self.qk_rope_head_dim
        inv_freq, mscale = self._get_yarn_inv_freq(dim, "float32")

        if start_index is None:
            start_index = 0

        positions = ops.arange(seq_len, dtype="float32")
        t = positions + ops.cast(start_index, "float32")
        freqs = ops.outer(t, inv_freq)  # (seq_len, dim//2)
        freqs = ops.concatenate([freqs, freqs], axis=-1)  # (seq_len, dim)
        freqs = ops.expand_dims(ops.expand_dims(freqs, 0), 0)  # (1, 1, seq, dim)

        cos = ops.cos(freqs) * ops.cast(mscale, "float32")
        sin = ops.sin(freqs) * ops.cast(mscale, "float32")

        x_fp32 = ops.cast(x, "float32")
        half = dim // 2
        x1 = x_fp32[..., :half]
        x2 = x_fp32[..., half:]

        # Standard rotate-half: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        cos_half = cos[..., :half]
        sin_half = sin[..., :half]
        rotated = ops.concatenate(
            [x1 * cos_half - x2 * sin_half, x1 * sin_half + x2 * cos_half],
            axis=-1,
        )

        return ops.cast(rotated, dtype)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=0,
        training=False,
    ):
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]
        dtype = hidden_states.dtype

        # --- Query projection (eq. 6-9) ---
        c_q = ops.cast(self.q_down_proj(hidden_states), dtype)

        q_nope = ops.cast(self.q_up_nope_proj(c_q), dtype)
        q_nope = ops.reshape(
            q_nope,
            [batch_size, seq_len, self.num_query_heads, self.qk_nope_head_dim],
        )
        q_nope = ops.transpose(q_nope, [0, 2, 1, 3])  # (B, H, S, D_nope)

        q_rope = ops.cast(self.q_up_rope_proj(c_q), dtype)
        q_rope = ops.reshape(
            q_rope,
            [batch_size, seq_len, self.num_query_heads, self.qk_rope_head_dim],
        )
        q_rope = ops.transpose(q_rope, [0, 2, 1, 3])  # (B, H, S, D_rope)

        # --- KV projection (eq. 1, 3) ---
        c_kv = ops.cast(self.kv_down_proj(hidden_states), dtype)  # (B, S, kv_lora_rank)
        k_rope = ops.cast(self.k_rope_proj(hidden_states), dtype)  # (B, S, D_rope)
        k_rope = ops.expand_dims(
            k_rope, axis=1
        )  # (B, 1, S, D_rope) — shared across heads

        # Apply RoPE to query and key rope components
        q_rope = self._apply_rotary_embeddings(q_rope, cache_update_index, dtype)
        k_rope = self._apply_rotary_embeddings(k_rope, cache_update_index, dtype)

        # --- KV Cache update ---
        if cache is not None:
            c_kv_cache, k_rope_cache = cache
            c_kv_update = ops.slice_update(c_kv_cache, [0, cache_update_index, 0], c_kv)
            k_rope_val = ops.squeeze(k_rope, axis=1)  # (B, S, D_rope)
            k_rope_update = ops.slice_update(
                k_rope_cache, [0, cache_update_index, 0], k_rope_val
            )
            new_cache = (c_kv_update, k_rope_update)

            # Use full cache tensors; causal mask handles future positions
            c_kv_hist = c_kv_update
            k_rope_hist = k_rope_update
            k_rope_hist = ops.expand_dims(k_rope_hist, axis=1)  # (B, 1, T, D_rope)
            hist_len = ops.shape(c_kv_cache)[1]
        else:
            new_cache = None
            c_kv_hist = c_kv
            k_rope_hist = k_rope
            hist_len = seq_len

        # --- MLA Attention Score Computation ---
        # Content scores: q_nope @ W_UK^T @ c_kv^T
        # This is the MLA absorption trick — we score q against the compressed
        # latent c_kv directly using the absorbed W_UK matrix
        w_uk_reshaped = ops.reshape(
            self.w_uk,
            [self.num_query_heads, self.qk_nope_head_dim, self.kv_lora_rank],
        )
        # q_pe: (B, H, S, kv_lora_rank) — queries projected into latent KV space
        q_pe = ops.cast(ops.einsum("bhld,hdk->bhlk", q_nope, w_uk_reshaped), dtype)

        # Score against cached KV latents: (B, H, S, T)
        c_kv_hist_t = ops.expand_dims(
            ops.transpose(c_kv_hist, [0, 2, 1]), axis=1
        )  # (B, 1, kv_lora_rank, T)
        score_content = ops.cast(ops.matmul(q_pe, c_kv_hist_t), dtype)

        # RoPE scores: (B, H, S, T)
        k_rope_hist_t = ops.transpose(k_rope_hist, [0, 1, 3, 2])  # (B, 1, D_rope, T)
        score_rope = ops.cast(ops.matmul(q_rope, k_rope_hist_t), dtype)

        # Combined scores with scaling (eq. 10)
        scores = (score_content + score_rope) * (
            1.0
            / ops.sqrt(ops.cast(self.qk_nope_head_dim + self.qk_rope_head_dim, dtype))
        )

        # --- XLA-Compatible Causal Mask ---
        idx = (
            ops.cast(cache_update_index, "int32")
            if cache_update_index is not None
            else 0
        )
        # i_indices shape: (seq_len, 1), j_indices shape: (1, hist_len)
        # Both have static shapes known at compile time → XLA-friendly
        i_indices = ops.arange(seq_len, dtype="int32")[:, None] + idx
        j_indices = ops.arange(hist_len, dtype="int32")[None, :]
        causal_mask = ops.cast(i_indices >= j_indices, dtype="bool")
        causal_mask = ops.reshape(causal_mask, [1, 1, seq_len, hist_len])

        if attention_mask is not None:
            attention_mask = ops.cast(attention_mask, "bool")
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif len(attention_mask.shape) == 3:
                attention_mask = attention_mask[:, None, :, :]
            mask = ops.logical_and(causal_mask, attention_mask)
        else:
            mask = causal_mask

        mask = ops.cast(mask, dtype=dtype)
        scores = scores + (1.0 - mask) * -1e9

        attn_weights = ops.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        # --- Value computation via W_UV absorption ---
        # ctx_latent: (B, H, S, kv_lora_rank)
        c_kv_hist_exp = ops.expand_dims(c_kv_hist, axis=1)  # (B, 1, T, kv_lora_rank)
        ctx_latent = ops.matmul(attn_weights, c_kv_hist_exp)

        # Project from latent space to value space using W_UV
        w_uv_reshaped = ops.reshape(
            self.w_uv,
            [self.num_key_value_heads, self.v_head_dim, self.kv_lora_rank],
        )
        attn_out = ops.einsum("bhlk,hvk->bhlv", ctx_latent, w_uv_reshaped)

        # Merge heads and project output (eq. 11)
        attn_out = ops.transpose(attn_out, [0, 2, 1, 3])
        attn_out = ops.reshape(
            attn_out,
            [batch_size, seq_len, self.num_key_value_heads * self.v_head_dim],
        )
        output = self.output_proj(attn_out)

        if cache is not None:
            return output, new_cache

        return output
