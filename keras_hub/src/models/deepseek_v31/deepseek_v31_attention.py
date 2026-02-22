"""DeepSeek V31 Multi-head Latent Attention layer."""

import keras
from keras import ops


class DeepSeekV31Attention(keras.layers.Layer):
    """Multi-head Latent Attention (MLA) for DeepSeek V31.

    Implements the MLA architecture from Section 2.1.1 of the DeepSeek-V3
    paper. MLA reduces KV cache size by compressing keys and values through
    a shared low-rank latent vector `c_kv` of dimension `kv_lora_rank`,
    rather than caching full per-head K and V tensors.

    The attention computation splits queries and keys into two components:

    - **Content (nope) component**: `q_nope`, `k_nope` — carries semantic
      information, does not receive positional encoding.
    - **RoPE component**: `q_rope`, `k_rope` — receives Rotary Position
      Embeddings (RoPE) for positional awareness.

    During inference only `(c_kv, k_rope)` need to be stored in the KV cache,
    not the full materialized K and V tensors. The content keys and values are
    recovered via the absorption matrices `w_uk` and `w_uv`:

        score = q_nope @ w_uk.T @ c_kv.T + q_rope @ k_rope.T

    YaRN (Yet another RoPE extensioN, Section 4.3) is applied to the RoPE
    frequencies to support contexts longer than the training length. Different
    frequency bands are scaled differently: high-frequency dimensions
    (short wavelength) receive little scaling while low-frequency dimensions
    (long wavelength) are scaled more aggressively.

    Args:
        hidden_dim: int. Dimensionality of model hidden states.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value heads. For MLA this
            equals `num_query_heads` since KV are recovered from a shared
            latent.
        q_lora_rank: int. Rank of the query down-projection.
        kv_lora_rank: int. Rank of the shared KV latent `c_kv`. This is the
            per-layer KV cache size per token.
        qk_nope_head_dim: int. Per-head dimension for the content (non-RoPE)
            query and key components.
        qk_rope_head_dim: int. Per-head dimension for the RoPE query and key
            components.
        v_head_dim: int. Per-head dimension for values.
        rope_max_wavelength: int. Base wavelength for RoPE inverse frequencies.
            Defaults to `10000`.
        rope_scaling_factor: float. YaRN context extension scale factor.
            Values greater than 1.0 extend the effective context length.
            Defaults to `1.0`.
        yarn_beta_fast: int. YaRN ramp upper threshold. Dimensions with
            wavelength above `yarn_beta_fast * original_max_position` are
            treated as low-frequency and receive full scaling. Defaults to
            `32`.
        yarn_beta_slow: int. YaRN ramp lower threshold. Dimensions with
            wavelength below `yarn_beta_slow * original_max_position` are
            treated as high-frequency and receive no scaling. Defaults to
            `1`.
        yarn_mscale: float. YaRN magnitude scaling factor applied to attention
            cosine/sine embeddings. Defaults to `1.0`.
        yarn_mscale_all_dim: float. If non-zero, overrides `yarn_mscale` for
            all dimensions. Defaults to `0.0`.
        yarn_original_max_position_embeddings: int. The context length used
            during pre-training, used as the reference for YaRN ramp
            thresholds. Defaults to `4096`.
        attention_dropout: float. Dropout probability applied to attention
            weights. Defaults to `0.0`.
        kernel_initializer: string or initializer. Initializer for Dense and
            raw weight matrices. Defaults to `"glorot_uniform"`.

    Example:

    ```python
    attn = keras_hub.layers.DeepSeekV31Attention(
        hidden_dim=512,
        num_query_heads=8,
        num_key_value_heads=8,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
    )
    hidden = keras.random.normal((2, 16, 512))
    output = attn(hidden)  # (2, 16, 512)
    ```

    Reference:
     - [DeepSeek-AI et al., 2024](https://arxiv.org/abs/2412.19437)
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
        yarn_original_max_position_embeddings=4096,
        attention_dropout=0.0,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)

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
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow
        self.yarn_mscale = yarn_mscale
        self.yarn_mscale_all_dim = yarn_mscale_all_dim
        self.yarn_original_max_position_embeddings = (
            yarn_original_max_position_embeddings
        )
        self.attention_dropout = attention_dropout
        self.kernel_initializer = kernel_initializer

        # Query low-rank compression (eq. 6-8).
        self.q_down_proj = keras.layers.Dense(
            q_lora_rank,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name="q_down_proj",
        )
        self.q_up_nope_proj = keras.layers.Dense(
            num_query_heads * qk_nope_head_dim,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name="q_up_nope_proj",
        )
        self.q_up_rope_proj = keras.layers.Dense(
            num_query_heads * qk_rope_head_dim,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name="q_up_rope_proj",
        )

        # KV low-rank compression (eq. 1-5).
        self.kv_down_proj = keras.layers.Dense(
            kv_lora_rank,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name="kv_down_proj",
        )
        self.k_rope_proj = keras.layers.Dense(
            qk_rope_head_dim,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name="k_rope_proj",
        )

        self.output_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name="output_proj",
        )
        self.dropout = keras.layers.Dropout(attention_dropout)

    def build(self, input_shape):
        # W_UK and W_UV are stored as raw weight matrices for the MLA
        # absorption trick. Rather than materializing full per-head K and V
        # tensors, we score queries against c_kv directly using these absorbed
        # matrices:  score = q_nope @ W_UK.T @ c_kv.T
        self.w_uk = self.add_weight(
            shape=(
                self.num_query_heads * self.qk_nope_head_dim,
                self.kv_lora_rank,
            ),
            initializer=self.kernel_initializer,
            name="w_uk",
        )
        self.w_uv = self.add_weight(
            shape=(
                self.num_query_heads * self.v_head_dim,
                self.kv_lora_rank,
            ),
            initializer=self.kernel_initializer,
            name="w_uv",
        )

        self.q_down_proj.build(input_shape)
        q_down_shape = list(input_shape[:-1]) + [self.q_lora_rank]
        self.q_up_nope_proj.build(q_down_shape)
        self.q_up_rope_proj.build(q_down_shape)
        self.kv_down_proj.build(input_shape)
        self.k_rope_proj.build(input_shape)
        self.output_proj.build(
            list(input_shape[:-1]) + [self.num_key_value_heads * self.v_head_dim]
        )
        super().build(input_shape)

    def _yarn_inv_freq(self, dtype):
        """Return YaRN-scaled RoPE inverse frequencies and magnitude scale."""
        dim = self.qk_rope_head_dim
        freqs = 1.0 / (
            self.rope_max_wavelength ** (ops.arange(0, dim, 2, dtype="float32") / dim)
        )

        if self.rope_scaling_factor <= 1.0:
            return ops.cast(freqs, dtype), self.yarn_mscale

        # Wavelength = 2π / freq. High-freq → small wavelength, low-freq →
        # large wavelength. YaRN applies more scaling to low-freq dimensions.
        wavelengths = 2.0 * 3.14159265358979 / freqs
        old_ctx = float(self.yarn_original_max_position_embeddings)
        beta_slow = float(self.yarn_beta_slow)
        beta_fast = float(self.yarn_beta_fast)

        # ramp=0 → high-freq (no extra scaling), ramp=1 → low-freq (full scale)
        ramp = ops.clip(
            (wavelengths / old_ctx - beta_slow) / (beta_fast - beta_slow),
            0.0,
            1.0,
        )
        scale = (1.0 - ramp) + ramp * self.rope_scaling_factor
        scaled_freqs = freqs / scale

        mscale = (
            self.yarn_mscale_all_dim
            if self.yarn_mscale_all_dim != 0.0
            else self.yarn_mscale
        )
        return ops.cast(scaled_freqs, dtype), mscale

    def _apply_rope(self, x, start_index, dtype, inv_freq, mscale):
        """Apply Rotary Position Embeddings to x of shape (B, H, S, D)."""
        seq_len = ops.shape(x)[-2]
        start = 0 if start_index is None else ops.cast(start_index, "float32")
        positions = ops.arange(seq_len, dtype="float32") + start
        freqs = ops.concatenate([ops.outer(positions, inv_freq)] * 2, axis=-1)
        freqs = ops.expand_dims(ops.expand_dims(freqs, 0), 0)  # (1,1,S,D)
        cos = ops.cos(freqs) * ops.cast(mscale, "float32")
        sin = ops.sin(freqs) * ops.cast(mscale, "float32")
        x_fp32 = ops.cast(x, "float32")
        half = self.qk_rope_head_dim // 2
        rotated = ops.concatenate(
            [
                x_fp32[..., :half] * cos[..., :half]
                - x_fp32[..., half:] * sin[..., half:],
                x_fp32[..., :half] * sin[..., half:]
                + x_fp32[..., half:] * cos[..., half:],
            ],
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

        # Query projections.
        c_q = self.q_down_proj(hidden_states)
        q_nope = ops.reshape(
            self.q_up_nope_proj(c_q),
            [batch_size, seq_len, self.num_query_heads, self.qk_nope_head_dim],
        )
        q_nope = ops.transpose(q_nope, [0, 2, 1, 3])  # (B, H, S, D_nope)

        q_rope = ops.reshape(
            self.q_up_rope_proj(c_q),
            [batch_size, seq_len, self.num_query_heads, self.qk_rope_head_dim],
        )
        q_rope = ops.transpose(q_rope, [0, 2, 1, 3])  # (B, H, S, D_rope)

        # KV projections.
        c_kv = self.kv_down_proj(hidden_states)  # (B, S, kv_lora_rank)
        k_rope = self.k_rope_proj(hidden_states)  # (B, S, D_rope)
        k_rope = ops.expand_dims(k_rope, axis=1)  # (B, 1, S, D_rope)

        # Apply RoPE to positional components.
        inv_freq, mscale = self._yarn_inv_freq("float32")  # compute once
        q_rope = self._apply_rope(q_rope, cache_update_index, dtype, inv_freq, mscale)
        k_rope = self._apply_rope(k_rope, cache_update_index, dtype, inv_freq, mscale)

        # KV cache: read full history and write current step.
        if cache is not None:
            c_kv_cache, k_rope_cache = cache
            c_kv = ops.slice_update(c_kv_cache, [0, cache_update_index, 0], c_kv)
            k_rope_sq = ops.squeeze(k_rope, axis=1)
            k_rope = ops.expand_dims(
                ops.slice_update(k_rope_cache, [0, cache_update_index, 0], k_rope_sq),
                axis=1,
            )
            new_cache = (c_kv, ops.squeeze(k_rope, axis=1))
            hist_len = ops.shape(c_kv_cache)[1]
        else:
            new_cache = None
            hist_len = seq_len

        # Content attention scores via MLA absorption trick (eq. 10).
        # q_nope @ w_uk.T projects queries into latent KV space, then scores
        # against c_kv without materialising per-head K tensors.
        w_uk = ops.reshape(
            self.w_uk,
            [self.num_query_heads, self.qk_nope_head_dim, self.kv_lora_rank],
        )
        q_latent = ops.einsum("bhsd,hdk->bhsk", q_nope, w_uk)  # (B,H,S,lora)
        c_kv_t = ops.expand_dims(ops.transpose(c_kv, [0, 2, 1]), axis=1)  # (B,1,lora,T)
        score_content = ops.matmul(q_latent, c_kv_t)  # (B,H,S,T)

        # RoPE attention scores.
        k_rope_t = ops.transpose(k_rope, [0, 1, 3, 2])  # (B,1,D_rope,T)
        score_rope = ops.matmul(q_rope, k_rope_t)  # (B,H,S,T)

        scale = ops.cast(
            1.0
            / ops.sqrt(
                ops.cast(self.qk_nope_head_dim + self.qk_rope_head_dim, "float32")
            ),
            dtype,
        )
        scores = (ops.cast(score_content, dtype) + ops.cast(score_rope, dtype)) * scale

        # XLA-compatible causal mask using static shapes.
        idx = ops.cast(0 if cache_update_index is None else cache_update_index, "int32")
        i_idx = ops.arange(seq_len, dtype="int32")[:, None] + idx
        j_idx = ops.arange(hist_len, dtype="int32")[None, :]
        causal = ops.reshape(ops.cast(i_idx >= j_idx, dtype), [1, 1, seq_len, hist_len])

        if attention_mask is not None:
            pad = ops.cast(attention_mask, dtype)
            if len(pad.shape) == 2:
                pad = pad[:, None, None, :]
            elif len(pad.shape) == 3:
                pad = pad[:, None, :, :]
            mask = ops.cast(
                ops.logical_and(ops.cast(causal, "bool"), ops.cast(pad, "bool")),
                dtype,
            )
        else:
            mask = causal

        large_neg = ops.cast(-3e4 if scores.dtype == "float16" else -1e9, scores.dtype)
        scores = scores + (1.0 - mask) * large_neg
        attn_weights = ops.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        # Value computation via W_UV absorption.
        c_kv_exp = ops.expand_dims(c_kv, axis=1)  # (B,1,T,lora)
        ctx = ops.matmul(attn_weights, c_kv_exp)  # (B,H,S,lora)
        w_uv = ops.reshape(
            self.w_uv,
            [self.num_query_heads, self.kv_lora_rank, self.v_head_dim],
        )
        attn_out = ops.einsum("bhsk,hvk->bhsv", ctx, w_uv)  # (B,H,S,v_dim)

        attn_out = ops.reshape(
            ops.transpose(attn_out, [0, 2, 1, 3]),
            [batch_size, seq_len, self.num_key_value_heads * self.v_head_dim],
        )
        output = self.output_proj(attn_out)

        if cache is not None:
            return output, new_cache
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "yarn_beta_fast": self.yarn_beta_fast,
                "yarn_beta_slow": self.yarn_beta_slow,
                "yarn_mscale": self.yarn_mscale,
                "yarn_mscale_all_dim": self.yarn_mscale_all_dim,
                "yarn_original_max_position_embeddings": (
                    self.yarn_original_max_position_embeddings
                ),
                "attention_dropout": self.attention_dropout,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config
