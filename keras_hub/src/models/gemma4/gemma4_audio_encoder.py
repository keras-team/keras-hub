import math

import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4FrozenNorm
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4VNorm


class Gemma4AudioRelativePositionEmbedding(keras.layers.Layer):
    """Sinusoidal relative position embedding for the audio conformer attention.

    Computes attention logits as `term_ac + term_bd_shifted`, where:
    - `term_ac` is the query-key content interaction.
    - `term_bd_shifted` is the query-position sinusoidal bias after applying
      the relative shift trick.

    Reference: Universal Speech Model (USM).

    Args:
        hidden_size: int. The conformer hidden dimension.
        num_heads: int. Number of attention heads.
        context_left: int. Left context size (inclusive of chunk position).
        context_right: int. Right context size.
        epsilon: float. Small value for numerical stability.
        dtype: Compute dtype.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        context_left,
        context_right,
        epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_backward = max(0, context_left - 1)
        self.max_forward = context_right
        self.epsilon = epsilon

        self.pos_proj = self.add_weight(
            name="pos_proj",
            shape=(self.hidden_size, self.num_heads * self.head_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        # Precompute inverse timescales for sinusoidal position encoding.
        # HF uses all-ones (non-persistent buffer), not log-spaced timescales.
        num_timescales = self.hidden_size // 2
        self._inv_timescales = np.ones((1, 1, num_timescales), dtype="float32")

    def build(self, input_shape):
        self.built = True

    def _get_timing_signal(self, positions):
        """Compute sinusoidal timing for given positions.

        Args:
            positions: int tensor of shape `(1, F_span)`.

        Returns:
            float tensor of shape `(1, F_span, hidden_size)`.
        """
        pos = ops.cast(positions, "float32")
        pos = ops.expand_dims(pos, axis=-1)  # (1, F_span, 1)
        inv_ts = ops.cast(self._inv_timescales, "float32")
        scaled = pos * inv_ts  # (1, F_span, D/2)
        signal = ops.concatenate(
            [ops.sin(scaled), ops.cos(scaled)], axis=-1
        )  # (1, F_span, D)
        return ops.cast(signal, self.compute_dtype)

    def _relative_shift(
        self,
        term_bd,
        batch_size,
        num_heads,
        num_blocks,
        block_size,
        context_size,
        max_span_plus_1,
    ):
        """Apply the relative-shift trick to produce [B, N, U, W, C] logits.

        Args:
            term_bd: float tensor `(B, N, U, W, F_span)`.

        Returns:
            float tensor `(B, N, U, W, C)`.
        """
        pad_amount = (context_size + 1) - max_span_plus_1
        # Pad last dimension on the right.
        term_bd_padded = ops.pad(
            term_bd, [(0, 0), (0, 0), (0, 0), (0, 0), (0, pad_amount)]
        )  # (B, N, U, W, C+1)
        term_bd_reshaped = ops.reshape(
            term_bd_padded,
            [
                batch_size,
                num_heads,
                num_blocks,
                block_size * (context_size + 1),
            ],
        )
        term_bd_sliced = term_bd_reshaped[:, :, :, : block_size * context_size]
        return ops.reshape(
            term_bd_sliced,
            [batch_size, num_heads, num_blocks, block_size, context_size],
        )

    def call(self, queries, keys):
        """Compute relative-position-biased attention logits.

        Args:
            queries: float tensor `(B, U, W, N, H)`.
            keys: float tensor `(B, U, C, N, H)`.

        Returns:
            float tensor `(B, N, U, W, C)`.
        """
        batch_size = ops.shape(queries)[0]
        num_blocks = ops.shape(queries)[1]
        block_size = ops.shape(queries)[2]
        context_size = ops.shape(keys)[2]
        max_span_plus_1 = self.max_backward + self.max_forward + 1

        # Relative position indices: [max_backward, ..., -max_forward].
        pos_indices = ops.expand_dims(
            ops.arange(
                self.max_backward, -(self.max_forward + 1), -1, dtype="int32"
            ),
            axis=0,
        )  # (1, F_span)

        sin_emb = self._get_timing_signal(pos_indices)  # (1, F_span, D)
        # Project to (1, F_span, N*H) and reshape.
        projected = ops.matmul(
            sin_emb, ops.cast(self.pos_proj, self.compute_dtype)
        )
        sin_emb = ops.reshape(
            projected, [1, max_span_plus_1, self.num_heads, self.head_dim]
        )  # (1, F_span, N, H)
        sin_emb = ops.squeeze(sin_emb, axis=0)  # (F_span, N, H)

        # term_ac: content-content interaction (B, N, U, W, C).
        queries_perm = ops.transpose(
            queries, (0, 3, 1, 2, 4)
        )  # (B, N, U, W, H)
        keys_perm_t = ops.transpose(keys, (0, 3, 1, 4, 2))  # (B, N, U, H, C)
        term_ac = ops.matmul(queries_perm, keys_perm_t)  # (B, N, U, W, C)

        # term_bd: content-position interaction.
        # queries_perm: (B, N, U, W, H)
        # sin_emb: (F_span, N, H) → permute to (N, H, F_span)
        s_perm = ops.transpose(sin_emb, (1, 2, 0))  # (N, H, F_span)
        q_flat = ops.reshape(
            queries_perm,
            [
                batch_size,
                self.num_heads,
                num_blocks * block_size,
                self.head_dim,
            ],
        )  # (B, N, U*W, H)
        term_bd_flat = ops.matmul(q_flat, s_perm)  # (B, N, U*W, F_span)
        term_bd_raw = ops.reshape(
            term_bd_flat,
            [
                batch_size,
                self.num_heads,
                num_blocks,
                block_size,
                max_span_plus_1,
            ],
        )
        term_bd = self._relative_shift(
            term_bd_raw,
            batch_size,
            self.num_heads,
            num_blocks,
            block_size,
            context_size,
            max_span_plus_1,
        )
        return term_ac + term_bd

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "context_left": self.max_backward + 1,
                "context_right": self.max_forward,
                "epsilon": self.epsilon,
            }
        )
        return config


class Gemma4AudioAttention(keras.layers.Layer):
    """Chunk-based local attention for the audio conformer.

    Queries and keys are split into non-overlapping blocks of size
    `chunk_size`, and each query block attends to a context window of
    `chunk_size + max_past_horizon + max_future_horizon` key positions.
    Relative-position sinusoidal biases are applied via the RPE module.

    Args:
        hidden_size: int. Conformer hidden dimension.
        num_heads: int. Number of attention heads.
        chunk_size: int. Block size for chunked attention.
        context_left: int. Left attention context (inclusive).
        context_right: int. Right attention context.
        logit_cap: float. Soft-cap applied to attention logits.
        invalid_logit_value: float. Logit value for masked-out positions.
        norm_eps: float. Epsilon for internal norms.
        dtype: Compute dtype.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        chunk_size,
        context_left,
        context_right,
        logit_cap=50.0,
        invalid_logit_value=-1e9,
        norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.max_past_horizon = max(0, context_left - 1)
        self.max_future_horizon = context_right
        self.context_size = (
            chunk_size + self.max_past_horizon + self.max_future_horizon
        )
        self.logit_cap = logit_cap
        self.invalid_logit_value = invalid_logit_value
        self.norm_eps = norm_eps
        # Scaling constants matching HF Gemma4AudioAttention:
        # q_scale = (head_dim^-0.5) / log(2), k_scale = log(1+e) / log(2).
        self.q_scale = (self.head_dim**-0.5) / math.log(2.0)
        self.k_scale = math.log(1.0 + math.e) / math.log(2.0)

        input_shape = (None, None, self.hidden_size)
        self.q_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="q_proj",
        )
        self.k_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="k_proj",
        )
        self.v_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="v_proj",
        )
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)

        # Learned per-dimension scales (applied via softplus at forward time).
        self.per_dim_scale = self.add_weight(
            name="per_dim_scale",
            shape=(self.head_dim,),
            initializer="zeros",
            trainable=True,
        )

        self.rpe = Gemma4AudioRelativePositionEmbedding(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            context_left=self.max_past_horizon + 1,
            context_right=self.max_future_horizon,
            epsilon=self.norm_eps,
            dtype=self.dtype_policy,
            name="rpe",
        )
        self.rpe.build(input_shape)

    def build(self, input_shape):
        self.built = True

    def _convert_to_block(self, x):
        """Reshape `(B, T, ...)` → `(B, U, W, ...)` with right-zero-padding.

        Args:
            x: tensor of shape `(B, T)` or `(B, T, D)` or `(B, T, N, H)`.

        Returns:
            Tensor with time split into non-overlapping blocks of `chunk_size`.
        """
        B = ops.shape(x)[0]
        static_T = x.shape[1]
        W = self.chunk_size

        if isinstance(static_T, int):
            # Static length coords setups coord absolute triggers setups cords
            # triggers absolute
            pad_len = (-static_T) % W
            if pad_len > 0:
                zeros_row = ops.zeros_like(x[:, :1, ...])
                zero_pad = ops.tile(
                    zeros_row, [1, pad_len] + [1] * (x.ndim - 2)
                )
                x = ops.concatenate([x, zero_pad], axis=1)
            num_blocks = (static_T + pad_len) // W
        else:
            # Symbolic tracing coords absolute setups coordinators layouts list
            # coords triggers
            T = ops.shape(x)[1]
            num_blocks = T // W

        ndim = len(x.shape)
        if ndim == 2:
            return ops.reshape(x, [B, num_blocks, W])
        elif ndim == 3:
            return ops.reshape(x, [B, num_blocks, W, x.shape[2]])
        else:  # ndim == 4
            return ops.reshape(x, [B, num_blocks, W, x.shape[2], x.shape[3]])

    def _extract_block_context(self, x, num_blocks):
        """Extract overlapping context windows.

        Transforms `(B, T, ...)` → `(B, U, C, ...)`.


        Pads `max_past_horizon` zeros on the left and
        `max_future_horizon + chunk_size - 1` zeros on the right (static
        values), then gathers windows of size `context_size` with step
        `chunk_size` for each block.

        Args:
            x: tensor of shape `(B, T)` or `(B, T, D)` or `(B, T, N, H)`.
            num_blocks: int tensor. Number of query blocks.

        Returns:
            Tensor of shape `(B, U, C, ...)`.
        """
        L = self.max_past_horizon
        pad_right = self.max_future_horizon + self.chunk_size - 1
        C = self.context_size

        ndim = len(x.shape)
        if ndim == 2:
            x_padded = ops.pad(x, [(0, 0), (L, pad_right)])
        elif ndim == 3:
            x_padded = ops.pad(x, [(0, 0), (L, pad_right), (0, 0)])
        else:  # ndim == 4
            x_padded = ops.pad(x, [(0, 0), (L, pad_right), (0, 0), (0, 0)])

        # Build index matrix: offsets[u, c] = u * chunk_size + c.
        u_idx = ops.arange(num_blocks, dtype="int32")  # (U,)
        c_idx = ops.arange(C, dtype="int32")  # (C,)
        offsets = ops.expand_dims(
            u_idx, axis=1
        ) * self.chunk_size + ops.expand_dims(c_idx, axis=0)  # (U, C)

        # ops.take with 2-D indices on axis=1:
        # result shape = x_padded.shape[:1] + offsets.shape + x_padded.shape[2:]
        return ops.take(x_padded, offsets, axis=1)

    def call(self, hidden_states, mask, causal_valid_mask):
        """Chunk-based local attention forward pass.

        Args:
            hidden_states: float tensor `(B, T, D)`.
            mask: bool tensor `(B, T)`. True = valid.
            causal_valid_mask: bool tensor `(W, C)`. True = causally valid.

        Returns:
            float tensor `(B, T, N, H)`.
        """
        # Cast to float32 for numerical stability (matches HF).
        hs = ops.cast(hidden_states, "float32")

        q = ops.cast(self.q_proj(hs), "float32")
        k = ops.cast(self.k_proj(hs), "float32")
        v = ops.cast(self.v_proj(hs), "float32")

        B, T, _ = ops.shape(hidden_states)

        # Reshape to (B, T, N, H).
        q = ops.reshape(q, [B, T, self.num_heads, self.head_dim])
        k = ops.reshape(k, [B, T, self.num_heads, self.head_dim])
        v = ops.reshape(v, [B, T, self.num_heads, self.head_dim])

        # Apply per-dimension scale via softplus.
        pds = ops.reshape(
            ops.softplus(ops.cast(self.per_dim_scale, "float32")),
            (1, 1, 1, self.head_dim),
        )
        q = q * self.q_scale * pds

        k = k * self.k_scale

        # Convert to blocks.
        q_blocks = self._convert_to_block(q)  # (B, U, W, N, H)
        static_U = q_blocks.shape[1]
        if isinstance(static_U, int):
            num_blocks = static_U
        else:
            num_blocks = ops.shape(q_blocks)[1]
        k_ctx = self._extract_block_context(k, num_blocks)  # (B, U, C, N, H)
        v_ctx = self._extract_block_context(v, num_blocks)  # (B, U, C, N, H)

        # Compute logits via relative position embedding.
        logits = self.rpe(q_blocks, k_ctx)  # (B, N, U, W, C)

        # Soft-cap logits.
        logits = ops.tanh(logits / self.logit_cap) * self.logit_cap

        # Build combined valid mask from input padding and causal window.
        valid_mask = mask  # True = valid
        valid_ctx = self._extract_block_context(
            ops.cast(valid_mask, "bool"), num_blocks
        )  # (B, U, C)

        # valid_ctx → (B, 1, U, 1, C)
        cond_input = ops.expand_dims(ops.expand_dims(valid_ctx, axis=1), axis=3)
        # causal_valid_mask → (1, 1, 1, W, C)
        cond_causal = ops.expand_dims(
            ops.expand_dims(
                ops.expand_dims(ops.cast(causal_valid_mask, "bool"), axis=0),
                axis=0,
            ),
            axis=0,
        )
        combined_mask = ops.logical_and(cond_input, cond_causal)

        logits = ops.where(combined_mask, logits, self.invalid_logit_value)

        probs = ops.cast(
            ops.softmax(ops.cast(logits, "float32"), axis=-1),
            self.compute_dtype,
        )  # (B, N, U, W, C)

        # Weighted sum over context values.
        # probs: (B, N, U, W, C), v_ctx: (B, U, C, N, H)
        # → context_vectors: (B, U, W, N, H)
        b, n, u, w, c = ops.shape(probs)
        h = self.head_dim
        # Reshape for bmm: (B*N*U, W, C) @ (B*N*U, C, H)
        probs_flat = ops.reshape(
            ops.transpose(probs, (0, 2, 1, 3, 4)), [-1, w, c]
        )
        v_flat = ops.reshape(ops.transpose(v_ctx, (0, 1, 3, 2, 4)), [-1, c, h])
        ctx_flat = ops.matmul(probs_flat, v_flat)  # (B*U*N, W, H)
        ctx = ops.reshape(ctx_flat, [b, u, n, w, h])
        ctx = ops.transpose(ctx, (0, 1, 3, 2, 4))  # (B, U, W, N, H)

        # Merge blocks and truncate to original sequence length.
        ctx = ops.reshape(
            ctx, [b, u * self.chunk_size, self.num_heads, self.head_dim]
        )
        ctx = ctx[:, :T, :, :]  # (B, T, N, H)

        # Force shape to T to avoid symbolic mismatch during tracing
        ctx = ops.reshape(ctx, [b, T, self.num_heads, self.head_dim])
        return ops.cast(ctx, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "chunk_size": self.chunk_size,
                "context_left": self.max_past_horizon + 1,
                "context_right": self.max_future_horizon,
                "logit_cap": self.logit_cap,
                "invalid_logit_value": self.invalid_logit_value,
                "norm_eps": self.norm_eps,
            }
        )
        return config


class Gemma4AudioConformerAttention(keras.layers.Layer):
    """Pre/post-norm wrapper around `Gemma4AudioAttention`.

    Applies pre-norm → chunk attention → dense projection → post-norm,
    then adds the result to the residual.

    Args:
        hidden_size: int. Conformer hidden dimension.
        num_heads: int. Number of attention heads.
        chunk_size: int. Block size for chunked attention.
        context_left: int. Left attention context (inclusive).
        context_right: int. Right attention context.
        logit_cap: float. Soft-cap on attention logits.
        invalid_logit_value: float. Logit fill value for masked positions.
        gradient_clipping: float. Clip value applied before projection.
        norm_eps: float. Epsilon for parameter-free RMS norms.
        dtype: Compute dtype.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        chunk_size,
        context_left,
        context_right,
        logit_cap=50.0,
        invalid_logit_value=-1e9,
        gradient_clipping=1e10,
        norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.context_left = context_left
        self.context_right = context_right
        self.logit_cap = logit_cap
        self.invalid_logit_value = invalid_logit_value
        self.gradient_clipping = gradient_clipping
        self.norm_eps = norm_eps

        input_shape = (None, None, self.hidden_size)
        self.pre_attn_norm = Gemma4FrozenNorm(
            epsilon=self.norm_eps,
            dtype=self.dtype_policy,
            name="pre_attn_norm",
        )
        self.pre_attn_norm.build(input_shape)

        self.attn = Gemma4AudioAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            chunk_size=self.chunk_size,
            context_left=self.context_left,
            context_right=self.context_right,
            logit_cap=self.logit_cap,
            invalid_logit_value=self.invalid_logit_value,
            norm_eps=self.norm_eps,
            dtype=self.dtype_policy,
            name="attn",
        )
        self.attn.build(input_shape)

        self.out_proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=False,
            dtype=self.dtype_policy,
            name="out_proj",
        )
        self.out_proj.build(input_shape)

        self.post_norm = Gemma4FrozenNorm(
            epsilon=self.norm_eps,
            dtype=self.dtype_policy,
            name="post_norm",
        )
        self.post_norm.build(input_shape)

    def build(self, input_shape):
        self.built = True

    def call(self, x, mask, causal_valid_mask):
        """Apply conformer attention sub-block.

        Args:
            x: float tensor `(B, T, D)`.
            mask: bool tensor `(B, T)`. True = invalid.
            causal_valid_mask: bool tensor `(W, C)`.

        Returns:
            float tensor `(B, T, D)`.
        """
        residual = x
        x = ops.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.pre_attn_norm(x)

        # Attention output: (B, T, N, H)
        attn_out = self.attn(x, mask, causal_valid_mask)

        # Reshape (B, T, N, H) → (B, T, N*H).
        B, T = ops.shape(x)[0], ops.shape(x)[1]
        attn_out = ops.reshape(attn_out, [B, T, self.hidden_size])
        attn_out = ops.cast(attn_out, self.compute_dtype)

        projected = self.out_proj(attn_out)
        projected = ops.clip(
            projected, -self.gradient_clipping, self.gradient_clipping
        )
        return residual + self.post_norm(projected)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "chunk_size": self.chunk_size,
                "context_left": self.context_left,
                "context_right": self.context_right,
                "logit_cap": self.logit_cap,
                "invalid_logit_value": self.invalid_logit_value,
                "gradient_clipping": self.gradient_clipping,
                "norm_eps": self.norm_eps,
            }
        )
        return config


class Gemma4AudioSSCPConvBlock(keras.layers.Layer):
    """Single 2-D convolution block for the SubSample Convolution Projection.

    Applies manual semicausal padding, a `data_format="channels_last"` Conv2D,
    LayerNorm over the channel axis, and ReLU. Maintains and downsamples the
    time-dimension validity mask along with the tensor.

    Args:
        in_channels: int. Number of input channels.
        out_channels: int. Number of output channels.
        kernel_t: int. Kernel size along the time dimension.
        kernel_f: int. Kernel size along the frequency dimension.
        stride_t: int. Stride along the time dimension.
        stride_f: int. Stride along the frequency dimension.
        pad_t_top: int. Padding rows added at the top (start) of time.
        pad_t_bottom: int. Padding rows added at the bottom (end) of time.
        norm_eps: float. Epsilon for LayerNorm.
        dtype: Compute dtype.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_t,
        kernel_f,
        stride_t,
        stride_f,
        pad_t_top,
        pad_t_bottom,
        norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_t = kernel_t
        self.kernel_f = kernel_f
        self.stride_t = stride_t
        self.stride_f = stride_f
        self.pad_t_top = pad_t_top
        self.pad_t_bottom = pad_t_bottom
        self.norm_eps = norm_eps
        # Frequency padding is always symmetric (1, 1).
        self.pad_f_left = 1
        self.pad_f_right = 1

        self.conv = keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=(self.kernel_t, self.kernel_f),
            strides=(self.stride_t, self.stride_f),
            padding="valid",
            use_bias=False,
            data_format="channels_last",
            dtype=self.dtype_policy,
            name="conv",
        )
        dummy_t = 3
        dummy_f = 5
        self.conv.build((None, dummy_t, dummy_f, self.in_channels))

        self.norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=self.norm_eps,
            center=False,
            dtype=self.dtype_policy,
            name="norm",
        )
        self.norm.build((None, None, None, self.out_channels))

    def build(self, input_shape):
        self.built = True

    def call(self, x, mask):
        """Apply one SSCP conv block.

        Args:
            x: float tensor `(B, T, F, C_in)`.
            mask: bool tensor `(B, T)`. True = valid.

        Returns:
            Tuple (output, new_mask) of shapes `(B, T_out, F_out, C_out)` and
            `(B, T_out)`.
        """
        # Zero out invalid time positions before convolution.
        mask_3d = ops.cast(
            ops.expand_dims(ops.expand_dims(mask, axis=-1), axis=-1), x.dtype
        )  # (B, T, 1, 1)
        x = ops.where(mask_3d, x, ops.zeros_like(x))

        # Manual padding: (B,T,F,C) → (B, T+top+bot, F+f_left+f_right, C).
        x = ops.pad(
            x,
            [
                (0, 0),
                (self.pad_t_top, self.pad_t_bottom),
                (self.pad_f_left, self.pad_f_right),
                (0, 0),
            ],
        )

        x = ops.cast(x, self.compute_dtype)
        x = self.conv(x)  # (B, T_out, F_out, C_out)

        # Downsample mask along time by the conv stride and trim length.
        t_out = ops.shape(x)[1]
        new_mask = mask[:, :: self.stride_t][:, :t_out]

        # Force shape to avoid symbolic mismatch during tracing
        B = ops.shape(x)[0]
        new_mask = ops.reshape(new_mask, (B, t_out))

        x = self.norm(x)  # LayerNorm over channels
        x = ops.relu(x)
        return x, new_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "kernel_t": self.kernel_t,
                "kernel_f": self.kernel_f,
                "stride_t": self.stride_t,
                "stride_f": self.stride_f,
                "pad_t_top": self.pad_t_top,
                "pad_t_bottom": self.pad_t_bottom,
                "norm_eps": self.norm_eps,
            }
        )
        return config


class Gemma4AudioSubSampleConvProjection(keras.layers.Layer):
    """Two-layer 2-D conv that subsamples the mel spectrogram by 4x in time.

    After two Conv2D blocks (each with stride 2x2), the frequency and channel
    dimensions are flattened and projected to `hidden_size` via a linear layer.

    Args:
        input_feat_size: int. Number of mel bins (frequency channels).
        hidden_size: int. Output (and conformer) hidden dimension.
        conv_channels: tuple of ints `(C0, C1)`. Output channel counts for
            the two conv blocks.
        kernel_sizes: tuple of `(kT, kF)` pairs. Kernel sizes per block.
        stride_sizes: tuple of `(sT, sF)` pairs. Stride sizes per block.
        pad_t_top: int. Time-dimension top padding per block.
        pad_t_bottom: int. Time-dimension bottom padding per block.
        norm_eps: float. LayerNorm epsilon.
        dtype: Compute dtype.
    """

    def __init__(
        self,
        input_feat_size,
        hidden_size,
        conv_channels,
        kernel_sizes,
        stride_sizes,
        pad_t_top,
        pad_t_bottom,
        norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.input_feat_size = input_feat_size
        self.hidden_size = hidden_size
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.stride_sizes = stride_sizes
        self.pad_t_top = pad_t_top
        self.pad_t_bottom = pad_t_bottom
        self.norm_eps = norm_eps

        # Compute output frequency dimension after each conv block.
        f = input_feat_size
        freq_dims = []
        for i in range(2):
            kf = kernel_sizes[i][1]
            sf = stride_sizes[i][1]
            f = (f + 2 - kf) // sf + 1  # pad_f=(1,1) always
            freq_dims.append(f)
        self.freq_dims = freq_dims
        # Input to the final linear layer.
        self.input_proj_in = conv_channels[1] * freq_dims[1]

        kt0, kf0 = self.kernel_sizes[0]
        st0, sf0 = self.stride_sizes[0]
        self.conv_0 = Gemma4AudioSSCPConvBlock(
            in_channels=1,
            out_channels=self.conv_channels[0],
            kernel_t=kt0,
            kernel_f=kf0,
            stride_t=st0,
            stride_f=sf0,
            pad_t_top=self.pad_t_top,
            pad_t_bottom=self.pad_t_bottom,
            norm_eps=self.norm_eps,
            dtype=self.dtype_policy,
            name="conv_0",
        )
        self.conv_0.build((None, None, self.input_feat_size, 1))

        kt1, kf1 = self.kernel_sizes[1]
        st1, sf1 = self.stride_sizes[1]
        self.conv_1 = Gemma4AudioSSCPConvBlock(
            in_channels=self.conv_channels[0],
            out_channels=self.conv_channels[1],
            kernel_t=kt1,
            kernel_f=kf1,
            stride_t=st1,
            stride_f=sf1,
            pad_t_top=self.pad_t_top,
            pad_t_bottom=self.pad_t_bottom,
            norm_eps=self.norm_eps,
            dtype=self.dtype_policy,
            name="conv_1",
        )
        self.conv_1.build(
            (None, None, self.freq_dims[0], self.conv_channels[0])
        )

        self.input_proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=False,
            dtype=self.dtype_policy,
            name="input_proj",
        )
        self.input_proj.build((None, None, self.input_proj_in))

    def build(self, input_shape):
        self.built = True

    def call(self, audio_mel, mask):
        """Subsample and project a mel spectrogram.

        Args:
            audio_mel: float tensor `(B, T, mel_bins)`.
            mask: bool tensor `(B, T)`. True = padded/invalid.

        Returns:
            Tuple `(output, new_mask)` with shapes `(B, T//4, hidden_size)`
            and `(B, T//4)`.
        """
        # Expand channel dim: (B, T, F) → (B, T, F, 1).
        x = ops.expand_dims(audio_mel, axis=-1)
        x = ops.cast(x, self.compute_dtype)

        x, mask = self.conv_0(x, mask)  # (B, T//2, F//2, C0)
        x, mask = self.conv_1(x, mask)  # (B, T//4, F//4, C1)

        B = ops.shape(x)[0]
        T_out = ops.shape(x)[1]
        # Flatten F and C: (B, T//4, F//4 * C1).
        x = ops.reshape(
            x, [B, T_out, self.freq_dims[1] * self.conv_channels[1]]
        )
        x = self.input_proj(x)  # (B, T//4, hidden_size)
        return x, mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_feat_size": self.input_feat_size,
                "hidden_size": self.hidden_size,
                "conv_channels": list(self.conv_channels),
                "kernel_sizes": [list(k) for k in self.kernel_sizes],
                "stride_sizes": [list(s) for s in self.stride_sizes],
                "pad_t_top": self.pad_t_top,
                "pad_t_bottom": self.pad_t_bottom,
                "norm_eps": self.norm_eps,
            }
        )
        return config


class Gemma4AudioConformerFeedForward(keras.layers.Layer):
    """Macaron-style feed-forward sub-block for the audio conformer.

    Applies pre-norm → FFW (SiLU activation) → post-norm, then adds
    `residual_weight * result` to the residual.

    Args:
        hidden_size: int. Input/output dimension.
        intermediate_dim: int. Hidden dimension inside the FFW (4× hidden).
        residual_weight: float. Weight for the residual connection (0.5).
        gradient_clipping: float. Clip value for intermediate activations.
        norm_eps: float. Epsilon for parameter-free RMS norms.
        dtype: Compute dtype.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_dim=None,
        residual_weight=0.5,
        gradient_clipping=1e10,
        norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_dim = intermediate_dim or hidden_size * 4
        self.residual_weight = residual_weight
        self.gradient_clipping = gradient_clipping
        self.norm_eps = norm_eps

        input_shape = (None, None, self.hidden_size)
        self.pre_norm = Gemma4FrozenNorm(
            epsilon=self.norm_eps,
            dtype=self.dtype_policy,
            name="pre_norm",
        )
        self.pre_norm.build(input_shape)

        self.ffw_1 = keras.layers.Dense(
            self.intermediate_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="ffw_1",
        )
        self.ffw_1.build(input_shape)

        self.ffw_2 = keras.layers.Dense(
            self.hidden_size,
            use_bias=False,
            dtype=self.dtype_policy,
            name="ffw_2",
        )
        self.ffw_2.build(input_shape[:-1] + (self.intermediate_dim,))

        self.post_norm = Gemma4FrozenNorm(
            epsilon=self.norm_eps,
            dtype=self.dtype_policy,
            name="post_norm",
        )
        self.post_norm.build(input_shape)

    def build(self, input_shape):
        self.built = True

    def call(self, x):
        """Apply the macaron FFW sub-block.

        Args:
            x: float tensor `(B, T, D)`.

        Returns:
            float tensor `(B, T, D)`.
        """
        residual = x
        x = ops.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.pre_norm(x)
        x = self.ffw_1(x)
        x = ops.silu(x)
        x = self.ffw_2(x)
        x = ops.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.post_norm(x)
        return residual + self.residual_weight * x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_dim": self.intermediate_dim,
                "residual_weight": self.residual_weight,
                "gradient_clipping": self.gradient_clipping,
                "norm_eps": self.norm_eps,
            }
        )
        return config


class Gemma4AudioConformerLightConv1d(keras.layers.Layer):
    """Lightweight causal depthwise Conv1D sub-block for the audio conformer.

    Applies pre-norm → linear_start (→ GLU) → causal depthwise Conv1D →
    post-norm → SiLU → linear_end → residual add.

    Args:
        hidden_size: int. Input/output dimension.
        kernel_size: int. Depthwise conv kernel size.
        gradient_clipping: float. Clip value applied after depthwise conv.
        norm_eps: float. Epsilon for parameter-free RMS norms.
        dtype: Compute dtype.
    """

    def __init__(
        self,
        hidden_size,
        kernel_size=5,
        gradient_clipping=1e10,
        norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.gradient_clipping = gradient_clipping
        self.norm_eps = norm_eps

        input_shape = (None, None, self.hidden_size)
        self.pre_norm = Gemma4FrozenNorm(
            epsilon=self.norm_eps,
            dtype=self.dtype_policy,
            name="pre_norm",
        )
        self.pre_norm.build(input_shape)

        self.linear_start = keras.layers.Dense(
            self.hidden_size * 2,
            use_bias=False,
            dtype=self.dtype_policy,
            name="linear_start",
        )
        self.linear_start.build(input_shape)

        # Causal depthwise conv: (B, T, D) → (B, T, D).
        self.depthwise_conv = keras.layers.DepthwiseConv1D(
            kernel_size=self.kernel_size,
            padding="valid",
            use_bias=False,
            data_format="channels_last",
            dtype=self.dtype_policy,
            name="depthwise_conv",
        )
        self.depthwise_conv.build(input_shape[:-1] + (self.hidden_size,))

        self.conv_norm = Gemma4FrozenNorm(
            epsilon=self.norm_eps,
            dtype=self.dtype_policy,
            name="conv_norm",
        )
        self.conv_norm.build(input_shape[:-1] + (self.hidden_size,))

        self.linear_end = keras.layers.Dense(
            self.hidden_size,
            use_bias=False,
            dtype=self.dtype_policy,
            name="linear_end",
        )
        self.linear_end.build(input_shape[:-1] + (self.hidden_size,))

    def build(self, input_shape):
        self.built = True

    def call(self, x):
        """Apply the lightweight Conv1D sub-block.

        Args:
            x: float tensor `(B, T, D)` (padded positions pre-zeroed by
               the conformer block).

        Returns:
            float tensor `(B, T, D)`.
        """
        residual = x
        x = self.pre_norm(x)
        x = self.linear_start(x)  # (B, T, 2D)

        # GLU: split in half and gate.
        x1, x2 = ops.split(x, 2, axis=-1)
        x = x1 * ops.sigmoid(x2)  # (B, T, D)

        # Causal padding: pad (kernel_size - 1) zeros on the left of time axis.
        x = ops.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        x = self.depthwise_conv(x)  # (B, T, D)
        x = ops.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.conv_norm(x)
        x = ops.silu(x)
        x = self.linear_end(x)
        return x + residual

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "kernel_size": self.kernel_size,
                "gradient_clipping": self.gradient_clipping,
                "norm_eps": self.norm_eps,
            }
        )
        return config


class Gemma4AudioConformerBlock(keras.layers.Layer):
    """Full Conformer block: FFW → attention → lconv → FFW → norm.

    Args:
        hidden_size: int. Input/output (conformer) hidden dimension.
        num_heads: int. Number of attention heads.
        chunk_size: int. Block size for chunked attention.
        context_left: int. Left attention context (inclusive).
        context_right: int. Right attention context.
        logit_cap: float. Attention logit soft-cap.
        invalid_logit_value: float. Fill for masked logits.
        conv_kernel_size: int. Depthwise conv kernel size in lconv1d.
        residual_weight: float. Weight for the macaron FFW residual (0.5).
        gradient_clipping: float. Gradient clip value.
        norm_eps: float. Epsilon for parameter-free RMS norms.
        dtype: Compute dtype.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        chunk_size,
        context_left,
        context_right,
        logit_cap=50.0,
        invalid_logit_value=-1e9,
        conv_kernel_size=5,
        residual_weight=0.5,
        gradient_clipping=1e10,
        norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.context_left = context_left
        self.context_right = context_right
        self.logit_cap = logit_cap
        self.invalid_logit_value = invalid_logit_value
        self.conv_kernel_size = conv_kernel_size
        self.residual_weight = residual_weight
        self.gradient_clipping = gradient_clipping
        self.norm_eps = norm_eps

        input_shape = (None, None, self.hidden_size)
        self.ffw_start = Gemma4AudioConformerFeedForward(
            hidden_size=self.hidden_size,
            residual_weight=self.residual_weight,
            gradient_clipping=self.gradient_clipping,
            norm_eps=self.norm_eps,
            dtype=self.dtype_policy,
            name="ffw_start",
        )
        self.ffw_start.build(input_shape)

        self.attention = Gemma4AudioConformerAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            chunk_size=self.chunk_size,
            context_left=self.context_left,
            context_right=self.context_right,
            logit_cap=self.logit_cap,
            invalid_logit_value=self.invalid_logit_value,
            gradient_clipping=self.gradient_clipping,
            norm_eps=self.norm_eps,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.attention.build(input_shape)

        self.lconv = Gemma4AudioConformerLightConv1d(
            hidden_size=self.hidden_size,
            kernel_size=self.conv_kernel_size,
            gradient_clipping=self.gradient_clipping,
            norm_eps=self.norm_eps,
            dtype=self.dtype_policy,
            name="lconv",
        )
        self.lconv.build(input_shape)

        self.ffw_end = Gemma4AudioConformerFeedForward(
            hidden_size=self.hidden_size,
            residual_weight=self.residual_weight,
            gradient_clipping=self.gradient_clipping,
            norm_eps=self.norm_eps,
            dtype=self.dtype_policy,
            name="ffw_end",
        )
        self.ffw_end.build(input_shape)

        # Final block norm (frozen RMS norm — has non-trainable scale).
        self.norm = Gemma4FrozenNorm(
            epsilon=self.norm_eps,
            dtype=self.dtype_policy,
            name="norm",
        )
        self.norm.build(input_shape)

    def build(self, input_shape):
        self.built = True

    def call(self, x, mask, causal_valid_mask):
        """Apply one Conformer block.

        Args:
            x: float tensor `(B, T, D)`.
            mask: bool tensor `(B, T)`. True = valid.
            causal_valid_mask: bool tensor `(W, C)`.

        Returns:
            float tensor `(B, T, D)`.
        """
        x = self.ffw_start(x)
        x = self.attention(x, mask, causal_valid_mask)

        # Zero out padded positions before the depthwise conv.
        valid = ops.cast(ops.expand_dims(mask, axis=-1), x.dtype)
        x_for_lconv = x * valid
        x = self.lconv(x_for_lconv)

        x = self.ffw_end(x)
        x = ops.clip(x, -self.gradient_clipping, self.gradient_clipping)
        return self.norm(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "chunk_size": self.chunk_size,
                "context_left": self.context_left,
                "context_right": self.context_right,
                "logit_cap": self.logit_cap,
                "invalid_logit_value": self.invalid_logit_value,
                "conv_kernel_size": self.conv_kernel_size,
                "residual_weight": self.residual_weight,
                "gradient_clipping": self.gradient_clipping,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras_hub_export("keras_hub.models.Gemma4AudioEncoder")
class Gemma4AudioEncoder(keras.Model):
    """Audio encoder for Gemma4 based on the Universal Speech Model (USM).

    Encodes mel spectrograms into audio token embeddings projected into the
    language model's hidden space.  The pipeline is:

    1. **SubSampleConvProjection**: two stacked Conv2D blocks that downsample
       time by 4× at 16ms hop rate, then a linear projection to `hidden_size`.
    2. **Conformer blocks** (`num_layers` of them): macaron-FFW → chunk
       attention with relative position bias → causal depthwise Conv1D →
       macaron-FFW → RMS norm.
    3. **Temporal striding** (if `reduction_factor > 1`): reduce sequence by
       taking every `reduction_factor`-th token.
    4. **Output projection**: linear `hidden_size → output_proj_dims` followed
       by another linear `output_proj_dims → output_dim` (= text hidden size)
       and a parameter-free RMS norm.

    Padded positions (indicated by `audio_mel_mask`) are zeroed out in the
    final output.

    Args:
        input_feat_size: int. Number of mel filterbank channels. Defaults to
            `128`.
        hidden_size: int. Conformer hidden dimension. Defaults to `1024`.
        num_heads: int. Number of conformer attention heads. Defaults to `8`.
        num_layers: int. Number of Conformer blocks. Defaults to `12`.
        chunk_size: int. Block size for chunk-based attention. Defaults to
            `12`.
        context_left: int. Left attention context (inclusive). Defaults to
            `13`.
        context_right: int. Right attention context. Defaults to `0`.
        logit_cap: float. Soft-cap on attention logits. Defaults to `50.0`.
        invalid_logit_value: float. Fill for masked logits. Defaults to
            `-1e9`.
        conv_kernel_size: int. Depthwise conv kernel size. Defaults to `5`.
        reduction_factor: int. Temporal stride after the conformer stack.
            Defaults to `1`.
        residual_weight: float. Macaron FFW residual weight. Defaults to
            `0.5`.
        gradient_clipping: float. Clip value. Defaults to `1e10`.
        sscp_conv_channels: tuple of two ints. Output channels per SSCP conv.
            Defaults to `(128, 32)`.
        sscp_kernel_sizes: tuple of two `(kT, kF)` pairs. Defaults to
            `((3, 3), (3, 3))`.
        sscp_stride_sizes: tuple of two `(sT, sF)` pairs. Defaults to
            `((2, 2), (2, 2))`.
        output_proj_dims: int or `None`. Intermediate audio projection
            dimension (e.g. 1536). `None` skips this projection.
        output_dim: int. Final output dimension = text backbone hidden size.
        norm_eps: float. Epsilon for conformer RMS norms. Defaults to `1e-6`.
        sscp_norm_eps: float. Epsilon for SSCP LayerNorm. Defaults to `1e-6`.
        dtype: Compute dtype. Defaults to `None`.
    """

    def __init__(
        self,
        input_feat_size=128,
        hidden_size=1024,
        num_heads=8,
        num_layers=12,
        chunk_size=12,
        context_left=13,
        context_right=0,
        logit_cap=50.0,
        invalid_logit_value=-1e9,
        conv_kernel_size=5,
        reduction_factor=1,
        residual_weight=0.5,
        gradient_clipping=1e10,
        sscp_conv_channels=(128, 32),
        sscp_kernel_sizes=((3, 3), (3, 3)),
        sscp_stride_sizes=((2, 2), (2, 2)),
        output_proj_dims=1536,
        output_dim=2048,
        norm_eps=1e-6,
        sscp_norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        # === Store config ===
        self.input_feat_size = input_feat_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.context_left = context_left
        self.context_right = context_right
        self.logit_cap = logit_cap
        self.invalid_logit_value = invalid_logit_value
        self.conv_kernel_size = conv_kernel_size
        self.reduction_factor = reduction_factor
        self.residual_weight = residual_weight
        self.gradient_clipping = gradient_clipping
        self.sscp_conv_channels = tuple(sscp_conv_channels)
        self.sscp_kernel_sizes = tuple(tuple(k) for k in sscp_kernel_sizes)
        self.sscp_stride_sizes = tuple(tuple(s) for s in sscp_stride_sizes)
        self.output_proj_dims = output_proj_dims
        self.output_dim = output_dim
        self.norm_eps = norm_eps
        self.sscp_norm_eps = sscp_norm_eps

        # Semicausal (non-streaming) padding for SSCP conv blocks.
        pad_t_top = self.sscp_kernel_sizes[0][0] // 2  # = 1 for 3x3
        pad_t_bottom = self.sscp_kernel_sizes[0][0] // 2  # = 1 (non-streaming)

        # === Layers ===
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(
            input_feat_size=input_feat_size,
            hidden_size=hidden_size,
            conv_channels=self.sscp_conv_channels,
            kernel_sizes=self.sscp_kernel_sizes,
            stride_sizes=self.sscp_stride_sizes,
            pad_t_top=pad_t_top,
            pad_t_bottom=pad_t_bottom,
            norm_eps=sscp_norm_eps,
            dtype=dtype,
            name="subsample_conv_projection",
        )

        self.conformer_blocks = [
            Gemma4AudioConformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                chunk_size=chunk_size,
                context_left=context_left,
                context_right=context_right,
                logit_cap=logit_cap,
                invalid_logit_value=invalid_logit_value,
                conv_kernel_size=conv_kernel_size,
                residual_weight=residual_weight,
                gradient_clipping=gradient_clipping,
                norm_eps=norm_eps,
                dtype=dtype,
                name=f"conformer_{i}",
            )
            for i in range(num_layers)
        ]

        # Optional intermediate projection: hidden_size → output_proj_dims.
        if output_proj_dims is not None:
            self.output_proj = keras.layers.Dense(
                output_proj_dims,
                use_bias=True,
                dtype=dtype,
                name="output_proj",
            )
        else:
            self.output_proj = None

        # Final projection into text embedding space + parameter-free
        # RMS norm (matches HF Gemma4MultimodalEmbedder with_scale=False).
        self.audio_output_projection = keras.layers.Dense(
            output_dim,
            use_bias=False,
            dtype=dtype,
            name="audio_output_projection",
        )
        self.output_norm = Gemma4VNorm(
            epsilon=norm_eps,
            dtype=dtype,
            name="output_norm",
        )

        # Precompute the causal valid mask (numpy, shape [W, C]).
        max_past = max(0, context_left - 1)
        max_future = context_right
        C = chunk_size + max_past + max_future
        upper_diagonal = max_past + max_future
        lower_causal = np.tril(np.ones((C, chunk_size), dtype=bool), k=0).T
        upper_causal = np.tril(
            np.ones((chunk_size, C), dtype=bool), k=upper_diagonal
        )
        self._causal_valid_mask_np = lower_causal & upper_causal  # (W, C)

    def call(self, audio_mel, audio_mel_mask):
        """Encode a batch of mel spectrograms.

        Args:
            audio_mel: float tensor `(B, N_clips, T, mel_bins)` or
                `(B, T, mel_bins)`. When 4-D the leading two dimensions are
                collapsed so that each clip is treated as an independent item
                in the batch (mirrors how `Gemma4VisionEncoder` handles
                `(B, N_images, H, W, C)`).
            audio_mel_mask: bool/int tensor matching the time axis of
                `audio_mel` — either `(B, N_clips, T)` or `(B, T)`.
                True = padded/invalid.

        Returns:
            float tensor `(B_out, T_out, output_dim)` where
            `B_out = B * N_clips` (or `B` for 3-D input).
            Padded positions are set to zero.
        """
        # Collapse (B, N_clips, T, F) → (B*N_clips, T, F).
        # This is done at runtime (inside call) so that dynamic shapes are
        # resolved correctly, exactly like Gemma4VisionEncoder does for
        # (B, N_images, H, W, C) → (B*N_images, H, W, C).
        is_4d = len(audio_mel.shape) == 4
        if is_4d:
            if audio_mel.shape[1] == 0:
                return ops.zeros((0, 0, 0, self.output_dim)), ops.cast(
                    ops.zeros((0, 0)), "bool"
                )
            s_input = ops.shape(audio_mel)
            audio_mel = ops.reshape(
                audio_mel, [s_input[0] * s_input[1], s_input[2], s_input[3]]
            )
            s_mask_input = ops.shape(audio_mel_mask)
            audio_mel_mask = ops.reshape(
                audio_mel_mask,
                [s_mask_input[0] * s_mask_input[1], s_mask_input[2]],
            )

        # 1. Subsample: (B, T, F) → (B, T//4, hidden_size).
        x, mask = self.subsample_conv_projection(audio_mel, audio_mel_mask)

        # 2. Convert causal mask to a Keras tensor (done once per forward).
        causal_valid_mask = ops.cast(
            ops.convert_to_tensor(self._causal_valid_mask_np), "bool"
        )

        # 3. Conformer blocks.
        for block in self.conformer_blocks:
            x = block(x, mask, causal_valid_mask)

        # 4. Optional temporal stride.
        if self.reduction_factor > 1:
            x = x[:, :: self.reduction_factor, :]
            mask = mask[:, :: self.reduction_factor]

        # 5. Intermediate projection: hidden_size → output_proj_dims.
        if self.output_proj is not None:
            x = self.output_proj(x)

        # Ensure mask length matches x length (they should match after the
        # same stride, but guard against any off-by-one rounding).
        static_T = x.shape[1]
        if static_T is not None:
            t_out = static_T
        else:
            t_out = ops.shape(x)[1]
        mask = mask[:, :t_out]

        # 6. Final projection into text space: norm THEN linear (matches HF
        #    Gemma4MultimodalEmbedder: norm → embedding_projection).
        x = self.output_norm(x)
        x = self.audio_output_projection(x)

        # 7. No zero-out of padded positions: let silence activations flow.

        # 8. Un-flatten if original input was 4-D
        if is_4d:
            s_out = ops.shape(x)
            x = ops.reshape(x, [s_input[0], s_input[1], s_out[1], s_out[2]])

        return x

    def compute_output_shape(self, audio_mel_shape):
        """Return output shape without tracing through call().

        This prevents Keras from symbolically tracing through the Conformer's
        call() (which uses `int(ops.shape(x)[1])`) when building the parent
        Functional backbone graph — exactly like Gemma4VisionEncoder does.

        Args:
            audio_mel_shape: Shape of the first input (audio_mel).
                Either `(B, N_clips, T, F)` or `(B, T, F)`.

        Returns:
            Output shape mapping to inputs.
        """
        if isinstance(audio_mel_shape, dict):
            pass  # not expecting dict

        if len(audio_mel_shape) == 4:
            return (
                audio_mel_shape[0],
                audio_mel_shape[1],
                None,
                self.output_dim,
            )
        return (audio_mel_shape[0], None, self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_feat_size": self.input_feat_size,
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "chunk_size": self.chunk_size,
                "context_left": self.context_left,
                "context_right": self.context_right,
                "logit_cap": self.logit_cap,
                "invalid_logit_value": self.invalid_logit_value,
                "conv_kernel_size": self.conv_kernel_size,
                "reduction_factor": self.reduction_factor,
                "residual_weight": self.residual_weight,
                "gradient_clipping": self.gradient_clipping,
                "sscp_conv_channels": list(self.sscp_conv_channels),
                "sscp_kernel_sizes": [list(k) for k in self.sscp_kernel_sizes],
                "sscp_stride_sizes": [list(s) for s in self.sscp_stride_sizes],
                "output_proj_dims": self.output_proj_dims,
                "output_dim": self.output_dim,
                "norm_eps": self.norm_eps,
                "sscp_norm_eps": self.sscp_norm_eps,
            }
        )
        return config
