import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_attention import (
    MuseGlimmerTextAttention,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerCenteredRMSNorm,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerRMSNorm,
)
from keras_hub.src.utils.keras_utils import clone_initializer


class MuseGlimmerTextDecoder(keras.layers.Layer):
    """MuseGlimmer transformer decoder layer.

    A "sandwich norm" block: `input_layernorm` and `pre_feedforward_norm`
    use `rms_norm_eps`; `post_attention_norm` and `post_feedforward_norm`
    use the separate (tighter) `post_norm_eps`. All four are the
    "centered" `(1 + weight)` RMSNorm variant. See
    `modeling_muse_glimmer.py`, `MuseGlimmerTextDecoderLayer.forward`.

    The assistant/drafter configuration (see
    `modeling_muse_glimmer_assistant.py`) differs in three ways, each
    controlled by an opt-in flag that defaults to the main model's current
    behavior:
    - `use_bidirectional_attention=True` replaces the causal mask with a
      mask that only accounts for padding (`MuseGlimmerAssistantAttention`
      hardcodes `is_causal = False`).
    - `use_sandwich_norm=False` skips `post_attention_layernorm`/
      `post_feedforward_layernorm` entirely — the assistant's decoder
      layer is a plain two-norm pre-norm block, not a sandwich.
    - `enable_qk_scale_and_gate=False` is threaded into
      `MuseGlimmerTextAttention` (see that layer's docstring).
    - `use_centered_norm=False` builds `input_layernorm`/
      `pre_feedforward_layernorm` as plain `weight * normalized(x)` RMSNorm
      (`MuseGlimmerAssistantRMSNorm`) instead of the main model's
      `(1 + weight) * normalized(x)` centered variant.

    `context_hidden_states`, when passed to `call()`, is threaded into
    the self-attention layer unchanged (assistant/drafter only).

    Args:
        intermediate_dim: int. SwiGLU MLP intermediate dimension.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value attention heads.
        head_dim: int. Per-head dimension.
        hidden_activation: str. MLP gate activation. Defaults to `"silu"`.
        rms_norm_eps: float. Epsilon for the pre-sublayer norms and QK-norm.
        post_norm_eps: float. Epsilon for the post-sublayer norms.
        qk_scale_factor: float. Extra multiplier on Q after QK-norm.
        use_rope: bool. Whether this layer applies RoPE to Q/K.
        rope_max_wavelength: float. RoPE base theta.
        sliding_window_size: int or None. Sliding window size, or `None`
            for full attention.
        use_bidirectional_attention: bool. If `True`, replaces the causal
            self-attention mask with a padding-only mask. Defaults to
            `False`.
        enable_qk_scale_and_gate: bool. Passed through to
            `MuseGlimmerTextAttention`. Defaults to `True`.
        qk_norm_with_scale: bool. Passed through to
            `MuseGlimmerTextAttention`. Defaults to `False`.
        use_sandwich_norm: bool. If `False`, skips
            `post_attention_layernorm`/`post_feedforward_layernorm`.
            Defaults to `True`.
        use_centered_norm: bool. If `False`, builds `input_layernorm`/
            `pre_feedforward_layernorm` as plain RMSNorm instead of the
            centered `(1 + weight)` variant. Defaults to `True`.
        kernel_initializer: initializer for the dense projections.
        dropout: float. Dropout rate.
    """

    def __init__(
        self,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        hidden_activation="silu",
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        qk_scale_factor=1.0,
        use_rope=True,
        rope_max_wavelength=500000.0,
        sliding_window_size=None,
        use_bidirectional_attention=False,
        enable_qk_scale_and_gate=True,
        qk_norm_with_scale=False,
        use_sandwich_norm=True,
        use_centered_norm=True,
        kernel_initializer="glorot_uniform",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.rms_norm_eps = rms_norm_eps
        self.post_norm_eps = post_norm_eps
        self.qk_scale_factor = qk_scale_factor
        self.use_rope = use_rope
        self.rope_max_wavelength = rope_max_wavelength
        self.sliding_window_size = sliding_window_size
        self.use_bidirectional_attention = use_bidirectional_attention
        self.enable_qk_scale_and_gate = enable_qk_scale_and_gate
        self.qk_norm_with_scale = qk_norm_with_scale
        self.use_sandwich_norm = use_sandwich_norm
        self.use_centered_norm = use_centered_norm
        self.dropout = dropout
        self.activation = keras.activations.get(hidden_activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.supports_masking = True

    def build(self, decoder_sequence_shape):
        self.hidden_dim = decoder_sequence_shape[-1]

        if self.use_centered_norm:
            self._input_layernorm = MuseGlimmerCenteredRMSNorm(
                eps=self.rms_norm_eps,
                dtype=self.dtype_policy,
                name="input_layernorm",
            )
        else:
            self._input_layernorm = MuseGlimmerRMSNorm(
                eps=self.rms_norm_eps,
                dtype=self.dtype_policy,
                name="input_layernorm",
            )
        self._input_layernorm.build(decoder_sequence_shape)

        self._self_attention_layer = MuseGlimmerTextAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            qk_scale_factor=self.qk_scale_factor,
            rms_norm_eps=self.rms_norm_eps,
            use_rope=self.use_rope,
            rope_max_wavelength=self.rope_max_wavelength,
            sliding_window_size=self.sliding_window_size,
            enable_qk_scale_and_gate=self.enable_qk_scale_and_gate,
            qk_norm_with_scale=self.qk_norm_with_scale,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self._self_attention_layer.build(decoder_sequence_shape)

        if self.use_sandwich_norm:
            self._post_attention_layernorm = MuseGlimmerCenteredRMSNorm(
                eps=self.post_norm_eps,
                dtype=self.dtype_policy,
                name="post_attention_layernorm",
            )
            self._post_attention_layernorm.build(decoder_sequence_shape)

        if self.use_centered_norm:
            self._pre_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(
                eps=self.rms_norm_eps,
                dtype=self.dtype_policy,
                name="pre_feedforward_layernorm",
            )
        else:
            self._pre_feedforward_layernorm = MuseGlimmerRMSNorm(
                eps=self.rms_norm_eps,
                dtype=self.dtype_policy,
                name="pre_feedforward_layernorm",
            )
        self._pre_feedforward_layernorm.build(decoder_sequence_shape)

        self._feedforward_gate_dense = keras.layers.Dense(
            self.intermediate_dim,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="feedforward_gate_dense",
        )
        self._feedforward_gate_dense.build(decoder_sequence_shape)

        self._feedforward_up_dense = keras.layers.Dense(
            self.intermediate_dim,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="feedforward_up_dense",
        )
        self._feedforward_up_dense.build(decoder_sequence_shape)

        self._feedforward_down_dense = keras.layers.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="feedforward_down_dense",
        )
        self._feedforward_down_dense.build(
            self._feedforward_gate_dense.compute_output_shape(
                decoder_sequence_shape
            )
        )

        if self.use_sandwich_norm:
            self._post_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(
                eps=self.post_norm_eps,
                dtype=self.dtype_policy,
                name="post_feedforward_layernorm",
            )
            self._post_feedforward_layernorm.build(decoder_sequence_shape)

        self._dropout_layer = keras.layers.Dropout(
            rate=self.dropout, dtype=self.dtype_policy
        )
        self.built = True

    def _compute_self_attention_mask(
        self,
        decoder_sequence,
        context_hidden_states,
        decoder_padding_mask,
        decoder_attention_mask,
        self_attention_cache,
        self_attention_cache_update_index,
    ):
        if self.use_bidirectional_attention:
            # Assistant/drafter configuration: queries attend
            # bi-directionally to each other and to the (unmasked) context
            # key/value positions — no causal component at all. See
            # `MuseGlimmerAssistantModel.forward`'s
            # `create_bidirectional_mask` usage.
            batch_size = ops.shape(decoder_sequence)[0]
            q_len = ops.shape(decoder_sequence)[1]

            if context_hidden_states is not None and (
                self_attention_cache is not None
            ):
                # DFlash context caching (see `MuseGlimmerTextAttention`'s
                # docstring): keys span the persisted context-cache buffer
                # (one real target position per cycle, growing over the
                # whole generation) followed by this cycle's fresh block.
                # Built directly over absolute positions rather than
                # `compute_causal_mask`/`_mask_sliding_window`, which both
                # assume a single contiguous, non-persisted key range.
                max_length = ops.shape(self_attention_cache)[2]
                write_start = ops.cast(
                    0
                    if self_attention_cache_update_index is None
                    else self_attention_cache_update_index,
                    "int32",
                )
                # Matches `MuseGlimmerTextAttention.call()`'s RoPE offset
                # (`start_index + context_len`): the number of context
                # positions written THIS call is `context_hidden_states`'s
                # own length, not always one — the checkpoint-conversion
                # script seeds the whole context in a single call.
                context_write_len = ops.shape(context_hidden_states)[1]
                valid_context_len = write_start + context_write_len
                context_positions = ops.arange(max_length, dtype="int32")
                block_positions = valid_context_len + ops.arange(
                    q_len, dtype="int32"
                )
                key_positions = ops.concatenate(
                    [context_positions, block_positions], axis=0
                )
                position_diff = ops.abs(
                    block_positions[:, None] - key_positions[None, :]
                )
                if self.sliding_window_size:
                    mask_2d = position_diff <= (self.sliding_window_size - 1)
                else:
                    mask_2d = ops.ones_like(position_diff, dtype="bool")
                # Cache slots below `valid_context_len` hold real,
                # already-written context; later slots are unwritten.
                context_valid = context_positions < valid_context_len
                block_valid = ops.ones((q_len,), dtype="bool")
                key_valid = ops.concatenate(
                    [context_valid, block_valid], axis=0
                )
                mask_2d = ops.logical_and(mask_2d, key_valid[None, :])
                mask = ops.broadcast_to(
                    mask_2d[None, :, :],
                    (batch_size, q_len, max_length + q_len),
                )
                if decoder_padding_mask is not None:
                    # `decoder_padding_mask` covers the block only — the
                    # cached context slot is always a single real,
                    # unpadded position written by the caller.
                    padding = ops.cast(decoder_padding_mask, "bool")
                    context_padding = ops.ones(
                        (batch_size, max_length), dtype="bool"
                    )
                    padding = ops.concatenate(
                        [context_padding, padding], axis=1
                    )
                    mask = ops.logical_and(mask, padding[:, None, :])
                return mask

            if context_hidden_states is not None:
                context_len = ops.shape(context_hidden_states)[1]
                kv_len = context_len + q_len
            else:
                kv_len = q_len
            mask = ops.ones((batch_size, q_len, kv_len), dtype="bool")
            if decoder_padding_mask is not None:
                padding = ops.cast(decoder_padding_mask, "bool")
                if context_hidden_states is not None:
                    context_padding = ops.ones(
                        (batch_size, context_len), dtype="bool"
                    )
                    padding = ops.concatenate(
                        [context_padding, padding], axis=1
                    )
                mask = ops.logical_and(mask, padding[:, None, :])
            return mask

        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        batch_size = ops.shape(decoder_sequence)[0]
        input_length = output_length = ops.shape(decoder_sequence)[1]
        if self_attention_cache is not None:
            input_length = ops.shape(self_attention_cache)[2]

        cache_update_index = (
            0
            if self_attention_cache_update_index is None
            else self_attention_cache_update_index
        )
        causal_mask = compute_causal_mask(
            batch_size, input_length, output_length, cache_update_index
        )
        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

    def call(
        self,
        decoder_sequence,
        context_hidden_states=None,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        training=None,
    ):
        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=decoder_sequence,
            context_hidden_states=context_hidden_states,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=(
                self_attention_cache_update_index
            ),
        )

        residual = decoder_sequence
        x = self._input_layernorm(decoder_sequence)
        x = self._self_attention_layer(
            x,
            context_hidden_states=context_hidden_states,
            attention_mask=self_attention_mask,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
            training=training,
        )
        if self_attention_cache is not None:
            x, self_attention_cache = x
        if self.use_sandwich_norm:
            x = self._post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self._pre_feedforward_layernorm(x)
        gate_output = self._feedforward_gate_dense(x)
        gate_output = ops.cast(gate_output, "float32")
        gate_output = self.activation(gate_output)
        gate_output = ops.cast(gate_output, self.compute_dtype)
        up_output = self._feedforward_up_dense(x)
        x = self._feedforward_down_dense(gate_output * up_output)
        if self.use_sandwich_norm:
            x = self._post_feedforward_layernorm(x)
        decoder_output = residual + x

        if self_attention_cache is not None:
            return decoder_output, self_attention_cache
        return decoder_output

    def compute_output_shape(self, decoder_sequence_shape):
        return decoder_sequence_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "hidden_activation": self.hidden_activation,
                "rms_norm_eps": self.rms_norm_eps,
                "post_norm_eps": self.post_norm_eps,
                "qk_scale_factor": self.qk_scale_factor,
                "use_rope": self.use_rope,
                "rope_max_wavelength": self.rope_max_wavelength,
                "sliding_window_size": self.sliding_window_size,
                "use_bidirectional_attention": (
                    self.use_bidirectional_attention
                ),
                "enable_qk_scale_and_gate": self.enable_qk_scale_and_gate,
                "qk_norm_with_scale": self.qk_norm_with_scale,
                "use_sandwich_norm": self.use_sandwich_norm,
                "use_centered_norm": self.use_centered_norm,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
            }
        )
        return config
