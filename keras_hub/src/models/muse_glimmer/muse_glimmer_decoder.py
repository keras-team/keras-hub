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
from keras_hub.src.utils.keras_utils import clone_initializer


class MuseGlimmerTextDecoder(keras.layers.Layer):
    """MuseGlimmer transformer decoder layer.

    A "sandwich norm" block: `input_layernorm` and `pre_feedforward_norm`
    use `rms_norm_eps`; `post_attention_norm` and `post_feedforward_norm`
    use the separate (tighter) `post_norm_eps`. All four are the
    "centered" `(1 + weight)` RMSNorm variant. See
    `modeling_muse_glimmer.py`, `MuseGlimmerTextDecoderLayer.forward`.

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
        self.dropout = dropout
        self.activation = keras.activations.get(hidden_activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.supports_masking = True

    def build(self, decoder_sequence_shape):
        self.hidden_dim = decoder_sequence_shape[-1]

        self._input_layernorm = MuseGlimmerCenteredRMSNorm(
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
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self._self_attention_layer.build(decoder_sequence_shape)

        self._post_attention_layernorm = MuseGlimmerCenteredRMSNorm(
            eps=self.post_norm_eps,
            dtype=self.dtype_policy,
            name="post_attention_layernorm",
        )
        self._post_attention_layernorm.build(decoder_sequence_shape)

        self._pre_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(
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
        decoder_padding_mask,
        decoder_attention_mask,
        self_attention_cache,
        self_attention_cache_update_index,
    ):
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
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        training=None,
    ):
        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=decoder_sequence,
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
            attention_mask=self_attention_mask,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
            training=training,
        )
        if self_attention_cache is not None:
            x, self_attention_cache = x
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
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
            }
        )
        return config
