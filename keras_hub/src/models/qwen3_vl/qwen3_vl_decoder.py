import keras
from keras import ops

from keras_hub.src.models.qwen3.qwen3_decoder import Qwen3TransformerDecoder
from keras_hub.src.models.qwen3_vl.qwen3_vl_attention import Qwen3_VLAttention
from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import Qwen3_VLLayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer


class Qwen3_VLTransformerDecoder(Qwen3TransformerDecoder):
    """A Transformer decoder layer for the Qwen3-VL text model.

    The layer is a pre-norm block with interleaved multimodal RoPE
    attention and a SwiGLU feed-forward network. It reuses the attributes, the
    self-attention mask and the output shape of `Qwen3TransformerDecoder`, and
    swaps in `Qwen3_VLAttention` for the attention.

    Args:
        intermediate_dim: int. Hidden size of the feed-forward network.
        head_dim: int. Size of each attention head.
        num_query_heads: int. Number of query heads.
        num_key_value_heads: int. Number of key/value heads.
        rope_max_wavelength: float. Base of the rotary frequencies.
        mrope_section: list of 3 ints. Frequencies used by the temporal,
            height and width position channels.
        layer_norm_epsilon: float. Epsilon of the RMSNorm layers.
        activation: Activation of the feed-forward gate.
        kernel_initializer: Initializer for the dense kernels.
        dropout: float. Dropout rate.
    """

    def __init__(
        self,
        intermediate_dim,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=5000000.0,
        mrope_section=(24, 20, 20),
        layer_norm_epsilon=1e-6,
        activation="silu",
        kernel_initializer="glorot_uniform",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            intermediate_dim=intermediate_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_max_wavelength=rope_max_wavelength,
            activation=activation,
            layer_norm_epsilon=layer_norm_epsilon,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            **kwargs,
        )
        self.mrope_section = list(mrope_section)

    def build(self, decoder_sequence_shape):
        self.hidden_dim = decoder_sequence_shape[-1]

        self._self_attention_layer = Qwen3_VLAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rope_max_wavelength=self.rope_max_wavelength,
            mrope_section=self.mrope_section,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            layer_norm_epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self._self_attention_layer.build(decoder_sequence_shape)
        self._self_attention_layernorm = Qwen3_VLLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layernorm",
        )
        self._self_attention_layernorm.build(decoder_sequence_shape)
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(decoder_sequence_shape)
        self._feedforward_gate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_gate_dense",
        )
        self._feedforward_gate_dense.build(decoder_sequence_shape)
        self._feedforward_output_dense = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        self._feedforward_output_dense.build(
            self._feedforward_gate_dense.compute_output_shape(
                decoder_sequence_shape
            )
        )
        self._feedforward_layernorm = Qwen3_VLLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layernorm",
        )
        self._feedforward_layernorm.build(decoder_sequence_shape)
        self.built = True

    def call(
        self,
        x,
        decoder_padding_mask=None,
        position_ids=None,
        cache=None,
        cache_update_index=None,
        rotary_embeddings=None,
        training=None,
    ):
        """Forward pass of the decoder layer.

        Args:
            x: Tensor of shape `(batch, seq_len, hidden_dim)`.
            decoder_padding_mask: Optional padding mask of shape
                `(batch, seq_len)`.
            position_ids: Optional int Tensor of shape `(batch, 3, seq_len)`
                with the T, H and W positions. `None` means text-only
                positions.
            cache: Optional key/value cache of this layer.
            cache_update_index: Index at which the cache is updated.
            rotary_embeddings: Optional `(cos, sin)` tuple shared by all the
                layers of a model, see
                `Qwen3_VLAttention.compute_rotary_embeddings`.
            training: Boolean, whether in training mode.

        Returns:
            The layer output, plus the updated cache if `cache` is given.
        """
        attention_mask = self._compute_self_attention_mask(
            decoder_sequence=x,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=None,
            self_attention_cache=cache,
            self_attention_cache_update_index=cache_update_index,
        )
        residual = x
        x = self._self_attention_layernorm(x)
        x = self._self_attention_layer(
            hidden_states=x,
            attention_mask=attention_mask,
            cache=cache,
            cache_update_index=cache_update_index,
            position_ids=position_ids,
            rotary_embeddings=rotary_embeddings,
            training=training,
        )
        if cache is not None:
            x, cache = x
        x = self._self_attention_dropout(x, training=training)
        x = x + residual

        residual = x
        x = self._feedforward_layernorm(x)
        # Run the activation in float32, as torch does for SiLU.
        gate = ops.cast(self._feedforward_gate_dense(x), "float32")
        gate = ops.cast(self.activation(gate), self.compute_dtype)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(ops.multiply(x, gate))
        x = x + residual

        if cache is not None:
            return x, cache
        return x

    def compute_output_spec(self, decoder_sequence, *args, cache=None, **kw):
        output = keras.KerasTensor(
            decoder_sequence.shape, dtype=self.compute_dtype
        )
        if cache is None:
            return output
        return output, keras.KerasTensor(cache.shape, dtype=cache.dtype)

    def get_config(self):
        config = super().get_config()
        config.pop("rope_scaling_factor")
        config.update(
            {"head_dim": self.head_dim, "mrope_section": self.mrope_section}
        )
        return config
