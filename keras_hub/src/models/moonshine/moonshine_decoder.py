import keras

from keras_hub.src.layers.modeling.transformer_decoder import TransformerDecoder
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineMLP
from keras_hub.src.models.moonshine.moonshine_layers import (
    moonshine_kernel_initializer,
)
from keras_hub.src.models.moonshine.moonshine_multi_head_attention import (
    MoonshineMultiHeadAttention,
)
from keras_hub.src.utils.keras_utils import clone_initializer


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineDecoderBlock(TransformerDecoder):
    """Moonshine decoder block for sequence processing.

    This layer implements a decoder block that includes self-attention with
    causal masking, cross-attention with precomputed key/value pairs, and a
    feedforward network.

    Args:
        hidden_dim: int. The dimensionality of the model's hidden
            representations.
        intermediate_dim: int. The dimensionality of the intermediate
            representations in the feedforward network.
        num_heads: int. The number of attention heads for multi-head attention
            mechanisms.
        feedforward_expansion_factor: int, optional. A multiplicative factor for
            scaling the feedforward network dimension. Defaults to 4.
        use_swiglu_activation: bool, optional. Whether to use the SwiGLU
            activation in the feedforward network for improved performance.
            Defaults to True.
        pad_head_dim_to_multiple_of: int, optional. If specified, pads the head
            dimension to be a multiple of this value for performance
            optimization. Defaults to None.
        initializer_range: float, optional. The standard deviation of the
            truncated normal distribution used to initialize model weights.
            Defaults to 0.02.
        attention_bias: bool, optional. Whether to add a bias term to the
            attention computations. Defaults to False.
        attention_dropout: float, optional. The dropout rate applied to
            attention weights during training. Defaults to 0.0.
        dtype: str, optional. The data type to use for model computations and
            weights. Defaults to None.
        **kwargs: Additional keyword arguments passed to the base layer.
    """

    # References:
    # Defined and formulated based on the UsefulSensors implementation of the
    # DecoderLayer class (https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L348-L466).

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_heads,
        feedforward_expansion_factor=4,
        use_swiglu_activation=True,
        pad_head_dim_to_multiple_of=None,
        initializer_range=0.02,
        attention_bias=False,
        attention_dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        kwargs.pop("dropout", None)
        kwargs.pop("activation", None)
        kwargs.pop("kernel_initializer", None)
        self.kernel_initializer = moonshine_kernel_initializer(
            initializer_range=initializer_range
        )
        super().__init__(
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            activation="gelu" if use_swiglu_activation else "silu",
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=dtype,
            **kwargs,
        )
        self.initializer_range = initializer_range
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.use_swiglu_activation = use_swiglu_activation
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias

        self.head_dim = hidden_dim // num_heads
        if pad_head_dim_to_multiple_of is not None:
            self.head_dim = (
                (self.head_dim + pad_head_dim_to_multiple_of - 1)
                // pad_head_dim_to_multiple_of
            ) * pad_head_dim_to_multiple_of

        self.norm1 = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            dtype=self.dtype,
        )
        self.self_attention = MoonshineMultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            use_causal_mask=True,
            apply_rotary_embedding=True,
            dtype=self.dtype,
        )
        self.norm2 = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            dtype=self.dtype,
        )
        self.cross_attention = MoonshineMultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            use_causal_mask=False,
            apply_rotary_embedding=False,
            dtype=self.dtype,
        )
        self.norm3 = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            dtype=self.dtype,
        )
        self.ff = MoonshineMLP(
            hidden_dim=hidden_dim,
            feedforward_expansion_factor=feedforward_expansion_factor,
            use_swiglu_activation=use_swiglu_activation,
            initializer_range=initializer_range,
            dtype=self.dtype,
        )

    def build(self, decoder_sequence_shape, encoder_sequence_shape=None):
        if encoder_sequence_shape is None:
            raise ValueError(
                "Encoder sequence shape must be provided for "
                "MoonshineDecoderBlock."
            )
        context_shape = encoder_sequence_shape  # Shape of context

        # Build sublayers.
        self.norm1.build(decoder_sequence_shape)
        self.norm2.build(decoder_sequence_shape)
        self.norm3.build(decoder_sequence_shape)

        self.self_attention.build(
            query_shape=decoder_sequence_shape,
            key_shape=decoder_sequence_shape,
            value_shape=decoder_sequence_shape,
        )

        self.cross_attention.build(
            query_shape=decoder_sequence_shape,
            key_shape=context_shape,
            value_shape=context_shape,
        )

        self.ff.build(decoder_sequence_shape)
        self.built = True

    def _compute_self_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            inputs=decoder_sequence,
            padding_mask=decoder_padding_mask,
            attention_mask=None,
        )
        if self.self_attention.use_causal_mask:
            batch_size = keras.ops.shape(decoder_sequence)[0]
            output_length = keras.ops.shape(decoder_sequence)[1]
            current_cache_update_index = (
                0
                if self_attention_cache_update_index is None
                else self_attention_cache_update_index
            )
            if self_attention_cache is not None:
                input_length = keras.ops.shape(self_attention_cache)[2]
            else:
                input_length = output_length
            causal_mask = compute_causal_mask(
                batch_size,
                input_length,
                output_length,
                current_cache_update_index,
            )
            return (
                keras.ops.minimum(decoder_mask, causal_mask)
                if decoder_mask is not None
                else causal_mask
            )
        return decoder_mask

    def call(
        self,
        decoder_sequence,
        encoder_sequence,
        rotary_embedding,
        encoder_attention_mask=None,
        decoder_padding_mask=None,
        encoder_padding_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        cross_attention_cache=None,
        cross_attention_cache_update_index=None,
        training=None,
    ):
        x = decoder_sequence
        context = encoder_sequence
        has_self_attention_cache = self_attention_cache is not None
        has_cross_attention_cache = cross_attention_cache is not None

        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=x,
            decoder_padding_mask=decoder_padding_mask,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=self_attention_cache_update_index,
        )

        # Self attention block.
        residual = x
        x_norm1 = self.norm1(x)
        x_self_attn = self.self_attention(
            query=x_norm1,
            key=x_norm1,
            value=x_norm1,
            rotary_embedding=rotary_embedding,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
            attention_mask=self_attention_mask,
            training=training,
        )
        if has_self_attention_cache:
            x_self_attn, self_attention_cache = x_self_attn
        x = x_self_attn + residual
        # Cross attention block.
        residual = x
        x_norm2 = self.norm2(x)
        cross_attention_mask = merge_padding_and_attention_mask(
            inputs=encoder_sequence,
            padding_mask=encoder_padding_mask,
            attention_mask=encoder_attention_mask,
        )
        x_cross_attn = self.cross_attention(
            query=x_norm2,
            key=context,
            value=context,
            cache=cross_attention_cache,
            cache_update_index=cross_attention_cache_update_index,
            attention_mask=cross_attention_mask,
            training=training,
        )
        if has_cross_attention_cache:
            x_cross_attn, cross_attention_cache = x_cross_attn
        x = x_cross_attn + residual
        residual = x
        x_norm3 = self.norm3(x)
        x_ff = self.ff(x_norm3)
        x = x_ff + residual

        if has_self_attention_cache:
            return x, self_attention_cache
        return x

    def compute_output_shape(
        self, decoder_sequence_shape, encoder_sequence_shape=None
    ):
        return decoder_sequence_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "feedforward_expansion_factor": self.feedforward_expansion_factor,  # noqa: E501
                "use_swiglu_activation": self.use_swiglu_activation,
                "pad_head_dim_to_multiple_of": self.pad_head_dim_to_multiple_of,  # noqa: E501
                "initializer_range": self.initializer_range,
                "attention_bias": self.attention_bias,
                "attention_dropout": self.attention_dropout,
                "dtype": self.dtype,
            }
        )
        return config
