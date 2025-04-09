import keras

from keras_hub.src.layers.modeling.transformer_decoder import TransformerDecoder
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

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) < 2:
            raise ValueError(
                "Expected input_shape to be a list of at least two shapes."
            )
        decoder_sequence_shape = (
            input_shape[0]["decoder_token_ids"]  # Shape of x
            if isinstance(input_shape[0], dict)
            else input_shape[0]
        )
        context_shape = (
            input_shape[1]["input_values"]  # Shape of context
            if isinstance(input_shape[1], dict)
            else input_shape[1]
        )

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

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) < 3:
            raise ValueError(
                "Expected input_shape to be a list/tuple of three shapes "
                "(decoder_sequence, context, rotary_embedding)."
            )
        decoder_sequence_shape = input_shape[0]
        batch_size = decoder_sequence_shape[0]
        sequence_length = decoder_sequence_shape[1]
        num_heads = self.self_attention._num_heads
        head_dim = self.self_attention._key_dim
        cache_shape = (batch_size, 2, sequence_length, num_heads, head_dim)
        return decoder_sequence_shape, cache_shape

    def call(
        self,
        inputs,
        training=None,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
    ):
        x, context, rotary_embedding = inputs

        residual = x
        x = self.norm1(x)
        x, self_cache = self.self_attention(
            query=x,
            key=x,
            value=x,
            rotary_embedding=rotary_embedding,
            attention_mask=decoder_attention_mask,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
            training=training,
        )
        x = x + residual

        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attention(
            query=x,
            key=context,
            value=context,
            attention_mask=encoder_attention_mask,
            training=training,
        )
        x = x + residual

        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual

        return x, self_cache

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
