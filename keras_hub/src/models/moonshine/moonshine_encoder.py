import keras

from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineLinearGeLU
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineSwiGLU
from keras_hub.src.models.moonshine.moonshine_multi_head_attention import (
    MoonshineMultiHeadAttention,
)


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineEncoderBlock(TransformerEncoder):
    """
    Moonshine encoder block for sequence processing.

    Implements a standard encoder block with self-attention and feedforward
    sublayers, including residual connections and layer normalization. The
    implementation utilizes Moonshine-specific attention and feedforward
    mechanisms.

    Args:
        hidden_dim: int. The dimensionality of the model's hidden
            representations throughout the block.
        intermediate_dim: int. The dimensionality used in projections before
            applying non-linearities.
        num_heads: int. The number of attention heads for multi-head attention
            computation.
        feedforward_expansion_factor: int, optional. A multiplier for expanding
            the dimension in the feedforward network. Defaults to 4.
        use_swiglu_activation: bool, optional. Whether to use SwiGLU activation
            (True) or LinearGeLU (False) in the feedforward sublayer. Defaults
            to False.
        pad_head_dim_to_multiple_of: int, optional. If specified, pads the head
            dimension to be a multiple of this value for hardware optimization.
            Defaults to None.
        initializer_range: float, optional. The standard deviation of the
            truncated normal distribution used for weight initialization.
            Defaults to 0.02.
        attention_bias: bool, optional. Whether to use a bias term in the
            attention mechanism. Defaults to False.
        attention_dropout: float, optional. The dropout rate applied to the
            attention weights. Defaults to 0.0.
        dtype: str, optional. The data type to use for model computations and
            weights. Defaults to None.
        **kwargs: Additional keyword arguments passed to the base layer.

    Returns:
        MoonshineEncoderBlock: An instance of `MoonshineEncoderBlock`, which
        is a specialized Transformer encoder block implementing
        Moonshine-specific self-attention and feedforward sublayers.

    ## References
    Defined and formulated based on the
    [UsefulSensors implementation of the EncoderLayer](https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L124)
    class.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_heads,
        feedforward_expansion_factor=4,
        use_swiglu_activation=False,
        pad_head_dim_to_multiple_of=None,
        dtype=None,
        initializer_range=0.02,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            activation="gelu" if use_swiglu_activation else "relu",
            layer_norm_epsilon=1e-5,
            kernel_initializer=keras.initializers.RandomNormal(
                stddev=initializer_range
            ),
            bias_initializer="zeros",
            normalize_first=True,
            dtype=dtype,
            **kwargs,
        )
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.use_swiglu_activation = use_swiglu_activation

        # Self-attention sublayers.
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of

        self.head_dim = hidden_dim // num_heads
        if pad_head_dim_to_multiple_of is not None:
            self.head_dim = (
                (self.head_dim + pad_head_dim_to_multiple_of - 1)
                // pad_head_dim_to_multiple_of
            ) * pad_head_dim_to_multiple_of

        self.self_attention_layer = MoonshineMultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                stddev=initializer_range
            ),
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            name="self_attention_layer",
            dtype=self.dtype,
        )
        self.self_attention_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            name="self_attention_layer_norm",
            dtype=self.dtype,
        )

        # Feedforward sublayers.
        self.feedforward_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            name="feedforward_layer_norm",
            dtype=self.dtype,
        )
        if use_swiglu_activation:
            self.feedforward = MoonshineSwiGLU(
                hidden_dim,
                feedforward_expansion_factor,
                kernel_initializer=keras.initializers.RandomNormal(
                    stddev=initializer_range
                ),
                name="feedforward",
                dtype=self.dtype,
            )
        else:
            self.feedforward = MoonshineLinearGeLU(
                hidden_dim,
                feedforward_expansion_factor,
                kernel_initializer=keras.initializers.RandomNormal(
                    stddev=initializer_range
                ),
                name="feedforward",
                dtype=self.dtype,
            )

    def build(self, input_shape):
        # Note: Avoid calling super.build(), as it creates downstream issues in
        # gradient tracking.
        # Build self-attention branch.
        self.self_attention_layer_norm.build(input_shape)
        self.self_attention_layer.build(input_shape, input_shape, input_shape)
        # Build feedforward branch.
        self.feedforward_layer_norm.build(input_shape)
        # The feedforward layer expects the last dimension to be hidden_dim.
        feed_forward_input_shape = list(input_shape)
        feed_forward_input_shape[-1] = self.hidden_dim
        self.feedforward.build(tuple(feed_forward_input_shape))
        self.built = True

    def call(self, inputs, rotary_embedding, training=None, **kwargs):
        x = inputs

        # Self-attention block with residual connection.
        attention_residual = x
        x = self.self_attention_layer_norm(x)
        x = self.self_attention_layer(
            query=x,
            value=x,
            key=x,
            rotary_embedding=rotary_embedding,
            training=training,
            **kwargs,
        )
        x = x + attention_residual

        # Feedforward block with residual connection.
        ff_residual = x
        x = self.feedforward_layer_norm(x)
        x = self.feedforward(x)
        x = x + ff_residual

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        # ==== Config ====
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "feedforward_expansion_factor": self.feedforward_expansion_factor,  # noqa: E501
                "use_swiglu_activation": self.use_swiglu_activation,
                "pad_head_dim_to_multiple_of": self.pad_head_dim_to_multiple_of,
                "initializer_range": self.initializer_range,
                "attention_bias": self.attention_bias,
                "attention_dropout": self.attention_dropout,
                "dtype": self.dtype,
            }
        )
        return config
