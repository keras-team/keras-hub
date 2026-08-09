import keras

from keras_hub.src.utils.keras_utils import clone_initializer


def t5gemma_kernel_initializer(initializer_range=0.01):
    """Creates a RandomNormal initializer for T5Gemma kernels.

    Args:
        initializer_range: float, The standard deviation of the normal
            distribution. Defaults to `0.01`.

    Returns:
        keras.initializers.RandomNormal: A Keras RandomNormal initializer.
    """
    return keras.initializers.RandomNormal(mean=0.0, stddev=initializer_range)


class T5GemmaMLP(keras.layers.Layer):
    """Multilayer Perceptron (MLP) block for the T5Gemma model.

    This layer implements the feed-forward part of a transformer block,
    consisting of two dense layers with a GELU activation and dropout.

    Args:
        hidden_size: int, The dimensionality of the input and output hidden
            states.
        intermediate_size: int, The dimensionality of the intermediate layer.
        hidden_activation: str, The activation function to use, e.g.,
            "gelu_approximate".
        dropout_rate: float, The dropout rate applied to the intermediate
            hidden states.
        initializer_range: float, The range for the random normal initializer
            for kernel weights. Defaults to `0.02`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_activation,
        dropout_rate,
        initializer_range=0.02,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_activation = hidden_activation
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.kernel_initializer = t5gemma_kernel_initializer(initializer_range)

        self.gate_proj = keras.layers.Dense(
            self.intermediate_size,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="gate_proj",
        )
        self.up_proj = keras.layers.Dense(
            self.intermediate_size,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="up_proj",
        )
        self.down_proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="down_proj",
        )
        if self.hidden_activation == "gelu_approximate":
            # NOTE: `gelu_pytorch_tanh` is the same as `gelu(approximate=True)`.
            self.act_fn = lambda x: keras.activations.gelu(x, approximate=True)
        else:
            self.act_fn = keras.activations.get(self.hidden_activation)
        self.dropout = keras.layers.Dropout(
            self.dropout_rate,
            dtype=self.dtype_policy,
            name="mlp_dropout",
        )

    def build(self, input_shape):
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)
        intermediate_shape = self.gate_proj.compute_output_shape(input_shape)
        self.dropout.build(intermediate_shape)
        self.down_proj.build(intermediate_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, training=None):
        hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        hidden_states = self.dropout(hidden_states, training=training)
        down_proj = self.down_proj(hidden_states)
        return down_proj

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "hidden_activation": self.hidden_activation,
                "dropout_rate": self.dropout_rate,
                "initializer_range": self.initializer_range,
            }
        )
        return config
