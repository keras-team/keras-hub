import keras
from keras import activations
from keras import initializers

from keras_hub.src.models.roformer_v2.roformer_v2_attention import (
    RoformerAttention,
)
from keras_hub.src.models.roformer_v2.roformer_v2_attention import RoformerNorm


@keras.saving.register_keras_serializable(package="keras_hub")
class RoformerV2Encoder(keras.layers.Layer):
    """A Transformer Encoder layer for the Roformer backbone."""

    def __init__(
        self,
        heads,
        head_size,
        intermediate_size=None,
        max_wavelength=10000,
        dropout=0,
        activation="relu",
        use_bias=False,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.intermediate_size = intermediate_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.max_wavelength = max_wavelength
        self.dropout = dropout
        self.activation = activations.get(activation)

    def build(self, input_shape):
        super().build(input_shape)
        self.attention_layer = RoformerAttention(
            heads=self.heads,
            head_size=self.head_size,
            use_bias=self.use_bias,
            max_wavelength=self.max_wavelength,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_layer",
        )
        self.attention_layer.build(input_shape)

        self.dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )
        self.dropout_layer.build([])

        # Feedforward layers.
        self.feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_size,
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            dtype=self.dtype_policy,
            activation=self.activation,
            name="feedforward_intermediate_dense",
        )
        self.feedforward_intermediate_dense.build(input_shape)

        self.feedforward_output_dense = keras.layers.Dense(
            input_shape[-1],
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )

        self.feedforward_output_dense.build(
            [None, None, self.intermediate_size]
        )

        self.attention_norm = RoformerNorm(
            epsilon=keras.backend.epsilon(),
            name="attention_norm",
            dtype=self.dtype_policy,
        )
        self.attention_norm.build(input_shape)

        self.feedforward_norm = RoformerNorm(
            epsilon=keras.backend.epsilon(),
            name="feedforward_norm",
            dtype=self.dtype_policy,
        )
        self.feedforward_norm.build(input_shape)

    def call(self, x, attention_mask=None):
        attention_output = self.attention_layer(
            x,
            attention_mask=attention_mask,
        )

        residual = x + self.dropout_layer(attention_output)
        x = self.attention_norm(residual)

        intermediate_output = self.feedforward_intermediate_dense(x)
        feedroward_output = self.feedforward_output_dense(intermediate_output)

        residual = x + self.dropout_layer(feedroward_output)
        return self.feedforward_norm(residual)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "heads": self.heads,
                "head_size": self.head_size,
                "intermediate_size": self.intermediate_size,
                "max_wavelength": self.max_wavelength,
                "use_bias": self.use_bias,
                "activation": activations.serialize(self.activation),
                "dropout": self.dropout,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config
