import keras
from keras import ops


class DebertaV2ConvLayer(keras.layers.Layer):
    """Convolution block used by the original DeBERTa-v2 checkpoints.

    DeBERTa-v2 models (e.g. `microsoft/deberta-v2-xlarge`) apply a 1D
    convolution to the embedding output and add it to the output of the first
    transformer layer, followed by layer normalization. DeBERTa-v3 checkpoints
    do not use this block.

    Args:
        hidden_dim: int. The hidden size of the model.
        kernel_size: int. The size of the convolution kernel.
        groups: int. The number of convolution groups.
        activation: string or callable. The activation applied to the
            convolution output.
        dropout: float. Dropout probability applied before the activation.
        layer_norm_epsilon: float. The epsilon of the layer normalization.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer of the convolution.
    """

    def __init__(
        self,
        hidden_dim,
        kernel_size=3,
        groups=1,
        activation="gelu",
        dropout=0.0,
        layer_norm_epsilon=1e-7,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.groups = groups
        self.activation = keras.activations.get(activation)
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        self.conv = keras.layers.Conv1D(
            filters=hidden_dim,
            kernel_size=kernel_size,
            padding="same",
            groups=groups,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="conv",
        )
        self.dropout_layer = keras.layers.Dropout(
            dropout,
            dtype=self.dtype_policy,
            name="dropout",
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm",
        )

    def build(self, input_shape):
        self.conv.build(input_shape)
        self.layer_norm.build(input_shape)
        self.built = True

    def call(self, embeddings, residual, padding_mask, training=None):
        mask = ops.expand_dims(ops.cast(padding_mask, self.compute_dtype), -1)
        # Padding positions are zero in the reference implementation, so they
        # must not leak into neighbouring real tokens through the convolution.
        x = self.conv(embeddings * mask)
        x = x * mask
        x = self.activation(self.dropout_layer(x, training=training))
        x = self.layer_norm(residual + x)
        return x * mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "kernel_size": self.kernel_size,
                "groups": self.groups,
                "activation": keras.activations.serialize(self.activation),
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
