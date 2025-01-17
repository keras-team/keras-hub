import keras

BN_AXIS = 3


class SqueezeAndExcite2D(keras.layers.Layer):
    """
    Description:
        This layer applies a content-aware mechanism to adaptively assign
        channel-wise weights. It uses global average pooling to compress
        feature maps into single values, which are then processed by
        two Conv1D layers: the first reduces the dimensionality, and
        the second restores it.
    Args:
        filters: Number of input and output filters. The number of input and
            output filters is same.
        bottleneck_filters: (Optional) Number of bottleneck filters. Defaults
            to `0.25 * filters`
        squeeze_activation: (Optional) String, callable (or
            keras.layers.Layer) or keras.activations.Activation instance
            denoting activation to be applied after squeeze convolution.
            Defaults to `relu`.
        excite_activation: (Optional) String, callable (or
            keras.layers.Layer) or keras.activations.Activation instance
            denoting activation to be applied after excite convolution.
            Defaults to `sigmoid`.
        name: Name of the layer
    """

    def __init__(
        self,
        filters,
        bottleneck_filters=None,
        squeeze_activation="relu",
        excite_activation="sigmoid",
        name=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.bottleneck_filters = bottleneck_filters
        self.squeeze_activation = squeeze_activation
        self.excite_activation = excite_activation
        self.name = name

        image_data_format = keras.config.image_data_format()
        if image_data_format == "channels_last":
            self.spatial_dims = (1, 2)
        else:
            self.spatial_dims = (2, 3)

        self.conv_reduce = keras.layers.Conv2D(
            bottleneck_filters,
            (1, 1),
            data_format=image_data_format,
            name=f"{name}_conv_reduce",
        )
        self.act1 = keras.layers.Activation(
            self.squeeze_activation, name=self.name + "squeeze_activation"
        )

        self.conv_expand = keras.layers.Conv2D(
            filters,
            (1, 1),
            data_format=image_data_format,
            name=f"{name}_conv_expand",
        )
        self.gate = keras.layers.Activation(
            self.excite_activation, name=self.name + "excite_activation"
        )

    def build(self, input_shape):
        if self.name is None:
            self.name = keras.backend.get_uid("block0")

    def call(self, inputs):
        x_se = keras.ops.mean(inputs, axis=self.spatial_dims, keepdims=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return inputs * self.gate(x_se)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "bottleneck_filters": self.bottleneck_filters,
                "squeeze_activation": self.squeeze_activation,
                "excite_activation": self.excite_activation,
                "name": self.name,
                "spatial_dims": self.spatial_dims,
            }
        )

        return config
