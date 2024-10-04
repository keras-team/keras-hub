import keras


class PredictionHead(keras.layers.Layer):
    """The classification/box predictions head.

    Arguments:
        output_filters: int. Number of convolution filters in the final layer.
        num_filters: int. Number of convolution filters used in base layers.
            Defaults to `256`.
        num_conv_layers: int. Number of convolution layers before final layer.
            Defaults to `4`.
        kernel_initializer: `str` or `keras.initializers`.
            The kernel initializer for the convolution layers.
            Defaults to `"random_normal"`.
        bias_initializer: `str` or `keras.initializers`.
            The bias initializer for the convolution layers.
            Defaults to `"zeros"`.
        kernel_regularizer: `str` or `keras.regularizers`.
            The kernel regularizer for the convolution layers.
            Defaults to `None`.
        bias_regularizer: `str` or `keras.regularizers`.
            The bias regularizer for the convolution layers.
            Defaults to `None`.

    Returns:
      A function representing either the classification
        or the box regression head depending on `output_filters`.
    """

    def __init__(
        self,
        output_filters,
        num_filters,
        num_conv_layers,
        activation="relu",
        kernel_initializer="random_normal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.output_filters = output_filters
        self.num_filters = num_filters
        self.num_conv_layers = num_conv_layers
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        if kernel_regularizer is not None:
            self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        else:
            self.kernel_regularizer = None
        if bias_regularizer is not None:
            self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        else:
            self.bias_regularizer = None

        self.data_format = keras.backend.image_data_format()

    def build(self, input_shape):
        self.conv_layers = [
            keras.layers.Conv2D(
                self.num_filters,
                kernel_size=3,
                padding="same",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activation=self.activation,
                data_format=self.data_format,
                dtype=self.dtype_policy,
            )
            for _ in range(self.num_conv_layers)
        ]

        intermediate_shape = input_shape
        for conv in self.conv_layers:
            conv.build(intermediate_shape)
            intermediate_shape = (
                input_shape[:-1] + (self.num_filters,)
                if self.data_format == "channels_last"
                else (input_shape[0], self.num_filters) + (input_shape[1:-1])
            )

        self.prediction_layer = keras.layers.Conv2D(
            self.output_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            dtype=self.dtype_policy,
        )

        self.prediction_layer.build(
            (None, None, None, self.num_filters)
            if self.data_format == "channels_last"
            else (None, self.num_filters, None, None)
        )

        self.built = True

    def call(self, input):
        x = input
        for conv in self.conv_layers:
            x = conv(x)
        output = self.prediction_layer(x)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_filters": self.output_filters,
                "num_filters": self.num_filters,
                "num_conv_layers": self.num_conv_layers,
                "activation": keras.activations.serialize(self.activation),
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "kernel_regularizer": (
                    keras.regularizers.serialize(self.kernel_regularizer)
                    if self.kernel_regularizer is not None
                    else None
                ),
                "bias_regularizer": (
                    keras.regularizers.serialize(self.bias_regularizer)
                    if self.bias_regularizer is not None
                    else None
                ),
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return (
            input_shape[:-1] + (self.output_filters,)
            if self.data_format == "channels_last"
            else (input_shape[0],) + (self.output_filters,) + input_shape[1:-1]
        )
