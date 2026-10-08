import keras

from keras_hub.src.utils.keras_utils import standardize_data_format


class PredictionHead(keras.layers.Layer):
    """A head for classification or bounding box regression predictions.

    Args:
        output_filters: int. The umber of convolution filters in the final
            layer. The number of output channels determines the prediction type:
                - **Classification**:
                    `output_filters = num_anchors * num_classes`
                    Predicts class probabilities for each anchor.
                - **Bounding Box Regression**:
                    `output_filters = num_anchors * 4` Predicts bounding box
                    offsets (x1, y1, x2, y2) for each anchor.
        num_filters: int. The number of convolution filters to use in the base
            layer.
        num_conv_layers: int. The number of convolution layers before the final
            layer.
        use_prior_probability: bool. Set to True to use prior probability in the
            bias initializer for the final convolution layer.
            Defaults to `False`.
        prior_probability: float. The prior probability value to use for
            initializing the bias. Only used if `use_prior_probability` is
            `True`. Defaults to `0.01`.
        kernel_initializer: `str` or `keras.initializers`. The kernel
            initializer for the convolution layers. Defaults to
            `"random_normal"`.
        bias_initializer: `str` or `keras.initializers`. The bias initializer
            for the convolution layers. Defaults to `"zeros"`.
        kernel_regularizer: `str` or `keras.regularizers`. The kernel
            regularizer for the convolution layers. Defaults to `None`.
        bias_regularizer: `str` or `keras.regularizers`. The bias regularizer
            for the convolution layers. Defaults to `None`.
        use_group_norm: bool. Whether to use Group Normalization after
            the convolution layers. Defaults to `False`.

    Returns:
        A function representing either the classification
            or the box regression head depending on `output_filters`.
    """

    def __init__(
        self,
        output_filters,
        num_filters,
        num_conv_layers,
        use_prior_probability=False,
        prior_probability=0.01,
        activation="relu",
        kernel_initializer="random_normal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        use_group_norm=False,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.output_filters = output_filters
        self.num_filters = num_filters
        self.num_conv_layers = num_conv_layers
        self.use_prior_probability = use_prior_probability
        self.prior_probability = prior_probability
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
        self.use_group_norm = use_group_norm
        self.data_format = standardize_data_format(data_format)

    def build(self, input_shape):
        intermediate_shape = input_shape
        self.conv_layers = []
        self.group_norm_layers = []
        for idx in range(self.num_conv_layers):
            conv = keras.layers.Conv2D(
                self.num_filters,
                kernel_size=3,
                padding="same",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                use_bias=not self.use_group_norm,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                data_format=self.data_format,
                dtype=self.dtype_policy,
                name=f"conv2d_{idx}",
            )
            conv.build(intermediate_shape)
            self.conv_layers.append(conv)
            intermediate_shape = (
                input_shape[:-1] + (self.num_filters,)
                if self.data_format == "channels_last"
                else (input_shape[0], self.num_filters) + (input_shape[1:-1])
            )
            if self.use_group_norm:
                group_norm = keras.layers.GroupNormalization(
                    groups=32,
                    axis=-1 if self.data_format == "channels_last" else 1,
                    dtype=self.dtype_policy,
                    name=f"group_norm_{idx}",
                )
                group_norm.build(intermediate_shape)
                self.group_norm_layers.append(group_norm)
        prior_probability = keras.initializers.Constant(
            -1
            * keras.ops.log(
                (1 - self.prior_probability) / self.prior_probability
            )
        )
        self.prediction_layer = keras.layers.Conv2D(
            self.output_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=(
                prior_probability
                if self.use_prior_probability
                else self.bias_initializer
            ),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            dtype=self.dtype_policy,
            name="logits_layer",
        )
        self.prediction_layer.build(
            (None, None, None, self.num_filters)
            if self.data_format == "channels_last"
            else (None, self.num_filters, None, None)
        )
        self.built = True

    def call(self, input):
        x = input
        for idx in range(self.num_conv_layers):
            x = self.conv_layers[idx](x)
            if self.use_group_norm:
                x = self.group_norm_layers[idx](x)
            x = self.activation(x)

        output = self.prediction_layer(x)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_filters": self.output_filters,
                "num_filters": self.num_filters,
                "num_conv_layers": self.num_conv_layers,
                "use_group_norm": self.use_group_norm,
                "use_prior_probability": self.use_prior_probability,
                "prior_probability": self.prior_probability,
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
