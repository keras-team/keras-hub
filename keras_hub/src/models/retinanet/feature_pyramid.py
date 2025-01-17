import math

import keras

from keras_hub.src.utils.keras_utils import standardize_data_format


class FeaturePyramid(keras.layers.Layer):
    """A Feature Pyramid Network (FPN) layer.

    This implements the paper:
        Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He,
        Bharath Hariharan, and Serge Belongie.
        Feature Pyramid Networks for Object Detection.
        (https://arxiv.org/pdf/1612.03144)

    Feature Pyramid Networks (FPNs) are basic components that are added to an
    existing feature extractor (CNN) to combine features at different scales.
    For the basic FPN, the inputs are features `Ci` from different levels of a
    CNN, which is usually the last block for each level, where the feature is
    scaled from the image by a factor of `1/2^i`.

    There is an output associated with each level in the basic FPN. The output
    Pi at level `i` (corresponding to Ci) is given by performing a merge
    operation on the outputs of:

    1) a lateral operation on Ci (usually a conv2D layer with kernel = 1 and
       strides = 1)
    2) a top-down upsampling operation from Pi+1 (except for the top most level)

    The final output of each level will also have a conv2D operation
    (typically with kernel = 3 and strides = 1).

    The inputs to the layer should be a dict with int keys should match the
    pyramid_levels, e.g. for `pyramid_levels` = [3,4,5], the expected input
    dict should be `{P3:c3, P4:c4, P5:c5}`.

    The output of the layer will have same structures as the inputs, a dict with
    extra coarser layers will be added based on the `max_level` provided.
    keys and value for each of the level.

    Args:
        min_level: int. The minimum level of the feature pyramid.
        max_level: int. The maximum level of the feature pyramid.
        use_p5: bool. If True, uses the output of the last layer (`P5` from
            Feature Pyramid Network) as input for creating coarser convolution
            layers (`P6`, `P7`).  If False, uses the direct input `P5`
            for creating coarser convolution  layers.
        num_filters: int. The number of filters in each feature map.
        activation: string or `keras.activations`. The activation function
            to be used in network.
            Defaults to `"relu"`.
        kernel_initializer: `str` or `keras.initializers`.
            The kernel initializer for the convolution layers.
            Defaults to `"VarianceScaling"`.
        bias_initializer: `str` or `keras.initializers`.
            The bias initializer for the convolution layers.
            Defaults to `"zeros"`.
        batch_norm_momentum: float.
            The momentum for the batch normalization layers.
            Defaults to `0.99`.
        batch_norm_epsilon: float.
            The epsilon for the batch normalization layers.
            Defaults to `0.001`.
        kernel_regularizer: `str` or `keras.regularizers`.
            The kernel regularizer for the convolution layers.
            Defaults to `None`.
        bias_regularizer: `str` or `keras.regularizers`.
            The bias regularizer for the convolution layers.
            Defaults to `None`.
        use_batch_norm: bool. Whether to use batch normalization.
            Defaults to `False`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.
    """

    def __init__(
        self,
        min_level,
        max_level,
        use_p5,
        num_filters=256,
        activation="relu",
        kernel_initializer="VarianceScaling",
        bias_initializer="zeros",
        batch_norm_momentum=0.99,
        batch_norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
        use_batch_norm=False,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if min_level > max_level:
            raise ValueError(
                f"Minimum level ({min_level}) must be less than or equal to "
                f"maximum level ({max_level})."
            )
        self.min_level = min_level
        self.max_level = max_level
        self.num_filters = num_filters
        self.use_p5 = use_p5
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_epsilon = batch_norm_epsilon
        self.use_batch_norm = use_batch_norm
        if kernel_regularizer is not None:
            self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        else:
            self.kernel_regularizer = None
        if bias_regularizer is not None:
            self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        else:
            self.bias_regularizer = None
        self.data_format = standardize_data_format(data_format)
        self.batch_norm_axis = -1 if data_format == "channels_last" else 1

    def build(self, input_shapes):
        input_shapes = {
            (
                input_name.split("_")[0]
                if "shape" in input_name
                else input_name
            ): input_shapes[input_name]
            for input_name in input_shapes
        }
        input_levels = [int(level[1]) for level in input_shapes]
        backbone_max_level = min(max(input_levels), self.max_level)
        # Build lateral layers
        self.lateral_conv_layers = {}
        for i in range(self.min_level, backbone_max_level + 1):
            level = f"P{i}"
            self.lateral_conv_layers[level] = keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=1,
                padding="same",
                data_format=self.data_format,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                dtype=self.dtype_policy,
                name=f"lateral_conv_{level}",
            )
            self.lateral_conv_layers[level].build(
                (None, None, None, input_shapes[level][-1])
                if self.data_format == "channels_last"
                else (None, input_shapes[level][1], None, None)
            )

        self.lateral_batch_norm_layers = {}
        if self.use_batch_norm:
            for i in range(self.min_level, backbone_max_level + 1):
                level = f"P{i}"
                self.lateral_batch_norm_layers[level] = (
                    keras.layers.BatchNormalization(
                        axis=self.batch_norm_axis,
                        momentum=self.batch_norm_epsilon,
                        epsilon=self.batch_norm_epsilon,
                        name=f"lateral_norm_{level}",
                    )
                )
                self.lateral_batch_norm_layers[level].build(
                    (None, None, None, self.num_filters)
                    if self.data_format == "channels_last"
                    else (None, self.num_filters, None, None)
                )

        # Build output layers
        self.output_conv_layers = {}
        for i in range(self.min_level, backbone_max_level + 1):
            level = f"P{i}"
            self.output_conv_layers[level] = keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=3,
                padding="same",
                data_format=self.data_format,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                dtype=self.dtype_policy,
                name=f"output_conv_{level}",
            )
            self.output_conv_layers[level].build(
                (None, None, None, self.num_filters)
                if self.data_format == "channels_last"
                else (None, self.num_filters, None, None)
            )

        # Build coarser layers
        for i in range(backbone_max_level + 1, self.max_level + 1):
            level = f"P{i}"
            self.output_conv_layers[level] = keras.layers.Conv2D(
                filters=self.num_filters,
                strides=2,
                kernel_size=3,
                padding="same",
                data_format=self.data_format,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                dtype=self.dtype_policy,
                name=f"coarser_{level}",
            )
            if i == backbone_max_level + 1 and self.use_p5:
                self.output_conv_layers[level].build(
                    (None, None, None, input_shapes[f"P{i - 1}"][-1])
                    if self.data_format == "channels_last"
                    else (None, input_shapes[f"P{i - 1}"][1], None, None)
                )
            else:
                self.output_conv_layers[level].build(
                    (None, None, None, self.num_filters)
                    if self.data_format == "channels_last"
                    else (None, self.num_filters, None, None)
                )

        # Build batch norm layers
        self.output_batch_norms = {}
        if self.use_batch_norm:
            for i in range(self.min_level, self.max_level + 1):
                level = f"P{i}"
                self.output_batch_norms[level] = (
                    keras.layers.BatchNormalization(
                        axis=self.batch_norm_axis,
                        momentum=self.batch_norm_epsilon,
                        epsilon=self.batch_norm_epsilon,
                        name=f"output_norm_{level}",
                    )
                )
                self.output_batch_norms[level].build(
                    (None, None, None, self.num_filters)
                    if self.data_format == "channels_last"
                    else (None, self.num_filters, None, None)
                )

        # The same upsampling layer is used for all levels
        self.top_down_op = keras.layers.UpSampling2D(
            size=2,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="upsampling",
        )
        # The same merge layer is used for all levels
        self.merge_op = keras.layers.Add(
            dtype=self.dtype_policy, name="merge_op"
        )

        self.built = True

    def call(self, inputs):
        """
        Inputs:
            The input to the model is expected to be an `Dict[Tensors]`,
                containing the feature maps on top of which the FPN
                will be added.

        Outputs:
            A dictionary of feature maps and added coarser levels based
                on minimum and maximum levels provided to the layer.
        """

        output_features = {}

        # Get the backbone max level
        input_levels = [int(level[1]) for level in inputs]
        backbone_max_level = min(max(input_levels), self.max_level)

        for i in range(backbone_max_level, self.min_level - 1, -1):
            level = f"P{i}"
            output = self.lateral_conv_layers[level](inputs[level])
            if i < backbone_max_level:
                # for the top most output, it doesn't need to merge with any
                # upper stream outputs
                upstream_output = self.top_down_op(output_features[f"P{i + 1}"])
                output = self.merge_op([output, upstream_output])
            output_features[level] = (
                self.lateral_batch_norm_layers[level](output)
                if self.use_batch_norm
                else output
            )

        # Post apply the output layers so that we don't leak them to the down
        # stream level
        for i in range(backbone_max_level, self.min_level - 1, -1):
            level = f"P{i}"
            output_features[level] = self.output_conv_layers[level](
                output_features[level]
            )

        for i in range(backbone_max_level + 1, self.max_level + 1):
            level = f"P{i}"
            feats_in = (
                inputs[f"P{i - 1}"]
                if i == backbone_max_level + 1 and self.use_p5
                else output_features[f"P{i - 1}"]
            )
            if i > backbone_max_level + 1:
                feats_in = self.activation(feats_in)
            output_features[level] = (
                self.output_batch_norms[level](
                    self.output_conv_layers[level](feats_in)
                )
                if self.use_batch_norm
                else self.output_conv_layers[level](feats_in)
            )
        output_features = {
            f"P{i}": output_features[f"P{i}"]
            for i in range(self.min_level, self.max_level + 1)
        }
        return output_features

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "min_level": self.min_level,
                "max_level": self.max_level,
                "num_filters": self.num_filters,
                "use_p5": self.use_p5,
                "use_batch_norm": self.use_batch_norm,
                "data_format": self.data_format,
                "activation": keras.activations.serialize(self.activation),
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "batch_norm_momentum": self.batch_norm_momentum,
                "batch_norm_epsilon": self.batch_norm_epsilon,
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

    def compute_output_shape(self, input_shapes):
        output_shape = {}
        input_levels = [int(level[1]) for level in input_shapes]
        backbone_max_level = min(max(input_levels), self.max_level)

        for i in range(self.min_level, backbone_max_level + 1):
            level = f"P{i}"
            if self.data_format == "channels_last":
                output_shape[level] = input_shapes[level][:-1] + (
                    self.num_filters,
                )
            else:
                output_shape[level] = (
                    input_shapes[level][0],
                    self.num_filters,
                ) + input_shapes[level][1:3]

        intermediate_shape = input_shapes[f"P{backbone_max_level}"]
        intermediate_shape = (
            (
                intermediate_shape[0],
                (
                    int(math.ceil(intermediate_shape[1] / 2))
                    if intermediate_shape[1] is not None
                    else None
                ),
                (
                    int(math.ceil(intermediate_shape[1] / 2))
                    if intermediate_shape[1] is not None
                    else None
                ),
                self.num_filters,
            )
            if self.data_format == "channels_last"
            else (
                intermediate_shape[0],
                self.num_filters,
                (
                    int(math.ceil(intermediate_shape[1] / 2))
                    if intermediate_shape[1] is not None
                    else None
                ),
                (
                    int(math.ceil(intermediate_shape[1] / 2))
                    if intermediate_shape[1] is not None
                    else None
                ),
            )
        )

        for i in range(backbone_max_level + 1, self.max_level + 1):
            level = f"P{i}"
            output_shape[level] = intermediate_shape
            intermediate_shape = (
                (
                    intermediate_shape[0],
                    (
                        int(math.ceil(intermediate_shape[1] / 2))
                        if intermediate_shape[1] is not None
                        else None
                    ),
                    (
                        int(math.ceil(intermediate_shape[1] / 2))
                        if intermediate_shape[1] is not None
                        else None
                    ),
                    self.num_filters,
                )
                if self.data_format == "channels_last"
                else (
                    intermediate_shape[0],
                    self.num_filters,
                    (
                        int(math.ceil(intermediate_shape[1] / 2))
                        if intermediate_shape[1] is not None
                        else None
                    ),
                    (
                        int(math.ceil(intermediate_shape[1] / 2))
                        if intermediate_shape[1] is not None
                        else None
                    ),
                )
            )

        return output_shape
