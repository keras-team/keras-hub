import keras
from keras import ops
from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.backbone import Backbone


CHANNEL_AXIS = -1
BN_EPSILON = 1e-3
BN_MOMENTUM = 0.999


@keras_nlp_export("keras_nlp.models.MobileNetV3Backbone")
class MobileNetV3Backbone(Backbone):
    """Instantiates the MobileNetV3 architecture.

    References:
        - [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
        (ICCV 2019)
        - [Based on the Original keras.applications MobileNetv3](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        stackwise_expansion: list of ints or floats, the expansion ratio for
            each inverted residual block in the model.
        stackwise_filters: list of ints, number of filters for each inverted
            residual block in the model.
        stackwise_stride: list of ints, stride length for each inverted
            residual block in the model.
        include_rescaling: bool, whether to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(scale=1 / 255)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        alpha: float, controls the width of the network. This is known as the
            depth multiplier in the MobileNetV3 paper, but the name is kept for
            consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                are used at each layer.

    Example:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone with a custom config
    model = MobileNetV3Backbone(
        stackwise_expansion=[1, 72.0 / 16, 88.0 / 24, 4, 6, 6, 3, 3, 6, 6, 6],
        stackwise_filters=[16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96],
        stackwise_kernel_size=[3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5],
        stackwise_stride=[2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1],
        stackwise_se_ratio=[0.25, None, None, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        stackwise_activation=["relu", "relu", "relu", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish", "hard_swish"],
        include_rescaling=False,
    )
    output = model(input_data)
    ```
    """  # noqa: E501

    def __init__(
        self,
        *,
        stackwise_expansion,
        stackwise_filters,
        stackwise_kernel_size,
        stackwise_stride,
        stackwise_se_ratio,
        stackwise_activation,
        include_rescaling,
        input_shape=(224, 224, 3),
        alpha=1.0,
        **kwargs,
    ):
        inputs = keras.layers.Input(shape=input_shape)
        x = inputs

        if include_rescaling:
            x = keras.layers.Rescaling(scale=1 / 255)(x)

        x = keras.layers.Conv2D(
            16,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            name="Conv",
        )(x)
        x = keras.layers.BatchNormalization(
            axis=CHANNEL_AXIS,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name="Conv_BatchNorm",
        )(x)
        x = apply_hard_swish(x)

        for stack_index in range(len(stackwise_filters)):

            x = apply_inverted_res_block(
                x,
                expansion=stackwise_expansion[stack_index],
                filters=adjust_channels(
                    (stackwise_filters[stack_index]) * alpha
                ),
                kernel_size=stackwise_kernel_size[stack_index],
                stride=stackwise_stride[stack_index],
                se_ratio=stackwise_se_ratio[stack_index],
                activation=stackwise_activation[stack_index],
                expansion_index=stack_index,
            )

        last_conv_ch = adjust_channels(x.shape[CHANNEL_AXIS] * 6)

        x = keras.layers.Conv2D(
            last_conv_ch,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="Conv_1",
        )(x)
        x = keras.layers.BatchNormalization(
            axis=CHANNEL_AXIS,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name="Conv_1_BatchNorm",
        )(x)
        x = apply_hard_swish(x)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.stackwise_expansion = stackwise_expansion
        self.stackwise_filters = stackwise_filters
        self.stackwise_kernel_size = stackwise_kernel_size
        self.stackwise_stride = stackwise_stride
        self.stackwise_se_ratio = stackwise_se_ratio
        self.stackwise_activation = stackwise_activation
        self.include_rescaling = include_rescaling
        self.alpha = alpha

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_expansion": self.stackwise_expansion,
                "stackwise_filters": self.stackwise_filters,
                "stackwise_kernel_size": self.stackwise_kernel_size,
                "stackwise_stride": self.stackwise_stride,
                "stackwise_se_ratio": self.stackwise_se_ratio,
                "stackwise_activation": self.stackwise_activation,
                "include_rescaling": self.include_rescaling,
                "input_shape": self.input_shape[1:],
                "alpha": self.alpha,
            }
        )
        return config


class HardSigmoidActivation(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return apply_hard_sigmoid(x)

    def get_config(self):
        return super().get_config()


def adjust_channels(x, divisor=8, min_value=None):
    """Ensure that all layers have a channel number divisible by the `divisor`.

    Args:
        x: integer, input value.
        divisor: integer, the value by which a channel number should be
            divisible, defaults to 8.
        min_value: float, optional minimum value for the new tensor. If None,
            defaults to value of divisor.

    Returns:
        the updated input scalar.
    """

    if min_value is None:
        min_value = divisor

    new_x = max(min_value, int(x + divisor / 2) // divisor * divisor)

    # make sure that round down does not go down by more than 10%.
    if new_x < 0.9 * x:
        new_x += divisor
    return new_x


def apply_hard_sigmoid(x):
    activation = keras.layers.ReLU(6.0)
    return activation(x + 3.0) * (1.0 / 6.0)


def apply_hard_swish(x):
    return keras.layers.Multiply()([x, apply_hard_sigmoid(x)])


def apply_inverted_res_block(
    x,
    expansion,
    filters,
    kernel_size,
    stride,
    se_ratio,
    activation,
    expansion_index,
):
    """An Inverted Residual Block.

    Args:
        x: input tensor.
        expansion: integer, the expansion ratio, multiplied with infilters to
            get the minimum value passed to adjust_channels.
        filters: integer, number of filters for convolution layer.
        kernel_size: integer, the kernel size for DepthWise Convolutions.
        stride: integer, the stride length for DepthWise Convolutions.
        se_ratio: float, ratio for bottleneck filters. Number of bottleneck
            filters = filters * se_ratio.
        activation: the activation layer to use.
        expansion_index: integer, a unique identification if you want to use
            expanded convolutions. If greater than 0, an additional Conv+BN
            layer is added after the expanded convolutional layer.

    Returns:
        the updated input tensor.
    """
    if isinstance(activation, str):
        if activation == "hard_swish":
            activation = apply_hard_swish
        else:
            activation = keras.activations.get(activation)

    shortcut = x
    prefix = "expanded_conv_"
    infilters = x.shape[CHANNEL_AXIS]

    if expansion_index > 0:
        prefix = f"expanded_conv_{expansion_index}_"

        x = keras.layers.Conv2D(
            adjust_channels(infilters * expansion),
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand",
        )(x)
        x = keras.layers.BatchNormalization(
            axis=CHANNEL_AXIS,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=prefix + "expand_BatchNorm",
        )(x)
        x = activation(x)

    if stride == 2:
        x = keras.layers.ZeroPadding2D(
            padding=utils.correct_pad_downsample(x, kernel_size),
            name=prefix + "depthwise_pad",
        )(x)

    x = keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding="same" if stride == 1 else "valid",
        use_bias=False,
        name=prefix + "depthwise",
    )(x)
    x = keras.layers.BatchNormalization(
        axis=CHANNEL_AXIS,
        epsilon=BN_EPSILON,
        momentum=BN_MOMENTUM,
        name=prefix + "depthwise_BatchNorm",
    )(x)
    x = activation(x)

    if se_ratio:
        se_filters = adjust_channels(infilters * expansion)
        x = SqueezeAndExcite2D(
            x,
            se_filters,
            adjust_channels(se_filters * se_ratio),
            "relu",
            HardSigmoidActivation(),
        )

    x = keras.layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=prefix + "project",
    )(x)
    x = keras.layers.BatchNormalization(
        axis=CHANNEL_AXIS,
        epsilon=BN_EPSILON,
        momentum=BN_MOMENTUM,
        name=prefix + "project_BatchNorm",
    )(x)

    if stride == 1 and infilters == filters:
        x = keras.layers.Add(name=prefix + "Add")([shortcut, x])

    return x


def SqueezeAndExcite2D(
    input,
    filters,
    bottleneck_filters=None,
    squeeze_activation="relu",
    excite_activation="sigmoid",
):
    """
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
    Example:

    ```python
    # (...)
    input = tf.ones((1, 5, 5, 16), dtype=tf.float32)
    x = keras.layers.Conv2D(16, (3, 3))(input)

    # (...)
    ```
    """
    if not bottleneck_filters:
        bottleneck_filters = filters // 4

    x = keras.layers.GlobalAveragePooling2D(keepdims=True)(input)
    x = keras.layers.Conv2D(
        bottleneck_filters,
        (1, 1),
        activation=self.squeeze_activation,
    )(x)
    x = keras.layers.Conv2D(
        self.filters, (1, 1), activation=self.excite_activation
    )(x)

    x = ops.multiply(x, input)
    return x
