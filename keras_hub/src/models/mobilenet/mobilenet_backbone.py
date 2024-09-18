# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone

BN_EPSILON = 1e-3
BN_MOMENTUM = 0.999


@keras_hub_export("keras_hub.models.MobileNetBackbone")
class MobileNetBackbone(Backbone):
    """Instantiates the MobileNet architecture.

    MobileNet is a lightweight convolutional neural network (CNN)
    optimized for mobile and edge devices, striking a balance between
    accuracy and efficiency. By employing depthwise separable convolutions
    and techniques like Squeeze-and-Excitation (SE) blocks,
    MobileNet models are highly suitable for real-time applications on
    resource-constrained devices.

    References:
        - [MobileNets: Efficient Convolutional Neural Networks
       for Mobile Vision Applications](
        https://arxiv.org/abs/1704.04861)
        - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](
        https://arxiv.org/abs/1801.04381) (CVPR 2018)
        - [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
        (ICCV 2019)

    Args:
        stackwise_expansion: list of ints or floats, the expansion ratio for
            each inverted residual block in the model.
        stackwise_num_filters: list of ints, number of filters for each inverted
            residual block in the model.
        stackwise_kernel_size: list of ints, kernel size for each inverted
            residual block in the model.
        stackwise_num_strides: list of ints, stride length for each inverted
            residual block in the model.
        stackwise_se_ratio: se ratio for each inverted residual block in the
            model. 0 if dont want to add Squeeze and Excite layer.
        stackwise_activation: list of activation functions, for each inverted
             residual block in the model.
        include_rescaling: bool, whether to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(scale=1 / 255)`
            layer.
        image_shape: optional shape tuple, defaults to (224, 224, 3).
        depth_multiplier: float, controls the width of the network.
            - If `depth_multiplier` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `depth_multiplier` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `depth_multiplier` = 1, default number of filters from the paper
                are used at each layer.
        input_num_filters: number of filters in first convolution layer
        output_num_filters: specifies whether to add conv and batch_norm in the end,
            if set to None, it will not add these layers in the end.
            'None' for MobileNetV1
        input_activation: activation function to be used in the input layer
            'hard_swish' for MobileNetV3,
            'relu6' for MobileNetV1 and MobileNetV2
        output_activation: activation function to be used in the output layer
            'hard_swish' for MobileNetV3,
            'relu6' for MobileNetV1 and MobileNetV2
        inverted_res_block: whether to use inverted residual blocks or not,
            'False' for MobileNetV1,
            'True' for MobileNetV2 and MobileNetV3


    Example:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone with a custom config
    model = MobileNetBackbone(
        stackwise_expansion=[1, 4, 6],
        stackwise_num_filters=[4, 8, 16],
        stackwise_kernel_size=[3, 3, 5],
        stackwise_num_strides=[2, 2, 1],
        stackwise_se_ratio=[0.25, None, 0.25],
        stackwise_activation=["relu", "relu6", "hard_swish"],
        include_rescaling=False,
        output_num_filters=1280,
        input_activation='hard_swish',
        output_activation='hard_swish',
        inverted_res_block=True,

    )
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        stackwise_expansion,
        stackwise_num_filters,
        stackwise_kernel_size,
        stackwise_num_strides,
        stackwise_se_ratio,
        stackwise_activation,
        include_rescaling,
        output_num_filters,
        inverted_res_block,
        image_shape=(224, 224, 3),
        input_activation="hard_swish",
        output_activation="hard_swish",
        depth_multiplier=1.0,
        input_num_filters=16,
        **kwargs,
    ):
        # === Functional Model ===
        channel_axis = (
            -1 if keras.config.image_data_format() == "channels_last" else 1
        )

        inputs = keras.layers.Input(shape=image_shape)
        x = inputs

        if include_rescaling:
            x = keras.layers.Rescaling(scale=1 / 255)(x)

        input_num_filters = adjust_channels(input_num_filters)
        x = keras.layers.Conv2D(
            input_num_filters,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            data_format=keras.config.image_data_format(),
            use_bias=False,
            name="input_conv",
        )(x)
        x = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name="input_batch_norm",
        )(x)
        x = keras.layers.Activation(input_activation)(x)

        for stack_index in range(len(stackwise_num_filters)):
            filters = adjust_channels(
                (stackwise_num_filters[stack_index]) * depth_multiplier
            )

            if inverted_res_block:
                x = apply_inverted_res_block(
                    x,
                    expansion=stackwise_expansion[stack_index],
                    filters=filters,
                    kernel_size=stackwise_kernel_size[stack_index],
                    stride=stackwise_num_strides[stack_index],
                    se_ratio=(stackwise_se_ratio[stack_index]),
                    activation=stackwise_activation[stack_index],
                    expansion_index=stack_index,
                )
            else:
                x = apply_depthwise_conv_block(
                    x,
                    filters=filters,
                    kernel_size=3,
                    stride=stackwise_num_strides[stack_index],
                    depth_multiplier=depth_multiplier,
                    block_id=stack_index,
                )

        if output_num_filters is not None:
            last_conv_ch = adjust_channels(x.shape[channel_axis] * 6)

            x = keras.layers.Conv2D(
                last_conv_ch,
                kernel_size=1,
                padding="same",
                data_format=keras.config.image_data_format(),
                use_bias=False,
                name="output_conv",
            )(x)
            x = keras.layers.BatchNormalization(
                axis=channel_axis,
                epsilon=BN_EPSILON,
                momentum=BN_MOMENTUM,
                name="output_batch_norm",
            )(x)
            x = keras.layers.Activation(output_activation)(x)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # === Config ===
        self.stackwise_expansion = stackwise_expansion
        self.stackwise_num_filters = stackwise_num_filters
        self.stackwise_kernel_size = stackwise_kernel_size
        self.stackwise_num_strides = stackwise_num_strides
        self.stackwise_se_ratio = stackwise_se_ratio
        self.stackwise_activation = stackwise_activation
        self.include_rescaling = include_rescaling
        self.depth_multiplier = depth_multiplier
        self.input_num_filters = input_num_filters
        self.output_num_filters = output_num_filters
        self.input_activation = keras.activations.get(input_activation)
        self.output_activation = keras.activations.get(output_activation)
        self.inverted_res_block = inverted_res_block
        self.image_shape = image_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_expansion": self.stackwise_expansion,
                "stackwise_num_filters": self.stackwise_num_filters,
                "stackwise_kernel_size": self.stackwise_kernel_size,
                "stackwise_num_strides": self.stackwise_num_strides,
                "stackwise_se_ratio": self.stackwise_se_ratio,
                "stackwise_activation": self.stackwise_activation,
                "include_rescaling": self.include_rescaling,
                "image_shape": self.image_shape,
                "depth_multiplier": self.depth_multiplier,
                "input_num_filters": self.input_num_filters,
                "output_num_filters": self.output_num_filters,
                "input_activation": keras.activations.serialize(
                    activation=self.input_activation
                ),
                "output_activation": keras.activations.serialize(
                    activation=self.output_activation
                ),
                "inverted_res_block": self.inverted_res_block,
            }
        )
        return config


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
    channel_axis = (
        -1 if keras.config.image_data_format() == "channels_last" else 1
    )
    activation = keras.activations.get(activation)
    shortcut = x
    prefix = "expanded_conv_"
    infilters = x.shape[channel_axis]

    if expansion_index > 0:
        prefix = f"expanded_conv_{expansion_index}_"

        x = keras.layers.Conv2D(
            adjust_channels(infilters * expansion),
            kernel_size=1,
            padding="same",
            data_format=keras.config.image_data_format(),
            use_bias=False,
            name=prefix + "expand",
        )(x)
        x = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=prefix + "expand_BatchNorm",
        )(x)
        x = keras.layers.Activation(activation=activation)(x)

    if stride == 2:
        x = keras.layers.ZeroPadding2D(
            padding=correct_pad_downsample(x, kernel_size),
            name=prefix + "depthwise_pad",
        )(x)

    x = keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding="same" if stride == 1 else "valid",
        data_format=keras.config.image_data_format(),
        use_bias=False,
        name=prefix + "depthwise",
    )(x)
    x = keras.layers.BatchNormalization(
        axis=channel_axis,
        epsilon=BN_EPSILON,
        momentum=BN_MOMENTUM,
        name=prefix + "depthwise_BatchNorm",
    )(x)
    x = keras.layers.Activation(activation=activation)(x)

    if se_ratio:
        se_filters = adjust_channels(infilters * expansion)
        x = SqueezeAndExcite2D(
            input=x,
            filters=se_filters,
            bottleneck_filters=adjust_channels(se_filters * se_ratio),
            squeeze_activation="relu",
            excite_activation=keras.activations.hard_sigmoid,
        )

    x = keras.layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        data_format=keras.config.image_data_format(),
        use_bias=False,
        name=prefix + "project",
    )(x)
    x = keras.layers.BatchNormalization(
        axis=channel_axis,
        epsilon=BN_EPSILON,
        momentum=BN_MOMENTUM,
        name=prefix + "project_BatchNorm",
    )(x)

    if stride == 1 and infilters == filters:
        x = keras.layers.Add(name=prefix + "Add")([shortcut, x])

    return x


def apply_depthwise_conv_block(
    x,
    filters,
    kernel_size=3,
    depth_multiplier=1,
    stride=1,
    block_id=1,
):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    Args:
        x: Input tensor of shape `(rows, cols, channels)
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        depth_multiplier: controls the width of the network.
            - If `depth_multiplier` < 1.0, proportionally decreases the number
                 of filters in each layer.
            - If `depth_multiplier` > 1.0, proportionally increases the number
              of filters in each layer.
            - If `depth_multiplier` = 1, default number of filters from the
                paper are used at each layer.
        strides: An integer or tuple/list of 2 integers, specifying the strides
            of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions. Specifying any stride value != 1 is
            incompatible with specifying any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating the block number.

    Input shape:
        4D tensor with shape: `(batch, rows, cols, channels)` in "channels_last"
        4D tensor with shape: `(batch, channels, rows, cols)` in "channels_first"
    Returns:
        Output tensor of block.
    """
    channel_axis = (
        -1 if keras.config.image_data_format() == "channels_last" else 1
    )
    if stride == 2:
        x = keras.layers.ZeroPadding2D(
            padding=correct_pad_downsample(x, kernel_size),
            name="conv_pad_%d" % block_id,
        )(x)

    x = keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding="same" if stride == 1 else "valid",
        data_format=keras.config.image_data_format(),
        depth_multiplier=depth_multiplier,
        use_bias=False,
        name="depthwise_%d" % block_id,
    )(x)
    x = keras.layers.BatchNormalization(
        axis=channel_axis,
        epsilon=BN_EPSILON,
        momentum=BN_MOMENTUM,
        name="depthwise_BatchNorm_%d" % block_id,
    )(x)
    x = keras.layers.ReLU(6.0)(x)

    x = keras.layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        data_format=keras.config.image_data_format(),
        use_bias=False,
        name="conv_%d" % block_id,
    )(x)
    x = keras.layers.BatchNormalization(
        axis=channel_axis,
        epsilon=BN_EPSILON,
        momentum=BN_MOMENTUM,
        name="BatchNorm_%d" % block_id,
    )(x)
    return keras.layers.ReLU(6.0)(x)


def SqueezeAndExcite2D(
    input,
    filters,
    bottleneck_filters=None,
    squeeze_activation="relu",
    excite_activation="sigmoid",
):
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
    """
    if not bottleneck_filters:
        bottleneck_filters = filters // 4

    x = keras.layers.GlobalAveragePooling2D(keepdims=True)(input)

    x = keras.layers.Conv2D(
        bottleneck_filters,
        (1, 1),
        data_format=keras.config.image_data_format(),
        activation=squeeze_activation,
    )(x)
    x = keras.layers.Conv2D(
        filters,
        (1, 1),
        data_format=keras.config.image_data_format(),
        activation=excite_activation,
    )(x)

    x = ops.multiply(x, input)
    return x


def correct_pad_downsample(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Args:
        inputs: Input tensor.
        kernel_size: An integer or tuple/list of 2 integers.

    Returns:
        A tuple.
    """
    img_dim = 1
    input_size = inputs.shape[img_dim : (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )
