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
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.feature_pyramid_backbone import FeaturePyramidBackbone
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.ResNetBackbone")
class ResNetBackbone(FeaturePyramidBackbone):
    """ResNet and ResNetV2 core network with hyperparameters.

    This class implements a ResNet backbone as described in [Deep Residual
    Learning for Image Recognition](https://arxiv.org/abs/1512.03385)(
    CVPR 2016), [Identity Mappings in Deep Residual Networks](
    https://arxiv.org/abs/1603.05027)(ECCV 2016), [ResNet strikes back: An
    improved training procedure in timm](https://arxiv.org/abs/2110.00476)(
    NeurIPS 2021 Workshop) and [Bag of Tricks for Image Classification with
    Convolutional Neural Networks](https://arxiv.org/abs/1812.01187).

    The difference in ResNet and ResNetV2 rests in the structure of their
    individual building blocks. In ResNetV2, the batch normalization and
    ReLU activation precede the convolution layers, as opposed to ResNet where
    the batch normalization and ReLU activation are applied after the
    convolution layers.

    ResNetVd introduces two key modifications to the standard ResNet. First,
    the initial convolutional layer is replaced by a series of three
    successive convolutional layers. Second, shortcut connections use an
    additional pooling operation rather than performing downsampling within
    the convolutional layers themselves.

    Note that `ResNetBackbone` expects the inputs to be images with a value
    range of `[0, 255]` when `include_rescaling=True`.

    Args:
        input_conv_filters: list of ints. The number of filters of the initial
            convolution(s).
        input_conv_kernel_sizes: list of ints. The kernel sizes of the initial
            convolution(s).
        stackwise_num_filters: list of ints. The number of filters for each
            stack.
        stackwise_num_blocks: list of ints. The number of blocks for each stack.
        stackwise_num_strides: list of ints. The number of strides for each
            stack.
        block_type: str. The block type to stack. One of `"basic_block"`,
            `"bottleneck_block"`, `"basic_block_vd"` or
            `"bottleneck_block_vd"`. Use `"basic_block"` for ResNet18 and
            ResNet34. Use `"bottleneck_block"` for ResNet50, ResNet101 and
            ResNet152 and the `"_vd"` prefix for the respective ResNet_vd
            variants.
        use_pre_activation: boolean. Whether to use pre-activation or not.
            `True` for ResNetV2, `False` for ResNet.
        include_rescaling: boolean. If `True`, rescale the input using
            `Rescaling` and `Normalization` layers. If `False`, do nothing.
            Defaults to `True`.
        image_shape: tuple. The input shape without the batch size.
            Defaults to `(None, None, 3)`.
        pooling: `None` or str. Pooling mode for feature extraction. Defaults
            to `"avg"`.
            - `None` means that the output of the model will be the 4D tensor
                from the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, resulting in a 2D
                tensor.
            - `max` means that global max pooling will be applied to the
                output of the last convolutional block, resulting in a 2D
                tensor.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Examples:
    ```python
    input_data = np.random.uniform(0, 255, size=(2, 224, 224, 3))

    # Pretrained ResNet backbone.
    model = keras_hub.models.ResNetBackbone.from_preset("resnet50")
    model(input_data)

    # Randomly initialized ResNetV2 backbone with a custom config.
    model = keras_hub.models.ResNetBackbone(
        input_conv_filters=[64],
        input_conv_kernel_sizes=[7],
        stackwise_num_filters=[64, 64, 64],
        stackwise_num_blocks=[2, 2, 2],
        stackwise_num_strides=[1, 2, 2],
        block_type="basic_block",
        use_pre_activation=True,
        pooling="avg",
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        input_conv_filters,
        input_conv_kernel_sizes,
        stackwise_num_filters,
        stackwise_num_blocks,
        stackwise_num_strides,
        block_type,
        use_pre_activation=False,
        include_rescaling=True,
        image_shape=(None, None, 3),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        if len(input_conv_filters) != len(input_conv_kernel_sizes):
            raise ValueError(
                "The length of `input_conv_filters` and"
                "`input_conv_kernel_sizes` must be the same. "
                f"Received: input_conv_filters={input_conv_filters}, "
                f"input_conv_kernel_sizes={input_conv_kernel_sizes}."
            )
        if len(stackwise_num_filters) != len(stackwise_num_blocks) or len(
            stackwise_num_filters
        ) != len(stackwise_num_strides):
            raise ValueError(
                "The length of `stackwise_num_filters`, `stackwise_num_blocks` "
                "and `stackwise_num_strides` must be the same. Received: "
                f"stackwise_num_filters={stackwise_num_filters}, "
                f"stackwise_num_blocks={stackwise_num_blocks}, "
                f"stackwise_num_strides={stackwise_num_strides}"
            )
        if stackwise_num_filters[0] != 64:
            raise ValueError(
                "The first element of `stackwise_num_filters` must be 64. "
                f"Received: stackwise_num_filters={stackwise_num_filters}"
            )
        if block_type not in (
            "basic_block",
            "bottleneck_block",
            "basic_block_vd",
            "bottleneck_block_vd",
        ):
            raise ValueError(
                '`block_type` must be either `"basic_block"`, '
                '`"bottleneck_block"`, `"basic_block_vd"` or '
                f'`"bottleneck_block_vd"`. Received block_type={block_type}.'
            )
        data_format = standardize_data_format(data_format)
        bn_axis = -1 if data_format == "channels_last" else 1
        num_input_convs = len(input_conv_filters)
        num_stacks = len(stackwise_num_filters)

        # === Functional Model ===
        image_input = layers.Input(shape=image_shape)
        if include_rescaling:
            x = layers.Rescaling(scale=1 / 255.0, dtype=dtype)(image_input)
            x = layers.Normalization(
                axis=bn_axis,
                mean=(0.485, 0.456, 0.406),
                variance=(0.229**2, 0.224**2, 0.225**2),
                dtype=dtype,
                name="normalization",
            )(x)
        else:
            x = image_input

        # The padding between torch and tensorflow/jax differs when `strides>1`.
        # Therefore, we need to manually pad the tensor.
        x = layers.ZeroPadding2D(
            (input_conv_kernel_sizes[0] - 1) // 2,
            data_format=data_format,
            dtype=dtype,
            name="conv1_pad",
        )(x)
        x = layers.Conv2D(
            input_conv_filters[0],
            input_conv_kernel_sizes[0],
            strides=2,
            data_format=data_format,
            use_bias=False,
            padding="valid",
            dtype=dtype,
            name="conv1_conv",
        )(x)
        for conv_index in range(1, num_input_convs):
            x = layers.BatchNormalization(
                axis=bn_axis,
                epsilon=1e-5,
                momentum=0.9,
                dtype=dtype,
                name=f"conv{conv_index}_bn",
            )(x)
            x = layers.Activation(
                "relu", dtype=dtype, name=f"conv{conv_index}_relu"
            )(x)
            x = layers.Conv2D(
                input_conv_filters[conv_index],
                input_conv_kernel_sizes[conv_index],
                strides=1,
                data_format=data_format,
                use_bias=False,
                padding="same",
                dtype=dtype,
                name=f"conv{conv_index+1}_conv",
            )(x)

        if not use_pre_activation:
            x = layers.BatchNormalization(
                axis=bn_axis,
                epsilon=1e-5,
                momentum=0.9,
                dtype=dtype,
                name=f"conv{num_input_convs}_bn",
            )(x)
            x = layers.Activation(
                "relu",
                dtype=dtype,
                name=f"conv{num_input_convs}_relu",
            )(x)

        if use_pre_activation:
            # A workaround for ResNetV2: we need -inf padding to prevent zeros
            # from being the max values in the following `MaxPooling2D`.
            pad_width = [[1, 1], [1, 1]]
            if data_format == "channels_last":
                pad_width += [[0, 0]]
            else:
                pad_width = [[0, 0]] + pad_width
            pad_width = [[0, 0]] + pad_width
            x = ops.pad(x, pad_width=pad_width, constant_values=float("-inf"))
        else:
            x = layers.ZeroPadding2D(
                1, data_format=data_format, dtype=dtype, name="pool1_pad"
            )(x)
        x = layers.MaxPooling2D(
            3,
            strides=2,
            data_format=data_format,
            dtype=dtype,
            name="pool1_pool",
        )(x)

        pyramid_outputs = {}
        for stack_index in range(num_stacks):
            x = apply_stack(
                x,
                filters=stackwise_num_filters[stack_index],
                blocks=stackwise_num_blocks[stack_index],
                stride=stackwise_num_strides[stack_index],
                block_type=block_type,
                use_pre_activation=use_pre_activation,
                first_shortcut=(block_type != "basic_block" or stack_index > 0),
                data_format=data_format,
                dtype=dtype,
                name=f"stack{stack_index}",
            )
            pyramid_outputs[f"P{stack_index + 2}"] = x

        if use_pre_activation:
            x = layers.BatchNormalization(
                axis=bn_axis,
                epsilon=1e-5,
                momentum=0.9,
                dtype=dtype,
                name="post_bn",
            )(x)
            x = layers.Activation("relu", dtype=dtype, name="post_relu")(x)

        super().__init__(
            inputs=image_input,
            outputs=x,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.input_conv_filters = input_conv_filters
        self.input_conv_kernel_sizes = input_conv_kernel_sizes
        self.stackwise_num_filters = stackwise_num_filters
        self.stackwise_num_blocks = stackwise_num_blocks
        self.stackwise_num_strides = stackwise_num_strides
        self.block_type = block_type
        self.use_pre_activation = use_pre_activation
        self.include_rescaling = include_rescaling
        self.image_shape = image_shape
        self.pyramid_outputs = pyramid_outputs
        self.data_format = data_format

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_conv_filters": self.input_conv_filters,
                "input_conv_kernel_sizes": self.input_conv_kernel_sizes,
                "stackwise_num_filters": self.stackwise_num_filters,
                "stackwise_num_blocks": self.stackwise_num_blocks,
                "stackwise_num_strides": self.stackwise_num_strides,
                "block_type": self.block_type,
                "use_pre_activation": self.use_pre_activation,
                "include_rescaling": self.include_rescaling,
                "image_shape": self.image_shape,
            }
        )
        return config


def apply_basic_block(
    x,
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=False,
    use_pre_activation=False,
    data_format=None,
    dtype=None,
    name=None,
):
    """Applies a basic residual block.

    Args:
        x: Tensor. The input tensor to pass through the block.
        filters: int. The number of filters in the block.
        kernel_size: int. The kernel size of the bottleneck layer. Defaults to
            `3`.
        stride: int. The stride length of the first layer. Defaults to `1`.
        conv_shortcut: bool. If `True`, use a convolution shortcut. If `False`,
            use an identity or pooling shortcut based on the stride. Defaults to
            `False`.
        use_pre_activation: boolean. Whether to use pre-activation or not.
            `True` for ResNetV2, `False` for ResNet. Defaults to `False`.
        data_format: `None` or str. the ordering of the dimensions in the
            inputs. Can be `"channels_last"`
             (`(batch_size, height, width, channels)`) or`"channels_first"`
            (`(batch_size, channels, height, width)`).
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the models computations and weights.
        name: str. A prefix for the layer names used in the block.

    Returns:
        The output tensor for the basic residual block.
    """
    data_format = data_format or keras.config.image_data_format()
    bn_axis = -1 if data_format == "channels_last" else 1

    x_preact = None
    if use_pre_activation:
        x_preact = layers.BatchNormalization(
            axis=bn_axis,
            epsilon=1e-5,
            momentum=0.9,
            dtype=dtype,
            name=f"{name}_pre_activation_bn",
        )(x)
        x_preact = layers.Activation(
            "relu", dtype=dtype, name=f"{name}_pre_activation_relu"
        )(x_preact)

    if conv_shortcut:
        if x_preact is not None:
            shortcut = x_preact
        else:
            shortcut = x
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            data_format=data_format,
            use_bias=False,
            dtype=dtype,
            name=f"{name}_0_conv",
        )(shortcut)
        if not use_pre_activation:
            shortcut = layers.BatchNormalization(
                axis=bn_axis,
                epsilon=1e-5,
                momentum=0.9,
                dtype=dtype,
                name=f"{name}_0_bn",
            )(shortcut)
    else:
        shortcut = x

    x = x_preact if x_preact is not None else x
    if stride > 1:
        x = layers.ZeroPadding2D(
            (kernel_size - 1) // 2,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_1_pad",
        )(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="valid" if stride > 1 else "same",
        data_format=data_format,
        use_bias=False,
        dtype=dtype,
        name=f"{name}_1_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_1_bn",
    )(x)
    x = layers.Activation("relu", dtype=dtype, name=f"{name}_1_relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=1,
        padding="same",
        data_format=data_format,
        use_bias=False,
        dtype=dtype,
        name=f"{name}_2_conv",
    )(x)
    if not use_pre_activation:
        x = layers.BatchNormalization(
            axis=bn_axis,
            epsilon=1e-5,
            momentum=0.9,
            dtype=dtype,
            name=f"{name}_2_bn",
        )(x)
        x = layers.Add(dtype=dtype, name=f"{name}_add")([shortcut, x])
        x = layers.Activation("relu", dtype=dtype, name=f"{name}_out")(x)
    else:
        x = layers.Add(dtype=dtype, name=f"{name}_out")([shortcut, x])
    return x


def apply_bottleneck_block(
    x,
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=False,
    use_pre_activation=False,
    data_format=None,
    dtype=None,
    name=None,
):
    """Applies a bottleneck residual block.

    Args:
        x: Tensor. The input tensor to pass through the block.
        filters: int. The number of filters in the block.
        kernel_size: int. The kernel size of the bottleneck layer. Defaults to
            `3`.
        stride: int. The stride length of the first layer. Defaults to `1`.
        conv_shortcut: bool. If `True`, use a convolution shortcut. If `False`,
            use an identity or pooling shortcut based on the stride. Defaults to
            `False`.
        use_pre_activation: boolean. Whether to use pre-activation or not.
            `True` for ResNetV2, `False` for ResNet. Defaults to `False`.
        data_format: `None` or str. the ordering of the dimensions in the
            inputs. Can be `"channels_last"`
             (`(batch_size, height, width, channels)`) or`"channels_first"`
            (`(batch_size, channels, height, width)`).
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the models computations and weights.
        name: str. A prefix for the layer names used in the block.

    Returns:
        The output tensor for the residual block.
    """
    data_format = data_format or keras.config.image_data_format()
    bn_axis = -1 if data_format == "channels_last" else 1

    x_preact = None
    if use_pre_activation:
        x_preact = layers.BatchNormalization(
            axis=bn_axis,
            epsilon=1e-5,
            momentum=0.9,
            dtype=dtype,
            name=f"{name}_pre_activation_bn",
        )(x)
        x_preact = layers.Activation(
            "relu", dtype=dtype, name=f"{name}_pre_activation_relu"
        )(x_preact)

    if conv_shortcut:
        if x_preact is not None:
            shortcut = x_preact
        else:
            shortcut = x
        shortcut = layers.Conv2D(
            4 * filters,
            1,
            strides=stride,
            data_format=data_format,
            use_bias=False,
            dtype=dtype,
            name=f"{name}_0_conv",
        )(shortcut)
        if not use_pre_activation:
            shortcut = layers.BatchNormalization(
                axis=bn_axis,
                epsilon=1e-5,
                momentum=0.9,
                dtype=dtype,
                name=f"{name}_0_bn",
            )(shortcut)
    else:
        shortcut = x

    x = x_preact if x_preact is not None else x
    x = layers.Conv2D(
        filters,
        1,
        strides=1,
        data_format=data_format,
        use_bias=False,
        dtype=dtype,
        name=f"{name}_1_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_1_bn",
    )(x)
    x = layers.Activation("relu", dtype=dtype, name=f"{name}_1_relu")(x)

    if stride > 1:
        x = layers.ZeroPadding2D(
            (kernel_size - 1) // 2,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_2_pad",
        )(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="valid" if stride > 1 else "same",
        data_format=data_format,
        use_bias=False,
        dtype=dtype,
        name=f"{name}_2_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_2_bn",
    )(x)
    x = layers.Activation("relu", dtype=dtype, name=f"{name}_2_relu")(x)

    x = layers.Conv2D(
        4 * filters,
        1,
        data_format=data_format,
        use_bias=False,
        dtype=dtype,
        name=f"{name}_3_conv",
    )(x)
    if not use_pre_activation:
        x = layers.BatchNormalization(
            axis=bn_axis,
            epsilon=1e-5,
            momentum=0.9,
            dtype=dtype,
            name=f"{name}_3_bn",
        )(x)
        x = layers.Add(dtype=dtype, name=f"{name}_add")([shortcut, x])
        x = layers.Activation("relu", dtype=dtype, name=f"{name}_out")(x)
    else:
        x = layers.Add(dtype=dtype, name=f"{name}_out")([shortcut, x])
    return x


def apply_basic_block_vd(
    x,
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=False,
    use_pre_activation=False,
    data_format=None,
    dtype=None,
    name=None,
):
    """Applies a basic residual block.

    Args:
        x: Tensor. The input tensor to pass through the block.
        filters: int. The number of filters in the block.
        kernel_size: int. The kernel size of the bottleneck layer. Defaults to
            `3`.
        stride: int. The stride length of the first layer. Defaults to `1`.
        conv_shortcut: bool. If `True`, use a convolution shortcut. If `False`,
            use an identity or pooling shortcut based on the stride. Defaults to
            `False`.
        use_pre_activation: boolean. Whether to use pre-activation or not.
            `True` for ResNetV2, `False` for ResNet. Defaults to `False`.
        data_format: `None` or str. the ordering of the dimensions in the
            inputs. Can be `"channels_last"`
             (`(batch_size, height, width, channels)`) or`"channels_first"`
            (`(batch_size, channels, height, width)`).
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the models computations and weights.
        name: str. A prefix for the layer names used in the block.

    Returns:
        The output tensor for the basic residual block.
    """
    data_format = data_format or keras.config.image_data_format()
    bn_axis = -1 if data_format == "channels_last" else 1

    x_preact = None
    if use_pre_activation:
        x_preact = layers.BatchNormalization(
            axis=bn_axis,
            epsilon=1e-5,
            momentum=0.9,
            dtype=dtype,
            name=f"{name}_pre_activation_bn",
        )(x)
        x_preact = layers.Activation(
            "relu", dtype=dtype, name=f"{name}_pre_activation_relu"
        )(x_preact)

    if conv_shortcut:
        if x_preact is not None:
            shortcut = x_preact
        elif stride > 1:
            shortcut = layers.AveragePooling2D(
                2,
                strides=stride,
                data_format=data_format,
                dtype=dtype,
                padding="same",
            )(x)
        else:
            shortcut = x
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=1,
            data_format=data_format,
            use_bias=False,
            dtype=dtype,
            name=f"{name}_0_conv",
        )(shortcut)
        if not use_pre_activation:
            shortcut = layers.BatchNormalization(
                axis=bn_axis,
                epsilon=1e-5,
                momentum=0.9,
                dtype=dtype,
                name=f"{name}_0_bn",
            )(shortcut)
    else:
        shortcut = x

    x = x_preact if x_preact is not None else x
    if stride > 1:
        x = layers.ZeroPadding2D(
            (kernel_size - 1) // 2,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_1_pad",
        )(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="valid" if stride > 1 else "same",
        data_format=data_format,
        use_bias=False,
        dtype=dtype,
        name=f"{name}_1_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_1_bn",
    )(x)
    x = layers.Activation("relu", dtype=dtype, name=f"{name}_1_relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=1,
        padding="same",
        data_format=data_format,
        use_bias=False,
        dtype=dtype,
        name=f"{name}_2_conv",
    )(x)
    if not use_pre_activation:
        x = layers.BatchNormalization(
            axis=bn_axis,
            epsilon=1e-5,
            momentum=0.9,
            dtype=dtype,
            name=f"{name}_2_bn",
        )(x)
        x = layers.Add(dtype=dtype, name=f"{name}_add")([shortcut, x])
        x = layers.Activation("relu", dtype=dtype, name=f"{name}_out")(x)
    else:
        x = layers.Add(dtype=dtype, name=f"{name}_out")([shortcut, x])
    return x


def apply_bottleneck_block_vd(
    x,
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=False,
    use_pre_activation=False,
    data_format=None,
    dtype=None,
    name=None,
):
    """Applies a bottleneck residual block.

    Args:
        x: Tensor. The input tensor to pass through the block.
        filters: int. The number of filters in the block.
        kernel_size: int. The kernel size of the bottleneck layer. Defaults to
            `3`.
        stride: int. The stride length of the first layer. Defaults to `1`.
        conv_shortcut: bool. If `True`, use a convolution shortcut. If `False`,
            use an identity or pooling shortcut based on the stride. Defaults to
            `False`.
        use_pre_activation: boolean. Whether to use pre-activation or not.
            `True` for ResNetV2, `False` for ResNet. Defaults to `False`.
        data_format: `None` or str. the ordering of the dimensions in the
            inputs. Can be `"channels_last"`
             (`(batch_size, height, width, channels)`) or`"channels_first"`
            (`(batch_size, channels, height, width)`).
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the models computations and weights.
        name: str. A prefix for the layer names used in the block.

    Returns:
        The output tensor for the residual block.
    """
    data_format = data_format or keras.config.image_data_format()
    bn_axis = -1 if data_format == "channels_last" else 1

    x_preact = None
    if use_pre_activation:
        x_preact = layers.BatchNormalization(
            axis=bn_axis,
            epsilon=1e-5,
            momentum=0.9,
            dtype=dtype,
            name=f"{name}_pre_activation_bn",
        )(x)
        x_preact = layers.Activation(
            "relu", dtype=dtype, name=f"{name}_pre_activation_relu"
        )(x_preact)

    if conv_shortcut:
        if x_preact is not None:
            shortcut = x_preact
        elif stride > 1:
            shortcut = layers.AveragePooling2D(
                2,
                strides=stride,
                data_format=data_format,
                dtype=dtype,
                padding="same",
            )(x)
        else:
            shortcut = x
        shortcut = layers.Conv2D(
            4 * filters,
            1,
            strides=1,
            data_format=data_format,
            use_bias=False,
            dtype=dtype,
            name=f"{name}_0_conv",
        )(shortcut)
        if not use_pre_activation:
            shortcut = layers.BatchNormalization(
                axis=bn_axis,
                epsilon=1e-5,
                momentum=0.9,
                dtype=dtype,
                name=f"{name}_0_bn",
            )(shortcut)
    else:
        shortcut = x

    x = x_preact if x_preact is not None else x
    x = layers.Conv2D(
        filters,
        1,
        strides=1,
        data_format=data_format,
        use_bias=False,
        dtype=dtype,
        name=f"{name}_1_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_1_bn",
    )(x)
    x = layers.Activation("relu", dtype=dtype, name=f"{name}_1_relu")(x)

    if stride > 1:
        x = layers.ZeroPadding2D(
            (kernel_size - 1) // 2,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_2_pad",
        )(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="valid" if stride > 1 else "same",
        data_format=data_format,
        use_bias=False,
        dtype=dtype,
        name=f"{name}_2_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_2_bn",
    )(x)
    x = layers.Activation("relu", dtype=dtype, name=f"{name}_2_relu")(x)

    x = layers.Conv2D(
        4 * filters,
        1,
        data_format=data_format,
        use_bias=False,
        dtype=dtype,
        name=f"{name}_3_conv",
    )(x)
    if not use_pre_activation:
        x = layers.BatchNormalization(
            axis=bn_axis,
            epsilon=1e-5,
            momentum=0.9,
            dtype=dtype,
            name=f"{name}_3_bn",
        )(x)
        x = layers.Add(dtype=dtype, name=f"{name}_add")([shortcut, x])
        x = layers.Activation("relu", dtype=dtype, name=f"{name}_out")(x)
    else:
        x = layers.Add(dtype=dtype, name=f"{name}_out")([shortcut, x])
    return x


def apply_stack(
    x,
    filters,
    blocks,
    stride,
    block_type,
    use_pre_activation,
    first_shortcut=True,
    data_format=None,
    dtype=None,
    name=None,
):
    """Applies a set of stacked residual blocks.

    Args:
        x: Tensor. The input tensor to pass through the stack.
        filters: int. The number of filters in a block.
        blocks: int. The number of blocks in the stack.
        stride: int. The stride length of the first layer in the first block.
        block_type: str. The block type to stack. One of `"basic_block"` or
            `"bottleneck_block"`, `"basic_block_vd"` or
            `"bottleneck_block_vd"`. Use `"basic_block"` for ResNet18 and
            ResNet34. Use `"bottleneck_block"` for ResNet50, ResNet101 and
            ResNet152 and the `"_vd"` prefix for the respective ResNet_vd
            variants.
        use_pre_activation: boolean. Whether to use pre-activation or not.
            `True` for ResNetV2, `False` for ResNet and ResNeXt.
        first_shortcut: bool. If `True`, use a convolution shortcut. If `False`,
            use an identity or pooling shortcut based on the stride. Defaults to
            `True`.
        data_format: `None` or str. the ordering of the dimensions in the
            inputs. Can be `"channels_last"`
             (`(batch_size, height, width, channels)`) or`"channels_first"`
            (`(batch_size, channels, height, width)`).
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the models computations and weights.
        name: str. A prefix for the layer names used in the stack.

    Returns:
        Output tensor for the stacked blocks.
    """
    if name is None:
        name = "stack"

    if block_type == "basic_block":
        block_fn = apply_basic_block
    elif block_type == "bottleneck_block":
        block_fn = apply_bottleneck_block
    elif block_type == "basic_block_vd":
        block_fn = apply_basic_block_vd
    elif block_type == "bottleneck_block_vd":
        block_fn = apply_bottleneck_block_vd
    else:
        raise ValueError(
            '`block_type` must be either `"basic_block"`, '
            '`"bottleneck_block"`, `"basic_block_vd"` or '
            f'`"bottleneck_block_vd"`. Received block_type={block_type}.'
        )
    for i in range(blocks):
        if i == 0:
            stride = stride
            conv_shortcut = first_shortcut
        else:
            stride = 1
            conv_shortcut = False
        x = block_fn(
            x,
            filters,
            stride=stride,
            conv_shortcut=conv_shortcut,
            use_pre_activation=use_pre_activation,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_block{str(i)}",
        )
    return x
