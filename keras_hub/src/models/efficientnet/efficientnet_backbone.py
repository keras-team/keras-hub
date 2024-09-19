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
import math

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.efficientnet.fusedmbconv import FusedMBConvBlock
from keras_hub.src.models.efficientnet.mbconv import MBConvBlock
from keras_hub.src.models.feature_pyramid_backbone import FeaturePyramidBackbone


@keras_hub_export("keras_hub.models.EfficientNetBackbone")
class EfficientNetBackbone(FeaturePyramidBackbone):
    """An EfficientNet backbone model.

    This class encapsulates the architectures for both EfficientNetV1 and
    EfficientNetV2. EfficientNetV2 uses Fused-MBConv Blocks and Neural
    Architecture Search (NAS) to make models sizes much smaller while still
    improving overall model quality.

    References:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks]
        (https://arxiv.org/abs/1905.11946) (ICML 2019)
    - [Based on the original keras.applications EfficientNet]
        (https://github.com/keras-team/keras/blob/master/keras/applications/efficientnet.py)
    - [EfficientNetV2: Smaller Models and Faster Training]
        (https://arxiv.org/abs/2104.00298) (ICML 2021)

    Args:
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        dropout: float, dropout rate at skip connections. The default
            value is set to 0.2.
        depth_divisor: integer, a unit of network width. The default value is
            set to 8.
        activation: activation function to use between each convolutional layer.
        input_shape: optional shape tuple, it should have exactly 3 input
            channels.
        stackwise_kernel_sizes:  list of ints, the kernel sizes used for each
            conv block.
        stackwise_num_repeats: list of ints, number of times to repeat each
            conv block.
        stackwise_input_filters: list of ints, number of input filters for
            each conv block.
        stackwise_output_filters: list of ints, number of output filters for
            each stack in the conv blocks model.
        stackwise_expansion_ratios: list of floats, expand ratio passed to the
            squeeze and excitation blocks.
        stackwise_strides: list of ints, stackwise_strides for each conv block.
        stackwise_squeeze_and_excite_ratios: list of ints, the squeeze and
            excite ratios passed to the squeeze and excitation blocks.
        stackwise_block_types: list of strings.  Each value is either 'v1',
            'unfused' or 'fused' depending on the desired blocks.  'v1' uses the
            original efficientnet block. FusedMBConvBlock is similar to
            MBConvBlock, but instead of using a depthwise convolution and a 1x1
            output convolution blocks fused blocks use a single 3x3 convolution
            block.
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        min_depth: integer, minimum number of filters. Can be None and ignored
            if use_depth_divisor_as_min_depth is set to True.
        include_initial_padding: bool, whether to include initial zero padding
            (as per v1).
        use_depth_divisor_as_min_depth: bool, whether to use depth_divisor as
            the minimum depth instead of min_depth (as per v1).
        cap_round_filter_decrease: bool, whether to cap the max decrease in the
            number of filters the rounding process potentially produces
            (as per v1).
        stem_conv_padding: str, can be 'same' or 'valid'. Padding for the stem.
        batch_norm_momentum: float, momentum for the moving average calcualtion
            in the batch normalization layers.

    Example:
    ```python
    # You can customize the EfficientNet architecture:
    model = EfficientNetBackbone(
        stackwise_kernel_sizes=[3, 3, 3, 3, 3, 3],
        stackwise_num_repeats=[2, 4, 4, 6, 9, 15],
        stackwise_input_filters=[24, 24, 48, 64, 128, 160],
        stackwise_output_filters=[24, 48, 64, 128, 160, 256],
        stackwise_expansion_ratios=[1, 4, 4, 4, 6, 6],
        stackwise_squeeze_and_excite_ratios=[0.0, 0.0, 0, 0.25, 0.25, 0.25],
        stackwise_strides=[1, 2, 2, 2, 1, 2],
        stackwise_block_types=[["fused"] * 3 + ["unfused"] * 3],
        width_coefficient=1.0,
        depth_coefficient=1.0,
        include_rescaling=False,
    )
    images = np.ones((1, 256, 256, 3))
    outputs = efficientnet.predict(images)
    ```
    """

    def __init__(
        self,
        *,
        width_coefficient,
        depth_coefficient,
        stackwise_kernel_sizes,
        stackwise_num_repeats,
        stackwise_input_filters,
        stackwise_output_filters,
        stackwise_expansion_ratios,
        stackwise_squeeze_and_excite_ratios,
        stackwise_strides,
        stackwise_block_types,
        include_rescaling=True,
        dropout=0.2,
        depth_divisor=8,
        min_depth=8,
        input_shape=(None, None, 3),
        activation="swish",
        include_initial_padding=False,
        use_depth_divisor_as_min_depth=False,
        cap_round_filter_decrease=False,
        stem_conv_padding="same",
        batch_norm_momentum=0.9,
        **kwargs,
    ):
        img_input = keras.layers.Input(shape=input_shape)

        x = img_input

        if include_rescaling:
            # Use common rescaling strategy across keras
            x = keras.layers.Rescaling(scale=1.0 / 255.0)(x)

        if include_initial_padding:
            x = keras.layers.ZeroPadding2D(
                padding=self._correct_pad_downsample(x, 3),
                name="stem_conv_pad",
            )(x)

        # Build stem
        stem_filters = round_filters(
            filters=stackwise_input_filters[0],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
            use_depth_divisor_as_min_depth=use_depth_divisor_as_min_depth,
            cap_round_filter_decrease=cap_round_filter_decrease,
        )

        x = keras.layers.Conv2D(
            filters=stem_filters,
            kernel_size=3,
            strides=2,
            padding=stem_conv_padding,
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name="stem_conv",
        )(x)

        x = keras.layers.BatchNormalization(
            momentum=batch_norm_momentum,
            name="stem_bn",
        )(x)
        x = keras.layers.Activation(activation, name="stem_activation")(x)

        # Build blocks
        block_id = 0
        blocks = float(sum(stackwise_num_repeats))

        self._pyramid_outputs = {}
        curr_pyramid_level = 1

        for i in range(len(stackwise_kernel_sizes)):
            num_repeats = stackwise_num_repeats[i]
            input_filters = stackwise_input_filters[i]
            output_filters = stackwise_output_filters[i]

            # Update block input and output filters based on depth multiplier.
            input_filters = round_filters(
                filters=input_filters,
                width_coefficient=width_coefficient,
                min_depth=min_depth,
                depth_divisor=depth_divisor,
                use_depth_divisor_as_min_depth=use_depth_divisor_as_min_depth,
                cap_round_filter_decrease=cap_round_filter_decrease,
            )
            output_filters = round_filters(
                filters=output_filters,
                width_coefficient=width_coefficient,
                min_depth=min_depth,
                depth_divisor=depth_divisor,
                use_depth_divisor_as_min_depth=use_depth_divisor_as_min_depth,
                cap_round_filter_decrease=cap_round_filter_decrease,
            )

            repeats = round_repeats(
                repeats=num_repeats,
                depth_coefficient=depth_coefficient,
            )
            strides = stackwise_strides[i]
            squeeze_and_excite_ratio = stackwise_squeeze_and_excite_ratios[i]

            for j in range(repeats):
                # The first block needs to take care of stride and filter size
                # increase.
                if j > 0:
                    strides = 1
                    input_filters = output_filters

                if strides != 1:
                    self._pyramid_outputs[f"P{curr_pyramid_level}"] = x
                    curr_pyramid_level += 1

                # 97 is the start of the lowercase alphabet.
                letter_identifier = chr(j + 97)
                stackwise_block_type = stackwise_block_types[i]
                block_name = f"block{i + 1}{letter_identifier}_"
                if stackwise_block_type == "v1":
                    x = self._apply_efficientnet_block(
                        inputs=x,
                        filters_in=input_filters,
                        filters_out=output_filters,
                        kernel_size=stackwise_kernel_sizes[i],
                        strides=strides,
                        expand_ratio=stackwise_expansion_ratios[i],
                        se_ratio=squeeze_and_excite_ratio,
                        activation=activation,
                        dropout=dropout * block_id / blocks,
                        name=block_name,
                    )
                else:
                    block = get_conv_constructor(stackwise_block_type)(
                        input_filters=input_filters,
                        output_filters=output_filters,
                        expand_ratio=stackwise_expansion_ratios[i],
                        kernel_size=stackwise_kernel_sizes[i],
                        strides=strides,
                        se_ratio=squeeze_and_excite_ratio,
                        activation=activation,
                        dropout=dropout * block_id / blocks,
                        batch_norm_momentum=batch_norm_momentum,
                        name=block_name,
                    )
                    x = block(x)
                block_id += 1

        # Build top
        top_filters = round_filters(
            filters=1280,
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
            use_depth_divisor_as_min_depth=use_depth_divisor_as_min_depth,
            cap_round_filter_decrease=cap_round_filter_decrease,
        )

        x = keras.layers.Conv2D(
            filters=top_filters,
            kernel_size=1,
            padding="same",
            strides=1,
            kernel_initializer=conv_kernel_initializer(),
            use_bias=False,
            name="top_conv",
            data_format="channels_last",
        )(x)
        x = keras.layers.BatchNormalization(
            momentum=batch_norm_momentum,
            name="top_bn",
        )(x)
        x = keras.layers.Activation(
            activation=activation, name="top_activation"
        )(x)

        self._pyramid_outputs[f"P{curr_pyramid_level}"] = x
        curr_pyramid_level += 1

        # Create model.
        super().__init__(inputs=img_input, outputs=x, **kwargs)

        # === Config ===
        self.include_rescaling = include_rescaling
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout = dropout
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.activation = activation
        self.stackwise_kernel_sizes = stackwise_kernel_sizes
        self.stackwise_num_repeats = stackwise_num_repeats
        self.stackwise_input_filters = stackwise_input_filters
        self.stackwise_output_filters = stackwise_output_filters
        self.stackwise_expansion_ratios = stackwise_expansion_ratios
        self.stackwise_squeeze_and_excite_ratios = (
            stackwise_squeeze_and_excite_ratios
        )
        self.stackwise_strides = stackwise_strides
        self.stackwise_block_types = stackwise_block_types

        self.include_initial_padding = include_initial_padding
        self.use_depth_divisor_as_min_depth = use_depth_divisor_as_min_depth
        self.cap_round_filter_decrease = cap_round_filter_decrease
        self.stem_conv_padding = stem_conv_padding
        self.batch_norm_momentum = batch_norm_momentum

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_rescaling": self.include_rescaling,
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "dropout": self.dropout,
                "depth_divisor": self.depth_divisor,
                "min_depth": self.min_depth,
                "activation": self.activation,
                "input_shape": self.input_shape[1:],
                "stackwise_kernel_sizes": self.stackwise_kernel_sizes,
                "stackwise_num_repeats": self.stackwise_num_repeats,
                "stackwise_input_filters": self.stackwise_input_filters,
                "stackwise_output_filters": self.stackwise_output_filters,
                "stackwise_expansion_ratios": self.stackwise_expansion_ratios,
                "stackwise_squeeze_and_excite_ratios": self.stackwise_squeeze_and_excite_ratios,
                "stackwise_strides": self.stackwise_strides,
                "stackwise_block_types": self.stackwise_block_types,
                "include_initial_padding": self.include_initial_padding,
                "use_depth_divisor_as_min_depth": self.use_depth_divisor_as_min_depth,
                "cap_round_filter_decrease": self.cap_round_filter_decrease,
                "stem_conv_padding": self.stem_conv_padding,
                "batch_norm_momentum": self.batch_norm_momentum,
            }
        )
        return config

    def _correct_pad_downsample(self, inputs, kernel_size):
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

    def _apply_efficientnet_block(
        self,
        inputs,
        filters_in=32,
        filters_out=16,
        kernel_size=3,
        strides=1,
        activation="swish",
        expand_ratio=1,
        se_ratio=0.0,
        dropout=0.0,
        name="",
    ):
        """An inverted residual block.

        Args:
            inputs: Tensor, The input tensor of the block
            filters_in: integer, the number of input filters.
            filters_out: integer, the number of output filters.
            kernel_size: integer, the dimension of the convolution window.
            strides: integer, the stride of the convolution.
            activation: activation function to use between each convolutional layer.
            expand_ratio: integer, scaling coefficient for the input filters.
            se_ratio: float between 0 and 1, fraction to squeeze the input filters.
            dropout: float between 0 and 1, fraction of the input units to drop.
            name: string, block label.

        Returns:
            output tensor for the block.
        """
        filters = filters_in * expand_ratio
        if expand_ratio != 1:
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=False,
                kernel_initializer=conv_kernel_initializer(),
                name=name + "expand_conv",
            )(inputs)
            x = keras.layers.BatchNormalization(
                axis=3,
                name=name + "expand_bn",
            )(x)
            x = keras.layers.Activation(
                activation, name=name + "expand_activation"
            )(x)
        else:
            x = inputs

        # Depthwise Convolution
        if strides == 2:
            x = keras.layers.ZeroPadding2D(
                padding=self._correct_pad_downsample(x, kernel_size),
                name=name + "dwconv_pad",
            )(x)
            conv_pad = "valid"
        else:
            conv_pad = "same"

        x = keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=conv_pad,
            use_bias=False,
            depthwise_initializer=conv_kernel_initializer(),
            name=name + "dwconv",
        )(x)
        x = keras.layers.BatchNormalization(
            axis=3,
            name=name + "dwconv_bn",
        )(x)
        x = keras.layers.Activation(
            activation, name=name + "dwconv_activation"
        )(x)

        # Squeeze and Excitation phase
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            se = keras.layers.GlobalAveragePooling2D(name=name + "se_squeeze")(
                x
            )
            se_shape = (1, 1, filters)
            se = keras.layers.Reshape(se_shape, name=name + "se_reshape")(se)
            se = keras.layers.Conv2D(
                filters_se,
                1,
                padding="same",
                activation=activation,
                kernel_initializer=conv_kernel_initializer(),
                name=name + "se_reduce",
            )(se)
            se = keras.layers.Conv2D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=conv_kernel_initializer(),
                name=name + "se_expand",
            )(se)
            x = keras.layers.multiply([x, se], name=name + "se_excite")

        # Output phase
        x = keras.layers.Conv2D(
            filters=filters_out,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name=name + "project",
        )(x)
        x = keras.layers.BatchNormalization(
            axis=3,
            name=name + "project_bn",
        )(x)
        x = keras.layers.Activation(
            activation, name=name + "project_activation"
        )(x)

        if strides == 1 and filters_in == filters_out:
            if dropout > 0:
                x = keras.layers.Dropout(
                    dropout,
                    noise_shape=(None, 1, 1, 1),
                    name=name + "drop",
                )(x)
            x = keras.layers.Add(name=name + "add")([x, inputs])

        return x


def conv_kernel_initializer(scale=2.0):
    return keras.initializers.VarianceScaling(
        scale=scale, mode="fan_out", distribution="truncated_normal"
    )


def round_filters(
    filters,
    width_coefficient,
    min_depth,
    depth_divisor,
    use_depth_divisor_as_min_depth,
    cap_round_filter_decrease,
):
    """Round number of filters based on depth multiplier.

    Args:
        filters: int, number of filters for Conv layer
        width_coefficient: float, denotes the scaling coefficient of network
            width
        depth_divisor: int, a unit of network width
        use_depth_divisor_as_min_depth: bool, whether to use depth_divisor as
            the minimum depth instead of min_depth (as per v1)
        max_round_filter_decrease: bool, whether to cap the decrease in the
            number of filters this process produces (as per v1)

    Returns:
        int, new rounded filters value for Conv layer
    """
    filters *= width_coefficient

    if use_depth_divisor_as_min_depth:
        min_depth = depth_divisor

    new_filters = max(
        min_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )

    if cap_round_filter_decrease:
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += depth_divisor

    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier.

    Args:
        repeats: int, number of repeats of efficientnet block
        depth_coefficient: float, denotes the scaling coefficient of network
            depth

    Returns:
        int, rounded repeats
    """
    return int(math.ceil(depth_coefficient * repeats))


def get_conv_constructor(conv_type):
    if conv_type == "unfused":
        return MBConvBlock
    elif conv_type == "fused":
        return FusedMBConvBlock
    else:
        raise ValueError(
            "Expected `conv_type` to be "
            "one of 'unfused', 'fused', but got "
            f"`conv_type={conv_type}`"
        )
