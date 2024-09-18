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

BN_AXIS = 3

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}


class MBConvBlock(keras.layers.Layer):
    def __init__(
        self,
        input_filters,
        output_filters,
        expand_ratio=1,
        kernel_size=3,
        strides=1,
        se_ratio=0.0,
        batch_norm_momentum=0.9,
        activation="swish",
        dropout=0.2,
        **kwargs
    ):
        """Implementation of the MBConv block

        Also known as a Mobile Inverted Residual Bottleneck block from:
            [MobileNetV2: Inverted Residuals and Linear Bottlenecks]
            (https://arxiv.org/abs/1801.04381v4).

        MBConv blocks are common blocks used in mobile-oriented and efficient
        architectures, present in architectures such as MobileNet, EfficientNet,
        MaxViT, etc.

        MBConv blocks follow a narrow-wide-narrow structure - expanding a 1x1
        convolution, applying depthwise convolution, and narrowing back to a 1x1
        convolution, which is a more efficient operation than conventional
        wide-narrow-wide structures.

        As they're frequently used for models to be deployed to edge devices,
        they're implemented as a layer for ease of use and re-use.

        Args:
            input_filters: int, the number of input filters
            output_filters: int, the optional number of output filters after
                Squeeze-Excitation
            expand_ratio: default 1, the ratio by which input_filters are
                multiplied to expand the structure in the middle expansion phase
            kernel_size: default 3, the kernel_size to apply to the expansion
                phase convolutions
            strides: default 1, the strides to apply to the expansion phase
                convolutions
            se_ratio: default 0.0, Squeeze-Excitation happens before depthwise
                convolution and before output convolution only if the se_ratio
                is above 0. The filters used in this phase are chosen as the
                maximum between 1 and input_filters*se_ratio
            batch_norm_momentum: default 0.9, the BatchNormalization momentum
            activation: default "swish", the activation function used between
                convolution operations
            dropout: float, the optional dropout rate to apply before the output
                convolution, defaults to 0.2

        Returns:
            A tensor representing a feature map, passed through the MBConv
            block


        Note:
            Not intended to be used outside of the EfficientNet architecture.
        """

        super().__init__(**kwargs)
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self.batch_norm_momentum = batch_norm_momentum
        self.activation = activation
        self.dropout = dropout
        self.filters = self.input_filters * self.expand_ratio
        self.filters_se = max(1, int(input_filters * se_ratio))

        self.conv1 = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=self.name + "expand_conv",
        )
        self.bn1 = keras.layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=self.batch_norm_momentum,
            name=self.name + "expand_bn",
        )
        self.act = keras.layers.Activation(
            self.activation, name=self.name + "activation"
        )
        self.depthwise = keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=self.name + "dwconv2",
        )

        self.bn2 = keras.layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=self.batch_norm_momentum,
            name=self.name + "bn",
        )

        self.se_conv1 = keras.layers.Conv2D(
            self.filters_se,
            1,
            padding="same",
            activation=self.activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=self.name + "se_reduce",
        )

        self.se_conv2 = keras.layers.Conv2D(
            self.filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=self.name + "se_expand",
        )

        self.output_conv = keras.layers.Conv2D(
            filters=self.output_filters,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=self.name + "project_conv",
        )

        self.bn3 = keras.layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=self.batch_norm_momentum,
            name=self.name + "project_bn",
        )

        if self.dropout:
            self.dropout_layer = keras.layers.Dropout(
                self.dropout,
                noise_shape=(None, 1, 1, 1),
                name=self.name + "drop",
            )

    def build(self, input_shape):
        if self.name is None:
            self.name = keras.backend.get_uid("block0")

    def call(self, inputs):
        # Expansion phase
        if self.expand_ratio != 1:
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = self.act(x)
        else:
            x = inputs

        # Depthwise conv
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act(x)

        # Squeeze and excite
        if 0 < self.se_ratio <= 1:
            se = keras.layers.GlobalAveragePooling2D(
                name=self.name + "se_squeeze"
            )(x)
            if BN_AXIS == 1:
                se_shape = (self.filters, 1, 1)
            else:
                se_shape = (1, 1, self.filters)
            se = keras.layers.Reshape(se_shape, name=self.name + "se_reshape")(
                se
            )

            se = self.se_conv1(se)
            se = self.se_conv2(se)

            x = keras.layers.multiply([x, se], name=self.name + "se_excite")

        # Output phase
        x = self.output_conv(x)
        x = self.bn3(x)

        if self.strides == 1 and self.input_filters == self.output_filters:
            if self.dropout:
                x = self.dropout_layer(x)
            x = keras.layers.Add(name=self.name + "add")([x, inputs])
        return x

    def get_config(self):
        config = {
            "input_filters": self.input_filters,
            "output_filters": self.output_filters,
            "expand_ratio": self.expand_ratio,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "se_ratio": self.se_ratio,
            "batch_norm_momentum": self.batch_norm_momentum,
            "activation": self.activation,
            "dropout": self.dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
