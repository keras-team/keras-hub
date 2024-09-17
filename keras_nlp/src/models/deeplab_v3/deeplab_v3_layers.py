# Copyright 2024 The KerasNLP Authors
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


class SpatialPyramidPooling(keras.layers.Layer):
    """Implements the Atrous Spatial Pyramid Pooling.

    Reference for Atrous Spatial Pyramid Pooling [Rethinking Atrous Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf) and
    [Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation](https://arxiv.org/pdf/1802.02611.pdf)

    inp = keras.layers.Input((384, 384, 3))
    backbone = keras.applications.EfficientNetB0(
        input_tensor=inp,
        include_top=False)
    output = backbone(inp)
    output = SpatialPyramidPooling(
        dilation_rates=[6, 12, 18])(output)

    # output[4].shape = [None, 16, 16, 256]
    """

    def __init__(
        self,
        dilation_rates,
        num_channels=256,
        activation="relu",
        dropout=0.0,
        **kwargs,
    ):
        """Initializes an Atrous Spatial Pyramid Pooling layer.

        Args:
            dilation_rates: A `list` of integers for parallel dilated conv.
                Usually a sample choice of rates are [6, 12, 18].
            num_channels: An `int` number of output channels, defaults to 256.
            activation: A `str` activation to be used, defaults to 'relu'.
            dropout: A `float` for the dropout rate of the final projection
                output after the activations and batch norm, defaults to 0.0,
                which means no dropout is applied to the output.
            **kwargs: Additional keyword arguments to be passed.
        """
        self.data_format = keras.config.image_data_format()
        self.channel_axis = -1 if self.data_format == "channels_last" else 1
        super().__init__(**kwargs)
        self.dilation_rates = dilation_rates
        self.num_channels = num_channels
        self.activation = activation
        self.dropout = dropout

    def build(self, input_shape):
        channels = input_shape[self.channel_axis]

        # This is the parallel networks that process the input features with
        # different dilation rates. The output from each channel will be merged
        # together and feed to the output.
        self.aspp_parallel_channels = []

        # Channel1 with Conv2D and 1x1 kernel size.
        conv_sequential = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.num_channels,
                    kernel_size=(1, 1),
                    use_bias=False,
                    data_format=self.data_format,
                ),
                keras.layers.BatchNormalization(axis=self.channel_axis),
                keras.layers.Activation(self.activation),
            ]
        )
        conv_sequential.build(input_shape)
        self.aspp_parallel_channels.append(conv_sequential)

        # Channel 2 and afterwards are based on self.dilation_rates, and each of
        # them will have conv2D with 3x3 kernel size.
        for dilation_rate in self.dilation_rates:
            conv_sequential = keras.Sequential(
                [
                    keras.layers.Conv2D(
                        filters=self.num_channels,
                        kernel_size=(3, 3),
                        padding="same",
                        dilation_rate=dilation_rate,
                        use_bias=False,
                        data_format=self.data_format,
                    ),
                    keras.layers.BatchNormalization(axis=self.channel_axis),
                    keras.layers.Activation(self.activation),
                ]
            )
            conv_sequential.build(input_shape)
            self.aspp_parallel_channels.append(conv_sequential)

        # Last channel is the global average pooling with conv2D 1x1 kernel.
        if self.channel_axis == -1:
            reshape = keras.layers.Reshape((1, 1, channels))
        else:
            reshape = keras.layers.Reshape((channels, 1, 1))
        pool_sequential = keras.Sequential(
            [
                keras.layers.GlobalAveragePooling2D(
                    data_format=self.data_format
                ),
                reshape,
                keras.layers.Conv2D(
                    filters=self.num_channels,
                    kernel_size=(1, 1),
                    use_bias=False,
                    data_format=self.data_format,
                ),
                keras.layers.BatchNormalization(axis=self.channel_axis),
                keras.layers.Activation(self.activation),
            ]
        )
        pool_sequential.build(input_shape)
        self.aspp_parallel_channels.append(pool_sequential)

        # Final projection layers
        projection = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.num_channels,
                    kernel_size=(1, 1),
                    use_bias=False,
                    data_format=self.data_format,
                ),
                keras.layers.BatchNormalization(axis=self.channel_axis),
                keras.layers.Activation(self.activation),
                keras.layers.Dropout(rate=self.dropout),
            ],
        )
        projection_input_channels = (
            2 + len(self.dilation_rates)
        ) * self.num_channels
        projection.build(tuple(input_shape[:-1]) + (projection_input_channels,))
        self.projection = projection

    def call(self, inputs, training=None):
        """Calls the Atrous Spatial Pyramid Pooling layer on an input.

        Args:
            inputs: A tensor of shape [batch, height, width, channels]

        Returns:
            A tensor of shape [batch, height, width, num_channels]
        """
        result = []

        for channel in self.aspp_parallel_channels:
            temp = ops.cast(channel(inputs, training=training), inputs.dtype)
            result.append(temp)

        image_shape = ops.shape(inputs)
        if self.channel_axis == -1:
            height, width = image_shape[1], image_shape[2]
        else:
            height, width = image_shape[2], image_shape[3]
        result[self.channel_axis] = keras.layers.Resizing(
            height,
            width,
            interpolation="bilinear",
        )(result[self.channel_axis])

        result = ops.concatenate(result, axis=self.channel_axis)
        result = self.projection(result, training=training)
        return result

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            return (
                input_shape[0],
                self.num_channels,
                input_shape[1],
                input_shape[2],
            )
        else:
            return (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                self.num_channels,
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dilation_rates": self.dilation_rates,
                "num_channels": self.num_channels,
                "activation": self.activation,
                "dropout": self.dropout,
            }
        )
        return config
