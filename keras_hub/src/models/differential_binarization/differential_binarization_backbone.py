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

from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.DifferentialBinarizationBackbone")
class DifferentialBinarizationBackbone(Backbone):
    """
    A Keras model implementing the Differential Binarization
    architecture for scene text detection, described in
    [Real-time Scene Text Detection with Differentiable Binarization](
    https://arxiv.org/abs/1911.08947).

    This class contains the backbone architecture containing the feature
    pyramid network.

    Args:
        image_encoder: A `keras_hub.models.ResNetBackbone` instance.
        fpn_channels: int. The number of channels to output by the feature
            pyramid network. Defaults to 256.
    """

    def __init__(
        self,
        image_encoder,
        fpn_channels=256,
        **kwargs,
    ):
        inputs = image_encoder.input
        x = image_encoder.pyramid_outputs
        x = diffbin_fpn_model(x, out_channels=fpn_channels)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.image_encoder = image_encoder
        self.fpn_channels = fpn_channels

    def get_config(self):
        config = super().get_config()
        config["fpn_channels"] = self.fpn_channels
        config["image_encoder"] = self.image_encoder
        return config


def diffbin_fpn_model(inputs, out_channels):
    in2 = layers.Conv2D(
        out_channels, kernel_size=1, use_bias=False, name="neck_in2"
    )(inputs["P2"])
    in3 = layers.Conv2D(
        out_channels, kernel_size=1, use_bias=False, name="neck_in3"
    )(inputs["P3"])
    in4 = layers.Conv2D(
        out_channels, kernel_size=1, use_bias=False, name="neck_in4"
    )(inputs["P4"])
    in5 = layers.Conv2D(
        out_channels, kernel_size=1, use_bias=False, name="neck_in5"
    )(inputs["P5"])
    out4 = layers.Add(name="add1")([layers.UpSampling2D()(in5), in4])
    out3 = layers.Add(name="add2")([layers.UpSampling2D()(out4), in3])
    out2 = layers.Add(name="add3")([layers.UpSampling2D()(out3), in2])
    p5 = layers.Conv2D(
        out_channels // 4,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="neck_p5",
    )(in5)
    p4 = layers.Conv2D(
        out_channels // 4,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="neck_p4",
    )(out4)
    p3 = layers.Conv2D(
        out_channels // 4,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="neck_p3",
    )(out3)
    p2 = layers.Conv2D(
        out_channels // 4,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="neck_p2",
    )(out2)
    p5 = layers.UpSampling2D((8, 8))(p5)
    p4 = layers.UpSampling2D((4, 4))(p4)
    p3 = layers.UpSampling2D((2, 2))(p3)

    fused = layers.Concatenate(axis=-1)([p5, p4, p3, p2])
    return fused
