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

import math

import keras
from keras import layers
from keras_hub.src.api_export import keras_nlp_export
from keras_hub.src.models.task import Task

from keras_hub.src.models.differential_binarization.losses import DBLoss


@keras_nlp_export("keras_hub.models.DifferentialBinarization")
class DifferentialBinarization(Task):
    """
    A Keras model implementing the Differential Binarization
    architecture for scene text detection, described in
    [Real-time Scene Text Detection with Differentiable Binarization](
    https://arxiv.org/abs/1911.08947).

    Args:
        backbone: A `keras_hub.models.ResNetBackbone` instance.
        fpn_channels: int. The number of channels to output by the Feature
            Pyramid Network. Defaults to 256.
        fpn_kernel_list: list of ints. The number of filters for probability
            and threshold maps. Defaults to [3, 2, 2].

    Examples:
    ```python
    input_data = np.ones(shape=(8, 224, 224, 3))

    backbone = keras_hub.models.ResNetBackbone.from_preset("resnet50_vd")
    detector = keras_hub.models.DifferentialBinarization(
        backbone=backbone
    )

    detector(input_data)
    ```
    """

    def __init__(
        self,
        backbone,
        fpn_channels=256,
        head_kernel_list=[3, 2, 2],
        step_function_k=50.0,
        preprocessor=None,  # adding this dummy arg for saved model test
        **kwargs,
    ):
        backbone = backbone

        inputs = backbone.input
        x = backbone.pyramid_outputs
        x = diffbin_fpn_model(x, out_channels=fpn_channels)
        probability_maps = diffbin_head(
            x,
            in_channels=fpn_channels,
            kernel_list=head_kernel_list,
            name="head_prob",
        )
        threshold_maps = diffbin_head(
            x,
            in_channels=fpn_channels,
            kernel_list=head_kernel_list,
            name="head_thresh",
        )
        binary_maps = step_function(
            probability_maps, threshold_maps, k=step_function_k
        )
        outputs = layers.Concatenate(axis=-1)(
            [probability_maps, threshold_maps, binary_maps]
        )

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.backbone = backbone
        self.fpn_channels = fpn_channels
        self.head_kernel_list = head_kernel_list
        self.step_function_k = step_function_k

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        **kwargs,
    ):
        """Configures the `DifferentialBinarization` task for training.

        `DifferentialBinarization` extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer` and `loss`. To
        override these defaults, pass any value to these arguments during
        compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default optimizer
                for `DifferentialBinarization`. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer` values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, in which case the default loss
                computation of `DifferentialBinarization` will be applied. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.SGD(
                learning_rate=0.007, weight_decay=0.0001, momentum=0.9
            )
        if loss == "auto":
            loss = DBLoss()
        super().compile(
            optimizer=optimizer,
            loss=loss,
            **kwargs,
        )

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "fpn_channels": self.fpn_channels,
                "head_kernel_list": self.head_kernel_list,
                "step_function_k": self.step_function_k,
            }
        )
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


def step_function(x, y, k):
    return 1.0 / (1.0 + keras.ops.exp(-k * (x - y)))


def diffbin_head(inputs, in_channels, kernel_list, name):
    x = layers.Conv2D(
        in_channels // 4,
        kernel_size=kernel_list[0],
        padding="same",
        use_bias=False,
        name=f"{name}_conv0_weights",
    )(inputs)
    x = layers.BatchNormalization(
        beta_initializer=keras.initializers.Constant(1e-4),
        gamma_initializer=keras.initializers.Constant(1.0),
        name=f"{name}_conv0_bn",
    )(x)
    x = layers.ReLU(name=f"{name}_conv0_relu")(x)
    x = layers.Conv2DTranspose(
        in_channels // 4,
        kernel_size=kernel_list[1],
        strides=2,
        padding="valid",
        bias_initializer=keras.initializers.RandomUniform(
            minval=-1.0 / math.sqrt(in_channels // 4 * 1.0),
            maxval=1.0 / math.sqrt(in_channels // 4 * 1.0),
        ),
        name=f"{name}_conv1_weights",
    )(x)
    x = layers.BatchNormalization(
        beta_initializer=keras.initializers.Constant(1e-4),
        gamma_initializer=keras.initializers.Constant(1.0),
        name=f"{name}_conv1_bn",
    )(x)
    x = layers.ReLU(name=f"{name}_conv1_relu")(x)
    x = layers.Conv2DTranspose(
        1,
        kernel_size=kernel_list[2],
        strides=2,
        padding="valid",
        activation="sigmoid",
        bias_initializer=keras.initializers.RandomUniform(
            minval=-1.0 / math.sqrt(in_channels // 4 * 1.0),
            maxval=1.0 / math.sqrt(in_channels // 4 * 1.0),
        ),
        name=f"{name}_conv2_weights",
    )(x)
    return x
