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

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.backbone import Backbone

BN_AXIS = 3
BN_EPSILON = 1.001e-5


@keras_nlp_export("keras_nlp.models.DenseNetBackbone")
class DenseNetBackbone(Backbone):
    """Instantiates the DenseNet architecture.

    This class implements a DenseNet backbone as described in
    [Densely Connected Convolutional Networks (CVPR 2017)](
       https://arxiv.org/abs/1608.06993
    ).

    Args:
        stackwise_num_repeats: list of ints, number of repeated convolutional
            blocks per dense block.
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer. Defaults to `True`.
        input_image_shape: optional shape tuple, defaults to (224, 224, 3).
        compression_ratio: float, compression rate at transition layers,
            defaults to 0.5.
        growth_rate: int, number of filters added by each dense block,
            defaults to 32

    Examples:
    ```python
    input_data = np.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_nlp.models.DenseNetBackbone.from_preset("densenet121_imagenet")
    model(input_data)

    # Randomly initialized backbone with a custom config
    model = keras_nlp.models.DenseNetBackbone(
        stackwise_num_repeats=[6, 12, 24, 16],
        include_rescaling=False,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        stackwise_num_repeats,
        include_rescaling=True,
        input_image_shape=(224, 224, 3),
        compression_ratio=0.5,
        growth_rate=32,
        **kwargs,
    ):
        # === Functional Model ===
        image_input = keras.layers.Input(shape=input_image_shape)

        x = image_input
        if include_rescaling:
            x = keras.layers.Rescaling(1 / 255.0)(x)

        x = keras.layers.Conv2D(
            64, 7, strides=2, use_bias=False, padding="same", name="conv1_conv"
        )(x)
        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="conv1_bn"
        )(x)
        x = keras.layers.Activation("relu", name="conv1_relu")(x)
        x = keras.layers.MaxPooling2D(
            3, strides=2, padding="same", name="pool1"
        )(x)

        for stack_index in range(len(stackwise_num_repeats) - 1):
            index = stack_index + 2
            x = apply_dense_block(
                x,
                stackwise_num_repeats[stack_index],
                growth_rate,
                name=f"conv{index}",
            )
            x = apply_transition_block(
                x, compression_ratio, name=f"pool{index}"
            )

        x = apply_dense_block(
            x,
            stackwise_num_repeats[-1],
            growth_rate,
            name=f"conv{len(stackwise_num_repeats) + 1}",
        )

        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="bn"
        )(x)
        x = keras.layers.Activation("relu", name="relu")(x)

        super().__init__(inputs=image_input, outputs=x, **kwargs)

        # === Config ===
        self.stackwise_num_repeats = stackwise_num_repeats
        self.include_rescaling = include_rescaling
        self.compression_ratio = compression_ratio
        self.growth_rate = growth_rate
        self.input_image_shape = input_image_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_num_repeats": self.stackwise_num_repeats,
                "include_rescaling": self.include_rescaling,
                "compression_ratio": self.compression_ratio,
                "growth_rate": self.growth_rate,
                "input_image_shape": self.input_image_shape,
            }
        )
        return config


def apply_dense_block(x, num_repeats, growth_rate, name=None):
    """A dense block.

    Args:
      x: input tensor.
      num_repeats: int, number of repeated convolutional blocks.
      growth_rate: int, number of filters added by each dense block.
      name: string, block label.
    """
    if name is None:
        name = f"dense_block_{keras.backend.get_uid('dense_block')}"

    for i in range(num_repeats):
        x = apply_conv_block(x, growth_rate, name=f"{name}_block_{i}")
    return x


def apply_transition_block(x, compression_ratio, name=None):
    """A transition block.

    Args:
      x: input tensor.
      compression_ratio: float, compression rate at transition layers.
      name: string, block label.
    """
    if name is None:
        name = f"transition_block_{keras.backend.get_uid('transition_block')}"

    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=f"{name}_bn"
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_relu")(x)
    x = keras.layers.Conv2D(
        int(x.shape[BN_AXIS] * compression_ratio),
        1,
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = keras.layers.AveragePooling2D(2, strides=2, name=f"{name}_pool")(x)
    return x


def apply_conv_block(x, growth_rate, name=None):
    """A building block for a dense block.

    Args:
      x: input tensor.
      growth_rate: int, number of filters added by each dense block.
      name: string, block label.
    """
    if name is None:
        name = f"conv_block_{keras.backend.get_uid('conv_block')}"

    shortcut = x
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=f"{name}_0_bn"
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_0_relu")(x)
    x = keras.layers.Conv2D(
        4 * growth_rate, 1, use_bias=False, name=f"{name}_1_conv"
    )(x)
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=f"{name}_1_bn"
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_1_relu")(x)
    x = keras.layers.Conv2D(
        growth_rate,
        3,
        padding="same",
        use_bias=False,
        name=f"{name}_2_conv",
    )(x)
    x = keras.layers.Concatenate(axis=BN_AXIS, name=f"{name}_concat")(
        [shortcut, x]
    )
    return x
