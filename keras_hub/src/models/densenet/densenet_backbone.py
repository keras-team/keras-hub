import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.feature_pyramid_backbone import FeaturePyramidBackbone

BN_EPSILON = 1.001e-5


@keras_hub_export("keras_hub.models.DenseNetBackbone")
class DenseNetBackbone(FeaturePyramidBackbone):
    """Instantiates the DenseNet architecture.

    This class implements a DenseNet backbone as described in
    [Densely Connected Convolutional Networks (CVPR 2017)](
       https://arxiv.org/abs/1608.06993
    ).

    Args:
        stackwise_num_repeats: list of ints, number of repeated convolutional
            blocks per dense block.
        image_shape: optional shape tuple, defaults to (None, None, 3).
        compression_ratio: float, compression rate at transition layers,
            defaults to 0.5.
        growth_rate: int, number of filters added by each dense block,
            defaults to 32

    Examples:
    ```python
    input_data = np.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_hub.models.DenseNetBackbone.from_preset(
        "densenet_121_imagenet"
    )
    model(input_data)

    # Randomly initialized backbone with a custom config
    model = keras_hub.models.DenseNetBackbone(
        stackwise_num_repeats=[6, 12, 24, 16],
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        stackwise_num_repeats,
        image_shape=(None, None, 3),
        compression_ratio=0.5,
        growth_rate=32,
        **kwargs,
    ):
        # === Functional Model ===
        data_format = keras.config.image_data_format()
        channel_axis = -1 if data_format == "channels_last" else 1
        image_input = keras.layers.Input(shape=image_shape)

        x = image_input  # Intermediate result.
        x = keras.layers.Conv2D(
            64,
            7,
            strides=2,
            use_bias=False,
            padding="same",
            data_format=data_format,
            name="conv1_conv",
        )(x)
        x = keras.layers.BatchNormalization(
            axis=channel_axis, epsilon=BN_EPSILON, name="conv1_bn"
        )(x)
        x = keras.layers.Activation("relu", name="conv1_relu")(x)
        x = keras.layers.MaxPooling2D(
            3, strides=2, padding="same", data_format=data_format, name="pool1"
        )(x)

        pyramid_outputs = {}
        for stack_index in range(len(stackwise_num_repeats) - 1):
            index = stack_index + 2
            x = apply_dense_block(
                x,
                channel_axis,
                stackwise_num_repeats[stack_index],
                growth_rate,
                name=f"stack{stack_index + 1}",
            )
            pyramid_outputs[f"P{index}"] = x
            x = apply_transition_block(
                x,
                channel_axis,
                compression_ratio,
                name=f"transition{stack_index + 1}",
            )

        x = apply_dense_block(
            x,
            channel_axis,
            stackwise_num_repeats[-1],
            growth_rate,
            name=f"stack{len(stackwise_num_repeats)}",
        )
        pyramid_outputs[f"P{len(stackwise_num_repeats) + 1}"] = x
        x = keras.layers.BatchNormalization(
            axis=channel_axis, epsilon=BN_EPSILON, name="bn"
        )(x)
        x = keras.layers.Activation("relu", name="relu")(x)

        super().__init__(inputs=image_input, outputs=x, **kwargs)

        # === Config ===
        self.stackwise_num_repeats = stackwise_num_repeats
        self.compression_ratio = compression_ratio
        self.growth_rate = growth_rate
        self.image_shape = image_shape
        self.pyramid_outputs = pyramid_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_num_repeats": self.stackwise_num_repeats,
                "compression_ratio": self.compression_ratio,
                "growth_rate": self.growth_rate,
                "image_shape": self.image_shape,
            }
        )
        return config


def apply_dense_block(x, channel_axis, num_repeats, growth_rate, name=None):
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
        x = apply_conv_block(
            x, channel_axis, growth_rate, name=f"{name}_block{i + 1}"
        )
    return x


def apply_transition_block(x, channel_axis, compression_ratio, name=None):
    """A transition block.

    Args:
      x: input tensor.
      compression_ratio: float, compression rate at transition layers.
      name: string, block label.
    """
    data_format = keras.config.image_data_format()
    if name is None:
        name = f"transition_block_{keras.backend.get_uid('transition_block')}"

    x = keras.layers.BatchNormalization(
        axis=channel_axis, epsilon=BN_EPSILON, name=f"{name}_bn"
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_relu")(x)
    x = keras.layers.Conv2D(
        int(x.shape[channel_axis] * compression_ratio),
        1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv",
    )(x)
    x = keras.layers.AveragePooling2D(
        2, strides=2, data_format=data_format, name=f"{name}_pool"
    )(x)
    return x


def apply_conv_block(x, channel_axis, growth_rate, name=None):
    """A building block for a dense block.

    Args:
      x: input tensor.
      growth_rate: int, number of filters added by each dense block.
      name: string, block label.
    """
    data_format = keras.config.image_data_format()
    if name is None:
        name = f"conv_block_{keras.backend.get_uid('conv_block')}"

    shortcut = x
    x = keras.layers.BatchNormalization(
        axis=channel_axis, epsilon=BN_EPSILON, name=f"{name}_1_bn"
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_1_relu")(x)
    x = keras.layers.Conv2D(
        4 * growth_rate,
        1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_1_conv",
    )(x)
    x = keras.layers.BatchNormalization(
        axis=channel_axis, epsilon=BN_EPSILON, name=f"{name}_2_bn"
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_2_relu")(x)
    x = keras.layers.Conv2D(
        growth_rate,
        3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_2_conv",
    )(x)
    x = keras.layers.Concatenate(axis=channel_axis, name=f"{name}_concat")(
        [shortcut, x]
    )
    return x
