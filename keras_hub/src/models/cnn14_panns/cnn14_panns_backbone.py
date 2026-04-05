import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras.saving.register_keras_serializable(package="keras_hub")
class ReduceMean(keras.layers.Layer):
    """Custom layer for mean reduction across a specific axis.

    Used in PANNs to reduce the frequency dimension and time dimension
    at different stages of the pooling process.
    """

    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return ops.mean(x, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class ReduceMax(keras.layers.Layer):
    """Custom layer for max reduction across a specific axis.

    Used in PANNs for global max pooling across the time dimension.
    """

    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return ops.max(x, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


def conv_block(x, filters, name, pool_size=(2, 2), dropout_rate=0.2):
    """A standard PANNs convolutional block.

    Consists of two 3x3 convolutions, each followed by BatchNormalization
    and ReLU, then AveragePooling and Dropout.
    """
    x = keras.layers.Conv2D(
        filters, (3, 3), padding="same", use_bias=False, name=f"{name}_conv1"
    )(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{name}_bn1")(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters, (3, 3), padding="same", use_bias=False, name=f"{name}_conv2"
    )(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{name}_bn2")(x)
    x = keras.layers.ReLU()(x)
    if pool_size:
        x = keras.layers.AveragePooling2D(
            pool_size=pool_size, name=f"{name}_pool"
        )(x)
    x = keras.layers.Dropout(rate=dropout_rate)(x)
    return x


@keras_hub_export("keras_hub.models.Cnn14PannsBackbone")
class Cnn14PannsBackbone(Backbone):
    """CNN14 PANNs Backbone architecture.

    This architecture is based on the Large-Scale Pretrained Audio Neural
    Networks (PANNs) for Audio Tagging. It consists of a stack of
    convolutional blocks that act as a feature extractor for audio log-mel
    spectrograms.

    Reference:
    - [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Tagging](https://arxiv.org/abs/1912.10211)
    - [Official PyTorch Implementation](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py)

    Args:
        stackwise_num_filters: list of ints. The number of filters for each
            convolutional block. Defaults to `[64, 128, 256, 512, 1024, 2048]`.
        input_shape: tuple, optional. Shape of the input log-mel spectrogram.
            Defaults to `(None, 64, 1)`.
        **kwargs: Standard `Backbone` arguments.
    """

    def __init__(
        self,
        stackwise_num_filters=(64, 128, 256, 512, 1024, 2048),
        input_shape=(None, 64, 1),
        **kwargs,
    ):
        # Define the functional model
        inputs = keras.Input(shape=input_shape, name="logmel")

        # Initial BatchNormalization on the frequency axis
        # Note: axis=-2 corresponds to the frequency bins (e.g., 64).
        # This requires the frequency dimension to be known at build time.
        x = keras.layers.BatchNormalization(axis=-2, epsilon=1e-5, name="bn0")(
            inputs
        )

        # Stacks of Convolutional blocks
        for i, filters in enumerate(stackwise_num_filters):
            # In the standard CNN14, the last block has (1, 1) pooling
            pool_size = (2, 2) if i < len(stackwise_num_filters) - 1 else (1, 1)
            x = conv_block(
                x, filters, name=f"conv_block{i + 1}", pool_size=pool_size
            )

        # Initialize the Backbone with the functional model
        # The output is the 4D feature map (Batch, Time, Freq, Channels)
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # === Config ===
        self.stackwise_num_filters = list(stackwise_num_filters)
        self._input_shape = input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_num_filters": self.stackwise_num_filters,
                "input_shape": self._input_shape,
            }
        )
        return config
