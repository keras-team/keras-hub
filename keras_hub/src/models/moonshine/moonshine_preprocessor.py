import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.preprocessor import Preprocessor


@keras_hub_export("keras_hub.models.MoonshinePreprocessor")
class MoonshinePreprocessor(Preprocessor):
    """
    Moonshine preprocessor layer.

    This preprocessor converts raw audio inputs into feature representations
    that are optimized for the Moonshine ASR model. The layer effectively
    downsamples and extracts key features from the audio signal through a series
    of convolutional operations, normalization, and non-linear activations.

    Defined and formulated in the UsefulSensors implementation of Moonshine:
    [moonshine/moonshine/model.py](https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L6)

    Args:
        filter_dim: int, The number of filters for the first convolutional
            layer. This parameter influences the dimensionality of the entire
            feature extraction pipeline and determines the richness of the audio
            representation.
        **kwargs: Additional keyword arguments passed to the base Preprocessor
            class for customization of the underlying preprocessing behavior.

    Returns:
        A tensor of shape `(batch_size, time_steps, filter_dim)` representing
        the processed audio features optimized for the Moonshine ASR model.

    Examples:

    ```python
    import keras
    from keras_hub.models import MoonshinePreprocessor

    dummy_audio = keras.ops.convert_to_tensor(
        [[0.1] * 16000],
        dtype="float32"
    )
    dummy_audio = keras.ops.expand_dims(dummy_audio, axis=-1)
    preprocessor = MoonshinePreprocessor(filter_dim=256)
    features = preprocessor(dummy_audio)
    print(features)
    ```
    """

    def __init__(self, filter_dim, **kwargs):
        super().__init__(**kwargs)
        inputs = keras.layers.Input(shape=[None, 1])
        conv1 = keras.layers.Conv1D(
            filters=filter_dim,
            kernel_size=127,
            strides=64,
            use_bias=False,
        )
        tanh = keras.layers.Activation("tanh")
        group_norm = keras.layers.GroupNormalization(
            groups=1, axis=-1, epsilon=1e-5
        )
        conv2 = keras.layers.Conv1D(
            filters=2 * filter_dim, kernel_size=7, strides=3, padding="valid"
        )
        gelu1 = keras.layers.Activation("gelu")
        conv3 = keras.layers.Conv1D(
            filters=filter_dim, kernel_size=3, strides=2, padding="valid"
        )
        gelu2 = keras.layers.Activation("gelu")
        preprocess = keras.Sequential(
            [conv1, tanh, group_norm, conv2, gelu1, conv3, gelu2]
        )
        outputs = preprocess(inputs)
        self.preprocess = keras.Model(inputs=inputs, outputs=outputs)
        self.filter_dim = filter_dim

    def call(self, inputs):
        return self.preprocess(inputs)
