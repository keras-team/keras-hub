import keras

try:
    import tensorflow as tf
except ImportError:
    tf = None
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.audio_converter import AudioConverter


@keras_hub_export("keras_hub.layers.MoonshineAudioConverter")
class MoonshineAudioConverter(AudioConverter):
    """Moonshine audio converter layer. This layer processes raw audio waveforms
    for the Moonshine ASR model.

    Audio is formatted as a batched tensor at 16kHz sample rate and length
    constraints are validated (0.1 to 64 seconds). The layer handles converting
    ragged tensors to dense tensors, reshaping single samples into batches, and
    padding or trimming audio to the appropriate length.

    Defined and formulated in the UsefulSensors implementation of Moonshine:
    [moonshine/moonshine/transcribe.py](https://github.com/usefulsensors/moonshine/blob/main/moonshine/transcribe.py)

    Args:
        sampling_rate: int, optional, The target audio sample rate in Hz.
            Moonshine models expect 16000Hz audio. Defaults to 16000.
        max_audio_length: int, optional, Maximum supported audio length in
            seconds. Longer audio will be truncated. Defaults to 64.
        **kwargs: Additional keyword arguments passed to the base layer.

    Returns:
        A tensor of shape (batch, samples) representing the processed audio
        waveform, padded or trimmed to the appropriate length.

    Examples:

    ```python
    import keras
    import numpy as np
    from keras_hub.layers import MoonshineAudioConverter

    sampling_rate = 16000
    duration = 0.5
    dummy_audio = np.random.randn(
        int(sampling_rate * duration)
    ).astype("float32")
    converter = MoonshineAudioConverter(
        sampling_rate=sampling_rate, max_audio_length=1
    )
    features = converter(dummy_audio)
    print(features.shape)
    ```
    """

    def __init__(self, sampling_rate=16000, max_audio_length=64, **kwargs):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.num_samples = sampling_rate * max_audio_length

    def audio_shape(self):
        return (self.num_samples,)

    def call(self, audio):
        """Process audio tensor through Moonshine preprocessing pipeline.

        1. Convert to a dense tensor if input is ragged.
        2. Expand to [batch, samples] if needed.
        3. Pad/trim to max_audio_length.
        """
        # Handle ragged or list input.
        if tf is not None:
            if isinstance(audio, (list, tuple)):
                audio = tf.ragged.constant(audio).to_tensor(default_value=0)
            elif isinstance(audio, tf.RaggedTensor):
                audio = audio.to_tensor(default_value=0)
        else:
            audio = keras.ops.convert_to_tensor(audio)

        # Expand dims if needed.
        if len(audio.shape) == 1:
            audio = keras.ops.expand_dims(audio, axis=0)
        if len(audio.shape) != 2:
            raise ValueError(
                f"Input must be 1-D (single sample) or 2-D (batch). "
                f"Got shape {audio.shape}"
            )
        # Pad/trim to target length.
        audio = self._pad_or_trim(audio)
        return audio

    def _pad_or_trim(self, audio):
        current_length = keras.ops.shape(audio)[1]
        target_length = self.num_samples
        # Truncate if too long.
        if current_length > target_length:
            audio = audio[:, :target_length]
        # Pad if too short.
        elif current_length < target_length:
            pad_amount = target_length - current_length
            audio = keras.ops.pad(
                audio,
                [[0, 0], [0, pad_amount]],
                mode="constant",
                constant_values=0.0,
            )
        return audio

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sampling_rate": self.sampling_rate,
                "max_audio_length": self.max_audio_length,
            }
        )
        return config
