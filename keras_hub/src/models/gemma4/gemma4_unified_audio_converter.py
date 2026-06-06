"""Encoder-free audio converter for the Gemma 4 Unified architecture.

Unlike the standard ``Gemma4AudioConverter`` which computes log-mel
spectrograms, this converter simply chunks raw 16 kHz audio waveforms
into fixed-length frames of ``audio_samples_per_token`` samples each.
Each frame becomes one audio soft-token whose feature vector is the raw
waveform samples.
"""

from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.audio_converter import AudioConverter
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone


@keras_hub_export("keras_hub.layers.Gemma4UnifiedAudioConverter")
class Gemma4UnifiedAudioConverter(AudioConverter):
    """Encoder-free audio feature extraction for Gemma 4 Unified.

    Chunks raw 16 kHz audio into fixed-length frames of
    ``audio_samples_per_token`` samples. Each frame corresponds to one
    audio soft-token with the raw waveform samples as features.

    Args:
        audio_samples_per_token: int. Number of raw audio samples per
            output token. At 16 kHz, 640 samples = 40 ms. Defaults to
            ``640``.
        sampling_rate: int. Expected sample rate of the input waveform in
            Hz. Defaults to ``16000``.

    Call arguments:
        audio: array of shape ``(num_samples,)`` or
            ``(batch_size, num_samples)``. Raw mono-channel audio
            waveform(s) at ``sampling_rate`` Hz.

    Returns:
        Tensor of shape ``(num_tokens, audio_samples_per_token)`` for a
        1-D input, or ``(batch_size, num_tokens, audio_samples_per_token)``
        for a 2-D input.

    Examples:

    ```python
    import numpy as np
    import keras_hub

    # Single waveform (1 second at 16 kHz).
    waveform = np.random.randn(16000).astype("float32")
    converter = keras_hub.layers.Gemma4UnifiedAudioConverter()
    features = converter(waveform)
    print(features.shape)  # (25, 640)

    # Batched waveforms.
    batch = np.random.randn(4, 16000).astype("float32")
    features = converter(batch)
    print(features.shape)  # (4, 25, 640)
    ```
    """

    backbone_cls = Gemma4Backbone

    def __init__(
        self,
        audio_samples_per_token=640,
        sampling_rate=16000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

        self.audio_samples_per_token = audio_samples_per_token
        self.sampling_rate = sampling_rate

        # For compatibility with the preprocessor's output length
        # calculation: output_length = waveform_samples // stride.
        self.stride = audio_samples_per_token

        self.built = True

    @property
    def audio_subsampling_factor(self):
        """Number of converter output frames per audio soft-token.

        The unified model maps each output frame to exactly one token
        (no conformer subsampling), so this is always ``1``.
        """
        return 1

    def call(self, audio):
        """Chunk raw waveform(s) into fixed-length frames.

        Args:
            audio: array of shape ``(num_samples,)`` or
                ``(batch_size, num_samples)``.

        Returns:
            Tensor of shape ``(num_tokens, audio_samples_per_token)`` or
            ``(batch_size, num_tokens, audio_samples_per_token)``.
        """
        audio = ops.convert_to_tensor(audio, dtype=self.compute_dtype)
        rank_1_input = len(ops.shape(audio)) == 1
        if rank_1_input:
            audio = ops.expand_dims(audio, axis=0)

        # Pad to be evenly divisible by audio_samples_per_token.
        current_len = ops.shape(audio)[1]
        pad_len = (
            self.audio_samples_per_token
            - current_len % self.audio_samples_per_token
        ) % self.audio_samples_per_token
        if pad_len > 0:
            audio = ops.pad(audio, [[0, 0], [0, pad_len]])

        # Reshape into frames: (B, num_tokens, samples_per_token)
        total_len = ops.shape(audio)[1]
        batch_size = ops.shape(audio)[0]
        num_tokens = total_len // self.audio_samples_per_token
        features = ops.reshape(
            audio, [batch_size, num_tokens, self.audio_samples_per_token]
        )

        if rank_1_input:
            features = ops.squeeze(features, axis=0)

        return features

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "audio_samples_per_token": self.audio_samples_per_token,
                "sampling_rate": self.sampling_rate,
            }
        )
        return config
