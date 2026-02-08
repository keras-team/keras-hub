"""Audio preprocessing converter for Qwen3-Omni."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.models.whisper.whisper_audio_converter import (
    WhisperAudioConverter,
)


@keras_hub_export("keras_hub.layers.Qwen3OmniAudioConverter")
class Qwen3OmniAudioConverter(WhisperAudioConverter):
    """Audio preprocessing for Qwen3-Omni.

    Converts raw audio to log-mel spectrogram features compatible with the
    Qwen3-Omni audio encoder. This uses Whisper-style featur extraction,
    which Qwen3-Omni uses for audio preprocessing.

    Compared to the base Whisper converter, the defaults are adjusted for
    Qwen3-Omni: 128 mel bins (vs 80) and 300s max audio length (vs 30s).

    Args:
        num_mels: int. The number of mel-frequency filters. Defaults to
            `128`.
        num_fft_bins: int. The size of the Fourier Transform in STFT.
            Defaults to `400`.
        stride: int. The distance between neighboring sliding window
            frames while computing STFT. Defaults to `160`.
        sampling_rate: int. The sample rate of the audio. Defaults to
            `16000`.
        max_audio_length: int. The length of each audio chunk in
            seconds. The input audio tensor will be padded/trimmed to
            `max_audio_length * sampling_rate`. Defaults to `300`.

    Examples:
    ```python
    converter = keras_hub.layers.Qwen3OmniAudioConverter.from_preset(
        "qwen3_omni_instruct"
    )
    audio = tf.ones((8000,), dtype="float32")
    mel_features = converter(audio)
    ```
    """

    backbone_cls = Qwen3OmniBackbone

    def __init__(
        self,
        num_mels=128,
        num_fft_bins=400,
        stride=160,
        sampling_rate=16000,
        max_audio_length=300,
        **kwargs,
    ):
        super().__init__(
            num_mels=num_mels,
            num_fft_bins=num_fft_bins,
            stride=stride,
            sampling_rate=sampling_rate,
            max_audio_length=max_audio_length,
            **kwargs,
        )
