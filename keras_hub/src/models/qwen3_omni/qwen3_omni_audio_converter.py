import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.audio_converter import AudioConverter
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)


@keras_hub_export("keras_hub.layers.Qwen3OmniAudioConverter")
class Qwen3OmniAudioConverter(AudioConverter):
    """Audio preprocessing for Qwen3-Omni.

    Converts raw audio to log-mel spectrogram features compatible with the
    Qwen3-Omni audio encoder. This uses log-mel spectrogram via STFT
    feature extraction.

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
    audio = np.ones((8000,), dtype="float32")
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
        super().__init__(**kwargs)

        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        self.built = True

        self.num_mels = num_mels
        self.num_fft_bins = num_fft_bins
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.num_samples = self.sampling_rate * self.max_audio_length
        self.mel_filters = self._get_mel_filters()

    def audio_shape(self):
        """Returns the preprocessed size of a single audio sample."""
        return (self.max_audio_length, self.num_mels)

    def _get_mel_filters(self):
        """Computes the Mel filter bank weights.

        Returns:
            A numpy array of shape `(num_fft_bins // 2 + 1, num_mels)`
            containing the Mel filter bank weights.
        """
        dtype = np.float32
        weights = np.zeros(
            (self.num_mels, int(1 + self.num_fft_bins // 2)), dtype=dtype
        )

        # Center freqs of each FFT bin and mel bands.
        fftfreqs = np.fft.rfftfreq(
            n=self.num_fft_bins, d=1.0 / self.sampling_rate
        )
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = np.linspace(min_mel, max_mel, self.num_mels + 2)
        mels = np.asanyarray(mels)

        # Linear scale.
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # Nonlinear (log) scale.
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0

        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(
            logstep * (mels[log_t] - min_log_mel)
        )

        mel_f = freqs
        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(self.num_mels):
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        # scale to approx constant energy per channel.
        enorm = 2.0 / (mel_f[2 : self.num_mels + 2] - mel_f[: self.num_mels])
        weights *= enorm[:, np.newaxis]

        # Transpose to (num_fft_bins // 2 + 1, num_mels).
        return np.transpose(weights)

    def _extract_audio_features(self, audio):
        """Compute log-mel spectrogram from audio waveform.

        Uses `keras.ops.stft` with `center=True` which internally applies
        reflection padding, matching the Whisper feature extraction pipeline.

        Args:
            audio: Float tensor of shape (batch, num_samples).

        Returns:
            Log-mel spectrogram of shape (batch, num_frames, num_mels).
        """
        audio = ops.cast(audio, self.compute_dtype)

        real, imag = ops.stft(
            audio,
            sequence_length=self.num_fft_bins,
            sequence_stride=self.stride,
            fft_length=self.num_fft_bins,
            window="hann",
            center=True,
        )

        magnitudes = ops.square(real[:, :-1, :]) + ops.square(imag[:, :-1, :])

        # Apply mel filter bank.
        mel_filters = ops.cast(
            ops.convert_to_tensor(self.mel_filters), self.compute_dtype
        )
        mel_spec = ops.matmul(magnitudes, mel_filters)

        # Log-mel spectrogram with numerical stability.
        mel_spec = ops.maximum(mel_spec, 1e-10)
        log_spec = ops.log(mel_spec) / ops.log(
            ops.cast(ops.convert_to_tensor(10.0), self.compute_dtype)
        )

        # Dynamic range compression.
        max_val = ops.max(log_spec, axis=(1, 2))
        max_val_minus_eight = ops.expand_dims(max_val - 8.0, axis=(1, 2))
        log_spec = ops.maximum(log_spec, max_val_minus_eight)

        # Normalization.
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    def call(self, audio):
        audio = ops.convert_to_tensor(audio, dtype=self.compute_dtype)

        rank_1_input = len(ops.shape(audio)) == 1
        if rank_1_input:
            audio = ops.expand_dims(audio, 0)

        current_len = ops.shape(audio)[-1]
        if current_len < self.num_samples:
            pad_width = [[0, 0], [0, self.num_samples - current_len]]
            audio = ops.pad(audio, pad_width, mode="constant")
        else:
            audio = audio[:, : self.num_samples]

        log_spec = self._extract_audio_features(audio)

        if rank_1_input:
            log_spec = ops.squeeze(log_spec, axis=0)

        return log_spec

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_mels": self.num_mels,
                "num_fft_bins": self.num_fft_bins,
                "stride": self.stride,
                "sampling_rate": self.sampling_rate,
                "max_audio_length": self.max_audio_length,
            }
        )
        return config
