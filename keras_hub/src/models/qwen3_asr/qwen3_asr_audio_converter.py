import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.audio_converter import AudioConverter


@keras_hub_export("keras_hub.layers.Qwen3ASRAudioConverter")
class Qwen3ASRAudioConverter(AudioConverter):
    """Qwen3-ASR audio feature extraction layer.

    Converts raw audio waveforms into log-mel spectrogram features for the
    Qwen3-ASR audio encoder. The processing pipeline is:

    1. Pad or trim the waveform to a fixed length of
       ``max_audio_length * sampling_rate`` samples.
    2. Compute a short-time Fourier transform using a Hann window with
       ``center=True`` to produce a power spectrogram.
    3. Apply an HTK-scale mel filterbank with Slaney normalisation.
    4. Apply log10 compression with dynamic range compression.
    5. Normalize using Whisper-style formula: `(log_spec + 4.0) / 4.0`.

    Args:
        num_mels: int. Number of mel filterbank channels. Defaults to ``128``.
        num_fft_bins: int. FFT window length in samples, also used as the
            STFT sequence length. Defaults to ``400``.
        stride: int. STFT hop length in samples. Defaults to ``160``.
        sampling_rate: int. Expected sample rate of the input waveform in Hz.
            Defaults to ``16000``.
        max_audio_length: int. Maximum audio clip length in seconds. Inputs
            longer than this are trimmed; shorter inputs are zero-padded.
            Defaults to ``30``.
        min_frequency: float. Lower frequency bound for the mel filterbank in
            Hz. Defaults to ``0.0``.
        max_frequency: float. Upper frequency bound for the mel filterbank in
            Hz. Defaults to ``8000.0``.
        dither: float. Dither to add to audio before recording. Defaults
            to ``0.0``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras_hub.layers.AudioConverter``.
    """

    def __init__(
        self,
        num_mels=128,
        num_fft_bins=400,
        stride=160,
        sampling_rate=16000,
        max_audio_length=30,
        min_frequency=0.0,
        max_frequency=8000.0,
        dither=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mels = num_mels
        self.num_fft_bins = num_fft_bins
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.dither = dither

        # Total number of samples for the fixed-length output.
        self.num_samples = self.sampling_rate * self.max_audio_length

        # HTK mel filterbank: shape (num_fft_bins // 2 + 1, num_mels).
        self.mel_filters = self._get_mel_filters()

        # Periodic Hann window matching HF
        length = self.num_fft_bins + 1
        window = np.hanning(length)
        self.window = ops.convert_to_tensor(window[:-1], dtype="float32")

        self.built = True

    def _get_mel_filters(self):
        """Build an HTK-scale mel filterbank with Slaney normalisation.

        Returns:
            Float32 array of shape ``(num_fft_bins // 2 + 1, num_mels)``
            containing the triangular filterbank weights.
        """
        return self._build_slaney_mel_filters()

    def _build_slaney_mel_filters(self):
        # Replicated from Gemma4AudioConverter for consistency
        n_mels = self.num_mels
        n_fft = self.num_fft_bins
        sr = self.sampling_rate
        fmin = self.min_frequency
        fmax = self.max_frequency

        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        # Mel frequencies
        min_mel = hz_to_mel(fmin)
        max_mel = hz_to_mel(fmax)
        mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # FFT bin frequencies
        fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

        # Triangle filters
        filters = np.zeros((n_fft // 2 + 1, n_mels))
        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]

            # Upsolpe
            up_slope = (fft_freqs - left) / (center - left)
            # Downslope
            down_slope = (right - fft_freqs) / (right - center)

            filters[:, i] = np.maximum(0.0, np.minimum(up_slope, down_slope))

        # Slaney normalization
        enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
        filters = filters * enorm[None, :]

        return filters.astype("float32")

    def _extract_audio_features(self, audio):
        """Compute log-mel features from audio."""
        if self.dither != 0.0:
            # Add dither
            noise = ops.random.normal(ops.shape(audio), dtype=audio.dtype)
            audio = audio + self.dither * noise

        # STFT
        real, imag = ops.stft(
            audio,
            sequence_length=self.num_fft_bins,
            sequence_stride=self.stride,
            fft_length=self.num_fft_bins,
            window=self.window,
            center=True,
        )

        # Power spectrum
        power = ops.square(real) + ops.square(imag)

        # Mel filterbank matmul: (B, T, F) @ (F, M) -> (B, T, M)
        mel_filters = ops.cast(
            ops.convert_to_tensor(self.mel_filters), self.compute_dtype
        )
        mel_spec = ops.matmul(power, mel_filters)

        # Log10 compression
        mel_spec = ops.maximum(mel_spec, 1e-10)
        log_spec = ops.log10(mel_spec)

        # Dynamic range compression (Whisper style)
        max_val = ops.max(log_spec, axis=[1, 2], keepdims=True)
        log_spec = ops.maximum(log_spec, max_val - 8.0)

        # Normalization
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    def call(self, audio):
        if not isinstance(
            audio, (keras.KerasTensor, type(ops.convert_to_tensor(0)))
        ):
            audio = ops.convert_to_tensor(audio)

        rank_1_input = len(ops.shape(audio)) == 1
        if rank_1_input:
            audio = ops.expand_dims(audio, 0)

        # Pad with zeros to ensure it has at least self.num_samples.
        audio = ops.pad(audio, [[0, 0], [0, self.num_samples]], mode="constant")
        # Slice to exact length
        audio = audio[:, : self.num_samples]

        # Find the log mel spectrogram.
        log_spec = self._extract_audio_features(audio)

        # Slice time dimension to exactly self.num_samples // self.stride
        expected_frames = self.num_samples // self.stride
        log_spec = log_spec[:, :expected_frames, :]

        if rank_1_input:
            log_spec = ops.squeeze(log_spec, 0)
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
                "min_frequency": self.min_frequency,
                "max_frequency": self.max_frequency,
                "dither": self.dither,
            }
        )
        return config
