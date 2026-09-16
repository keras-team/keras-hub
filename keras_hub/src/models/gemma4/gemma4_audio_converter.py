import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.audio_converter import AudioConverter
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.utils.audio_utils import hann_window
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.layers.Gemma4AudioConverter")
class Gemma4AudioConverter(AudioConverter):
    """Gemma4 audio feature extraction layer.

    Converts raw audio waveforms into log-mel spectrogram features for the
    Gemma4 USM audio encoder. The processing pipeline is:

    1. Pad or trim the waveform to a fixed length of
       ``max_audio_length * sampling_rate`` samples.
    2. Compute a short-time Fourier transform using a Hann window with
       ``center=True`` to produce a power spectrogram.
    3. Apply an HTK-scale mel filterbank with Slaney normalisation.
    4. Apply log compression with a configurable floor value.
    5. Optionally subtract per-bin mean and divide by per-bin standard
       deviation.

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
        mel_floor: float. Minimum value applied before the log compression for
            numerical stability. Defaults to ``1e-5``.
        per_bin_mean: list[float] or None. Per-channel mean subtracted after
            log compression. ``None`` disables mean subtraction.
            Defaults to ``None``.
        per_bin_stddev: list[float] or None. Per-channel standard deviation
            used to scale the output after mean subtraction. ``None`` disables
            scaling. Defaults to ``None``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras_hub.layers.AudioConverter``.

    Call arguments:
        audio: array of shape ``(num_samples,)`` or
            ``(batch_size, num_samples)``. Raw mono-channel audio waveform(s)
            at ``sampling_rate`` Hz.

    Returns:
        Log-mel spectrogram of shape ``(num_frames, num_mels)`` for a 1-D
        input, or ``(batch_size, num_frames, num_mels)`` for a 2-D input,
        where ``num_frames = num_samples // stride``.

    Examples:

    ```python
    import numpy as np
    import keras_hub

    # Single waveform (1 second at 16 kHz).
    waveform = np.random.randn(16000).astype("float32")
    converter = keras_hub.layers.Gemma4AudioConverter()
    features = converter(waveform)
    print(features.shape)  # (100, 128)

    # Batched waveforms.
    batch = np.random.randn(4, 16000).astype("float32")
    features = converter(batch)
    print(features.shape)  # (4, 100, 128)
    ```
    """

    backbone_cls = Gemma4Backbone

    def __init__(
        self,
        num_mels=128,
        num_fft_bins=512,
        stride=160,
        sampling_rate=16000,
        max_audio_length=30,
        min_frequency=0.0,
        max_frequency=8000.0,
        mel_floor=1e-5,
        per_bin_mean=None,
        per_bin_stddev=None,
        frame_length=320,
        **kwargs,
    ):
        _allow_python_workflow = kwargs.pop("_allow_python_workflow", True)
        super().__init__(
            _allow_python_workflow=_allow_python_workflow, **kwargs
        )

        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

        self.num_mels = num_mels
        self.num_fft_bins = num_fft_bins
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.mel_floor = float(mel_floor)
        self.frame_length = frame_length or num_fft_bins
        # Store as Python lists so they round-trip through get_config().
        self.per_bin_mean = (
            list(per_bin_mean) if per_bin_mean is not None else None
        )
        self.per_bin_stddev = (
            list(per_bin_stddev) if per_bin_stddev is not None else None
        )

        # Total number of samples for the fixed-length output.
        self.num_samples = self.sampling_rate * self.max_audio_length

        # HTK mel filterbank: shape (num_fft_bins // 2 + 1, num_mels).
        self.mel_filters = self._get_mel_filters()

        # Periodic Hann window matching HF. Kept as NumPy so the layer holds
        # no backend tensors (Grain pickles layers into worker processes).
        self.window = hann_window(
            self.frame_length, periodic=True, dtype="float32"
        )

        # Precompute indices for manual framing
        num_frames = self.num_samples // self.stride
        one_frame_indices = np.arange(self.frame_length)
        start_indices = np.arange(num_frames) * self.stride
        indices = start_indices[:, None] + one_frame_indices[None, :]
        self.indices = indices.astype("int32")

        self.built = True

    def _get_mel_filters(self):
        """Build an HTK-scale mel filterbank with Slaney normalisation.

        Returns:
            Float32 array of shape ``(num_fft_bins // 2 + 1, num_mels)``
            containing the triangular filterbank weights.
        """
        n = self.num_mels

        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(self.min_frequency)
        mel_max = hz_to_mel(self.max_frequency)
        mels = np.linspace(mel_min, mel_max, n + 2)
        freqs = mel_to_hz(mels)  # (n + 2,) Hz breakpoints

        # FFT bin centre frequencies.
        fft_freqs = np.fft.rfftfreq(
            self.num_fft_bins, d=1.0 / self.sampling_rate
        )  # (num_fft_bins // 2 + 1,)

        # Triangular filters: shape (num_fft_bins // 2 + 1, n).
        freq_diff = np.diff(freqs)  # (n + 1,)
        slopes = (
            freqs[np.newaxis, :] - fft_freqs[:, np.newaxis]
        )  # (n_fft, n + 2)
        lower = -slopes[:, :-2] / freq_diff[:-1]
        upper = slopes[:, 2:] / freq_diff[1:]
        filters = np.maximum(0.0, np.minimum(lower, upper))  # (n_fft, n)

        return filters.astype(np.float32)

    def _extract_audio_features(self, audio):
        """Convert a fixed-length waveform batch to log-mel features.

        Args:
            audio: Tensor of shape ``(batch_size, num_samples)``.

        Returns:
            Tensor of shape ``(batch_size, num_frames, num_mels)``.
        """
        audio = ops.cast(audio, self.compute_dtype)

        # Pad left by frame_length // 2 for semicausal padding
        pad_left = self.frame_length // 2
        paddings = [[0, 0], [pad_left, 0]]
        audio = ops.pad(audio, paddings, mode="constant")

        # `self.indices` and `self.window` are NumPy constants; materialize
        # them as tensors for the graph/backend path.
        indices = ops.convert_to_tensor(self.indices)
        window = ops.cast(
            ops.convert_to_tensor(self.window), self.compute_dtype
        )

        def true_fn():
            max_idx = ops.shape(audio)[1]
            safe_indices = ops.minimum(indices, ops.maximum(max_idx - 1, 0))
            frames = ops.take(audio, safe_indices, axis=1)
            out_of_bounds = indices >= max_idx
            out_of_bounds = ops.expand_dims(out_of_bounds, axis=0)
            return ops.where(out_of_bounds, ops.cast(0.0, frames.dtype), frames)

        def false_fn():
            batch_size = ops.shape(audio)[0]
            num_frames = self.num_samples // self.stride
            return ops.zeros(
                (batch_size, num_frames, self.frame_length), dtype=audio.dtype
            )

        max_idx = ops.shape(audio)[1]
        frames = ops.cond(max_idx > 0, true_fn, false_fn)

        # Apply window
        frames = frames * window

        # Zero-pad to fft_length
        padding = self.num_fft_bins - self.frame_length
        if padding > 0:
            frames_padded = ops.pad(frames, [[0, 0], [0, 0], [0, padding]])
        else:
            frames_padded = frames[..., : self.num_fft_bins]

        # FFT
        real, imag = ops.fft((frames_padded, ops.zeros_like(frames_padded)))

        # Truncate to first half of bins (rfft equivalent)
        real = real[..., : self.num_fft_bins // 2 + 1]
        imag = imag[..., : self.num_fft_bins // 2 + 1]

        # Magnitude spectrum
        magnitudes = ops.sqrt(ops.square(real) + ops.square(imag))

        # Mel filterbank matmul: (batch, num_frames, fft_bins) @
        #   (fft_bins, num_mels) → (batch, num_frames, num_mels)
        mel_filters = ops.cast(
            ops.convert_to_tensor(self.mel_filters), self.compute_dtype
        )
        mel_spec = ops.matmul(magnitudes, mel_filters)

        # Log compression.
        mel_floor = ops.cast(self.mel_floor, self.compute_dtype)
        log_spec = ops.log(mel_spec + mel_floor)

        # Optional per-bin mean / stddev normalisation.
        if self.per_bin_mean is not None:
            mean = ops.cast(
                ops.convert_to_tensor(
                    np.array(self.per_bin_mean, dtype=np.float32).reshape(
                        1, 1, -1
                    )
                ),
                self.compute_dtype,
            )
            log_spec = log_spec - mean
        if self.per_bin_stddev is not None:
            stddev = ops.cast(
                ops.convert_to_tensor(
                    np.array(self.per_bin_stddev, dtype=np.float32).reshape(
                        1, 1, -1
                    )
                ),
                self.compute_dtype,
            )
            log_spec = log_spec / stddev

        return log_spec

    def audio_shape(self):
        """Returns the output shape of a single preprocessed audio sample."""
        num_frames = self.num_samples // self.stride
        return (num_frames, self.num_mels)

    def _extract_audio_features_python(self, audio):
        """NumPy equivalent of `_extract_audio_features`.

        Args:
            audio: NumPy array of shape ``(batch_size, num_samples)``.

        Returns:
            NumPy array of shape ``(batch_size, num_frames, num_mels)``.
        """
        # Pad left by frame_length // 2 for semicausal padding
        pad_left = self.frame_length // 2
        audio = np.pad(audio, ((0, 0), (pad_left, 0)), mode="constant")

        # Framing with zeros for out of bounds indices.
        max_idx = audio.shape[1]
        safe_indices = np.minimum(self.indices, max(max_idx - 1, 0))
        frames = audio[:, safe_indices]
        frames = np.where(self.indices[None, ...] >= max_idx, 0.0, frames)

        # Apply window
        frames = frames * self.window

        # Zero-pad to fft_length
        padding = self.num_fft_bins - self.frame_length
        if padding > 0:
            frames = np.pad(frames, ((0, 0), (0, 0), (0, padding)))
        else:
            frames = frames[..., : self.num_fft_bins]

        # Magnitude spectrum of the real FFT.
        magnitudes = np.abs(np.fft.rfft(frames, n=self.num_fft_bins, axis=-1))

        # Mel filterbank matmul: (batch, num_frames, fft_bins) @
        #   (fft_bins, num_mels) → (batch, num_frames, num_mels)
        mel_spec = np.matmul(magnitudes, self.mel_filters)

        # Log compression.
        log_spec = np.log(mel_spec + self.mel_floor)

        # Optional per-bin mean / stddev normalisation.
        if self.per_bin_mean is not None:
            mean = np.array(self.per_bin_mean, dtype="float32").reshape(
                1, 1, -1
            )
            log_spec = log_spec - mean
        if self.per_bin_stddev is not None:
            stddev = np.array(self.per_bin_stddev, dtype="float32").reshape(
                1, 1, -1
            )
            log_spec = log_spec / stddev

        return log_spec.astype(self.compute_dtype)

    def _call_python(self, audio):
        audio = convert_to_numpy(audio)
        audio = np.asarray(audio, dtype=self.compute_dtype)
        rank_1_input = audio.ndim == 1
        if rank_1_input:
            audio = np.expand_dims(audio, axis=0)

        # Trim to num_samples, then zero-pad any remaining deficit.
        audio = audio[:, : self.num_samples]
        padding = self.num_samples - audio.shape[1]
        audio = np.pad(audio, ((0, 0), (0, padding)))

        log_spec = self._extract_audio_features_python(audio)

        if rank_1_input:
            log_spec = np.squeeze(log_spec, axis=0)

        return convert_preprocessing_outputs_python(log_spec)

    @preprocessing_function
    def _call_tf(self, audio):
        audio = ops.convert_to_tensor(audio, dtype=self.compute_dtype)
        rank_1_input = len(ops.shape(audio)) == 1
        if rank_1_input:
            audio = ops.expand_dims(audio, axis=0)

        # Trim to num_samples, then zero-pad any remaining deficit.
        audio = audio[:, : self.num_samples]
        current_len = ops.shape(audio)[1]
        padding = self.num_samples - current_len
        audio = ops.pad(audio, [[0, 0], [0, padding]])

        log_spec = self._extract_audio_features(audio)

        if rank_1_input:
            log_spec = ops.squeeze(log_spec, axis=0)

        return log_spec

    def call(self, audio):
        """Convert raw waveform(s) to log-mel spectrogram features.

        Args:
            audio: array of shape ``(num_samples,)`` or
                ``(batch_size, num_samples)``.

        Returns:
            Log-mel spectrogram of shape ``(num_frames, num_mels)`` or
            ``(batch_size, num_frames, num_mels)``.
        """
        if not self._allow_python_workflow or in_tf_function():
            return self._call_tf(audio)
        else:
            return self._call_python(audio)

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
                "mel_floor": self.mel_floor,
                "per_bin_mean": self.per_bin_mean,
                "per_bin_stddev": self.per_bin_stddev,
                "frame_length": self.frame_length,
            }
        )
        return config
