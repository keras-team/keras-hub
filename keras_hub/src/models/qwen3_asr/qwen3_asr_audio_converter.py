import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.audio_converter import AudioConverter
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.layers.Qwen3ASRAudioConverter")
class Qwen3ASRAudioConverter(AudioConverter):
    """Audio converter for Qwen3-ASR models.

    This layer converts raw audio waveforms into log-mel spectrogram features
    suitable for the Qwen3-ASR audio encoder. It computes a Short-Time Fourier
    Transform (STFT), applies a mel filterbank, and performs log-mel
    normalization.

    The input audio should be sampled at 16kHz. The output is a tensor of shape
    ``(batch_size, num_frames, num_mels)`` where ``num_frames`` depends on the
    audio length and ``num_mels`` defaults to 128.

    Args:
        num_mels: int. Number of mel frequency bins. Defaults to ``128``.
        num_fft_bins: int. FFT window size for STFT. Defaults to ``400``.
        stride: int. Hop length between STFT frames. Defaults to ``160``.
        sampling_rate: int. Expected audio sample rate in Hz.
            Defaults to ``16000``.
        max_audio_length: int. Maximum audio duration in seconds. Audio
            longer than this is truncated. Defaults to ``30``.

    Examples:
    ```python
    audio = np.random.randn(16000).astype("float32")  # 1 second at 16kHz
    converter = keras_hub.layers.Qwen3ASRAudioConverter()
    features = converter(audio)
    # features.shape == (1, num_frames, 128)
    ```
    """

    backbone_cls = Qwen3ASRBackbone

    def __init__(
        self,
        num_mels=128,
        num_fft_bins=400,
        stride=160,
        sampling_rate=16000,
        max_audio_length=30,
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
        """Compute mel filterbank matrix.

        Adapted from the Hugging Face Whisper feature extractor.
        """
        dtype = np.float32
        weights = np.zeros(
            (self.num_mels, int(1 + self.num_fft_bins // 2)), dtype=dtype
        )
        fftfreqs = np.fft.rfftfreq(
            n=self.num_fft_bins, d=1.0 / self.sampling_rate
        )

        min_mel = 0.0
        max_mel = 45.245640471924965
        mels = np.linspace(min_mel, max_mel, self.num_mels + 2)
        mels = np.asanyarray(mels)

        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

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

        enorm = 2.0 / (mel_f[2 : self.num_mels + 2] - mel_f[: self.num_mels])
        weights *= enorm[:, np.newaxis]
        weights = np.transpose(weights)
        return tf.constant(weights, dtype=self.compute_dtype)

    def _extract_audio_features(self, audio):
        audio = tf.cast(audio, self.compute_dtype)
        audio = tf.pad(
            audio,
            paddings=[
                [0, 0],
                [self.num_fft_bins // 2, self.num_fft_bins // 2],
            ],
            mode="REFLECT",
        )

        stft = tf.signal.stft(
            audio,
            frame_length=self.num_fft_bins,
            frame_step=self.stride,
            fft_length=self.num_fft_bins,
        )
        magnitudes = tf.square(tf.abs(stft[:, :-1, :]))
        mel_spec = tf.matmul(magnitudes, self.mel_filters)

        def tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        mel_spec = tf.maximum(mel_spec, 1e-10)
        log_spec = tf_log10(mel_spec)

        log_spec_shape = tf.shape(log_spec)
        max_value_minus_eight = tf.math.subtract(
            tf.math.reduce_max(log_spec, axis=[1, 2]),
            tf.cast(8, dtype=log_spec.dtype),
        )
        max_value_minus_eight = tf.expand_dims(max_value_minus_eight, axis=1)
        max_value_minus_eight = tf.repeat(
            max_value_minus_eight,
            repeats=log_spec_shape[1] * log_spec_shape[2],
            axis=1,
        )
        max_value_minus_eight = tf.reshape(
            max_value_minus_eight, shape=log_spec_shape
        )
        log_spec = tf.maximum(log_spec, max_value_minus_eight)

        type_cast_four = tf.cast(4, dtype=log_spec.dtype)
        log_spec = tf.math.divide(
            tf.math.add(log_spec, type_cast_four),
            type_cast_four,
        )
        return log_spec

    def call(self, audio):
        if not isinstance(audio, (tf.Tensor, tf.RaggedTensor)):
            audio = tf.convert_to_tensor(audio)

        rank_1_input = audio.shape.rank == 1
        if rank_1_input:
            audio = tf.expand_dims(audio, 0)

        # Strip channel dim if present: (batch, time, 1) -> (batch, time).
        if audio.shape.rank == 3:
            audio = tf.squeeze(audio, axis=-1)

        if isinstance(audio, tf.Tensor):
            audio = tf.RaggedTensor.from_tensor(audio)

        audio_shape = audio.shape.as_list()
        audio_shape[-1] = self.num_samples
        audio = audio.to_tensor(shape=audio_shape)

        log_spec = self._extract_audio_features(audio)
        if rank_1_input:
            log_spec = tf.squeeze(log_spec, 0)
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
