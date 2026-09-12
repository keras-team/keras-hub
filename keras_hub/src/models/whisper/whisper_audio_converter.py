import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.audio_converter import AudioConverter
from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.utils.audio_utils import frame_signal
from keras_hub.src.utils.audio_utils import hann_window
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.layers.WhisperAudioConverter")
class WhisperAudioConverter(AudioConverter):
    """Whisper audio converter layer.

    This layer takes in a batch of audio tensors, and computes the log-mel
    spectrogram features for each audio tensor.

    The input audio tensor can either be of shape `(length_of_audio,)` or
    `(batch_size, length_of_audio)`. The output is a tensor of shape
    `(batch_size, num_frames, num_mels)`, where `num_frames` is
    `(max_audio_length * sampling_rate) / stride`.

    Args:
        num_mels: int. The number of mel-frequency filters. Defaults to `80`.
        num_fft_bins: int. The size of the Fourier Transform in STFT.
            Defaults to `400`.
        stride: int. The distance between neighboring
            sliding window frames while computing STFT.
            Defaults to `160`.
        sampling_rate: int. The sample rate of the audio. Defaults to `16000`.
        max_audio_length: int. The length of each audio chunk in
            seconds. The input audio tensor will be padded/trimmed to
            `max_audio_length * sampling_rate`. Defaults to `30`.

    Examples:
    ```python
    audio_tensor = tf.ones((8000,), dtype="float32")

    # Compute the log-mel spectrogram.
    audio_converter = keras_hub.layers.WhisperAudioConverter.from_preset(
        "whisper_base_en",
    )
    audio_converter(audio_tensor)

    # Compute the log-mel spectrogram for a batch of audio tensors.
    audio_tensor_1 = tf.ones((8000,), dtype="float32")
    audio_tensor_2 = tf.ones((10000,), dtype="float32")
    audio_tensor = tf.ragged.stack([audio_tensor_1, audio_tensor_2], axis=0)
    audio_converter(audio_tensor)
    ```
    """

    backbone_cls = WhisperBackbone

    def __init__(
        self,
        num_mels=80,
        num_fft_bins=400,
        stride=160,
        sampling_rate=16000,
        max_audio_length=30,
        **kwargs,
    ):
        _allow_python_workflow = kwargs.pop("_allow_python_workflow", True)
        super().__init__(
            _allow_python_workflow=_allow_python_workflow, **kwargs
        )

        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        self.built = True

        self.num_mels = num_mels
        self.num_fft_bins = num_fft_bins
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.num_samples = self.sampling_rate * self.max_audio_length

        # After transposition, `self.mel_filters`'s shape is
        # `(num_fft_bins // 2 + 1, num_mels).`
        self.mel_filters = self._get_mel_filters()

    def audio_shape(self):
        """Returns the preprocessed size of a single audio sample."""
        return (self.max_audio_length, self.num_mels)

    def _get_mel_filters(self):
        """
        Adapted from Hugging Face
        (https://github.com/huggingface/transformers/blob/v4.27.1/src/transformers/models/whisper/feature_extraction_whisper.py#L86)
        """

        # This is deliberately computed with NumPy. The filterbank is a
        # constant, and NumPy keeps it usable without TensorFlow installed.

        dtype = np.float32
        # Initialize the weights
        weights = np.zeros(
            (self.num_mels, int(1 + self.num_fft_bins // 2)), dtype=dtype
        )

        # Center freqs of each FFT bin
        fftfreqs = np.fft.rfftfreq(
            n=self.num_fft_bins, d=1.0 / self.sampling_rate
        )

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = np.linspace(min_mel, max_mel, self.num_mels + 2)

        mels = np.asanyarray(mels)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(
            logstep * (mels[log_t] - min_log_mel)
        )

        mel_f = freqs

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(self.num_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : self.num_mels + 2] - mel_f[: self.num_mels])
        weights *= enorm[:, np.newaxis]

        weights = np.transpose(weights)
        # Keep the filterbank as NumPy. Both the TF and the Python paths cast
        # it to the compute dtype when it is used, and a NumPy array can be
        # pickled into Grain worker processes.
        return weights.astype(dtype)

    def _extract_audio_features(self, audio):
        audio = tf.cast(audio, self.compute_dtype)
        # Use "reflection" padding - `tf.signal.stft` uses symmetric padding
        # internally.
        audio = tf.pad(
            audio,
            paddings=[[0, 0], [self.num_fft_bins // 2, self.num_fft_bins // 2]],
            mode="REFLECT",
        )

        # Compute the mel spectrogram.
        stft = tf.signal.stft(
            audio,
            frame_length=self.num_fft_bins,
            frame_step=self.stride,
            fft_length=self.num_fft_bins,
        )
        magnitudes = tf.square(tf.abs(stft[:, :-1, :]))

        mel_spec = tf.matmul(
            magnitudes,
            tf.constant(self.mel_filters, dtype=self.compute_dtype),
        )

        def tf_log10(x):
            """Computes log base 10 of input tensor using TensorFlow."""
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        # Clamp the values to a minimum value of 1e-10. This is done to avoid
        # taking the log of 0, i.e., for numerical stability.
        mel_spec = tf.maximum(mel_spec, 1e-10)

        # Calculate the log mel spectrogram.
        log_spec = tf_log10(mel_spec)
        # Dynamic range compression.
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
        # Normalization.
        type_cast_four = tf.cast(4, dtype=log_spec.dtype)
        log_spec = tf.math.divide(
            tf.math.add(log_spec, type_cast_four),
            type_cast_four,
        )

        return log_spec

    def _extract_audio_features_python(self, audio):
        """NumPy equivalent of `_extract_audio_features`."""
        dtype = self.compute_dtype
        # Use "reflection" padding - `tf.signal.stft` uses symmetric padding
        # internally.
        pad_width = self.num_fft_bins // 2
        audio = np.pad(audio, ((0, 0), (pad_width, pad_width)), mode="reflect")

        # Compute the mel spectrogram.
        frames = frame_signal(audio, self.num_fft_bins, self.stride)
        frames = frames * hann_window(self.num_fft_bins, dtype="float64")
        stft = np.fft.rfft(frames, n=self.num_fft_bins, axis=-1)
        magnitudes = np.square(np.abs(stft[:, :-1, :]))

        mel_spec = np.matmul(magnitudes, self.mel_filters)

        # Clamp the values to a minimum value of 1e-10. This is done to avoid
        # taking the log of 0, i.e., for numerical stability.
        mel_spec = np.maximum(mel_spec, 1e-10)

        # Calculate the log mel spectrogram.
        log_spec = np.log10(mel_spec)
        # Dynamic range compression.
        max_value_minus_eight = np.max(log_spec, axis=(1, 2), keepdims=True) - 8
        log_spec = np.maximum(log_spec, max_value_minus_eight)
        # Normalization.
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec.astype(dtype)

    def _to_batched_array(self, audio):
        """Convert any supported input to a dense `(batch, num_samples)` array.

        Returns the padded/trimmed waveform batch and whether the input was
        unbatched.
        """
        if tf is not None and isinstance(audio, tf.RaggedTensor):
            rows, unbatched = audio.to_list(), False
        else:
            try:
                array = convert_to_numpy(audio)
            except ValueError:
                # A ragged list of waveforms cannot be made rectangular.
                array = None
            if array is None or array.dtype == object:
                rows, unbatched = list(audio), False
            elif array.ndim == 1:
                rows, unbatched = [array], True
            else:
                rows, unbatched = list(array), False

        batch = np.zeros(
            (len(rows), self.num_samples), dtype=self.compute_dtype
        )
        for i, row in enumerate(rows):
            row = np.reshape(convert_to_numpy(row), (-1,))
            row = row[: self.num_samples]
            batch[i, : row.shape[0]] = row
        return batch, unbatched

    def _call_python(self, audio):
        audio, rank_1_input = self._to_batched_array(audio)

        # Find the log mel spectrogram.
        log_spec = self._extract_audio_features_python(audio)
        if rank_1_input:
            log_spec = np.squeeze(log_spec, 0)
        return convert_preprocessing_outputs_python(log_spec)

    @preprocessing_function
    def _call_tf(self, audio):
        if not isinstance(audio, (tf.Tensor, tf.RaggedTensor)):
            audio = tf.convert_to_tensor(audio)

        rank_1_input = audio.shape.rank == 1
        if rank_1_input:
            audio = tf.expand_dims(audio, 0)

        # Convert the tensor to a Ragged Tensor.
        if isinstance(audio, tf.Tensor):
            audio = tf.RaggedTensor.from_tensor(audio)

        # Pad audio.
        audio_shape = audio.shape.as_list()
        audio_shape[-1] = self.num_samples
        audio = audio.to_tensor(shape=audio_shape)

        # Find the log mel spectrogram.
        log_spec = self._extract_audio_features(audio)
        if rank_1_input:
            log_spec = tf.squeeze(log_spec, 0)
        return log_spec

    def call(self, audio):
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
            }
        )
        return config
