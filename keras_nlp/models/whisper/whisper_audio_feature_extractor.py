# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export

NUM_MELS = 80


@keras_nlp_export("keras_nlp.models.WhisperAudioFeatureExtractor")
class WhisperAudioFeatureExtractor(keras.layers.Layer):
    """
    Whisper audio feature extractor layer.

    This layer takes in a batch of audio tensors, and computes the log-mel
    spectrogram features for each audio tensor.

    The input audio tensor can either be of shape `(length_of_audio,)` or
    `(batch_size, length_of_audio)`. The output is a tensor of shape
    `(batch_size, num_frames, NUM_MELS)`, where `num_frames` is
    `(max_audio_length * sample_rate) / stride` and `NUM_MELS` is 80.

    Args:
        sample_rate: int, defaults to 16000. The sample rate of the audio.
        num_fft_bins: int, defaults to 400. The size of the Fourier Transform in
            STFT.
        stride: int, defaults to 160. The distance between neighboring
            sliding window frames while computing STFT.
        max_audio_length: int, defaults to 30. The length of each audio chunk in
            seconds. The input audio tensor will be padded/trimmed to
            `max_audio_length*sample_rate`.

    Examples:
    ```python

    # Load an audio file.
    audio_tensor = keras_nlp.utils.audio_utils.load_audio("path/to/audio.mp3")

    # Compute the log-mel spectrogram.
    whisper_audio_feature_extractor = keras_nlp.models.WhisperAudioFeatureExtractor()
    whisper_audio_feature_extractor(audio_tensor)

    # Compute the log-mel spectrogram for a batch of audio tensors.
    audio_tensor_1 = load_audio("path/to/audio_1.mp3")
    audio_tensor_2 = load_audio("path/to/audio_2.mp3")
    audio_tensor = tf.ragged.stack([audio_tensor_1, audio_tensor_2], axis=0)
    whisper_audio_feature_extractor(audio_tensor)
    ```
    """

    def __init__(
        self,
        sample_rate=16000,
        num_fft_bins=400,
        stride=160,
        max_audio_length=30,
        **kwargs,
    ):
        # Check dtype and provide a default.
        if "dtype" not in kwargs or kwargs["dtype"] is None:
            kwargs["dtype"] = tf.float32
        else:
            dtype = tf.dtypes.as_dtype(kwargs["dtype"])
            if not dtype.is_floating:
                raise ValueError(
                    f"dtype must be a floating type. Received: dtype={dtype}"
                )

        super().__init__(**kwargs)

        self.sample_rate = sample_rate
        self.num_fft_bins = num_fft_bins
        self.stride = stride
        self.max_audio_length = max_audio_length
        self.n_samples = self.sample_rate * self.max_audio_length

        # After transposition, `self.mel_filters`'s shape is
        # `(num_fft_bins // 2 + 1, NUM_MELS).`
        self.mel_filters = self._get_mel_filters()

    def _get_mel_filters(self):
        """
        Adapted from Hugging Face
        (https://github.com/huggingface/transformers/blob/v4.27.1/src/transformers/models/whisper/feature_extraction_whisper.py#L86)
        """

        # TODO: Convert to TensorFlow ops (if possible).

        dtype = np.float32
        # Initialize the weights
        weights = np.zeros(
            (NUM_MELS, int(1 + self.num_fft_bins // 2)), dtype=dtype
        )

        # Center freqs of each FFT bin
        fftfreqs = np.fft.rfftfreq(
            n=self.num_fft_bins, d=1.0 / self.sample_rate
        )

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = np.linspace(min_mel, max_mel, NUM_MELS + 2)

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

        for i in range(NUM_MELS):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : NUM_MELS + 2] - mel_f[:NUM_MELS])
        weights *= enorm[:, np.newaxis]

        weights = tf.transpose(tf.constant(weights))
        return weights

    def _extract_audio_features(self, audio):
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
            self.mel_filters,
        )

        def tf_log10(x):
            """
            Computes log base 10 of input tensor using TensorFlow's natural log operator.
            """
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        # Clamp the values to a minimum value of 1e-10. This is done to avoid
        # taking the log of 0, i.e., for numerical stability.
        mel_spec = tf.maximum(mel_spec, 1e-10)

        # Calculate the log mel spectrogram.
        log_spec = tf_log10(mel_spec)
        # Dynamic range compression.
        log_spec_shape = log_spec.shape.as_list()
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

    def call(self, audio):
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
        audio_shape[-1] = self.n_samples
        audio = audio.to_tensor(shape=audio_shape)

        # Find the log mel spectrogram.
        log_spec = self._extract_audio_features(audio)
        return log_spec

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sample_rate": self.sample_rate,
                "num_fft_bins": self.num_fft_bins,
                "stride": self.stride,
                "max_audio_length": self.max_audio_length,
            }
        )
        return config
