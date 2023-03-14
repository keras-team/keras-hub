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
import tensorflow as tf
from tensorflow import keras

try:
    import librosa
except ImportError:
    librosa = None


NUM_MELS = 80


class WhisperAudioFeatureExtractor(keras.layers.Layer):
    """
    Whisper audio feature extractor layer.

    This layer takes in a batch of audio tensors, and computes the log-mel
    spectrogram features for each audio tensor.

    Args:
        sample_rate: int, defaults to 16000. The sample rate of the audio.
        n_fft: int, defaults to 400. The size of the Fourier Transform in STFT.
        hop_length: int, defaults to 160. The distance between neighboring
            sliding window frames while computing STFT.
        chunk_length: int, defaults to 30. The length of each audio chunk in
            seconds. The input audio tensor will be padded/trimmed to
            `chunk_length*sample_rate`.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        chunk_length=30,
        **kwargs,
    ):
        if librosa is None:
            raise ImportError(
                f"{self.__class__.__name__} requires the `librosa` "
                "package. Please install it with `pip install librosa`."
            )

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
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = self.sample_rate * self.chunk_length

        # After transposition, `self.mel_filters`'s shape is
        # `(n_fft // 2 + 1, NUM_MELS).`
        self.mel_filters = tf.constant(
            librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=NUM_MELS,
            ),
            dtype=self.dtype,
        )
        self.mel_filters = tf.transpose(self.mel_filters)

    def _extract_audio_features(self, audio):
        # Use "reflection" padding - `tf.signal.stft` uses symmetric padding
        # internally.
        audio = tf.pad(
            audio,
            paddings=[[0, 0], [self.n_fft // 2, self.n_fft // 2]],
            mode="REFLECT",
        )

        # Compute the mel spectrogram.
        stft = tf.signal.stft(
            audio,
            frame_length=self.n_fft,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
        )
        magnitudes = tf.square(tf.abs(stft[:, :-1, :]))

        mel_spec = tf.matmul(
            magnitudes,
            self.mel_filters,
        )

        def tf_log10(x):
            """
            Computes log base 10 of input tensor.

            TensorFlow does not have a native implementation of log base 10, but
            does have a log base `e` (`tf.math.log`). Hence, this short
            workaround.
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
        max_value_minus_eight = tf.math.subtract(
            tf.math.reduce_max(log_spec), tf.cast(8, dtype=log_spec.dtype)
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
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "chunk_length": self.chunk_length,
            }
        )
        return config
