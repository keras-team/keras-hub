# Copyright 2022 The KerasNLP Authors
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

"""Sinusoidal position embedding layer."""

import tensorflow as tf
from tensorflow import keras


class SinePositionEncoding(keras.layers.Layer):
    """Sinusoidal positional encoding layer.

    This layer calculates the position encoding as a mix of sine and cosine
    functions with geometrically increasing wavelengths. Defined and formulized
    in [Attention is All You Need](https://arxiv.org/abs/1706.03762).

    Usage:
        Takes as input an embedded token tensor, where the last dimension is
        assumed to be the feature dim, and the second to last dimension is
        assumed to be the sequence dimension. This layer will return a
        positional encoding the same size as the embedded token tensor, which
        can be added directly to the embedded token tensor.

    Args:
        min_frequency: The minimum frequency for each position embedding,
                       initalized to 1.0e-4.

    Example:
    ```python
    # create a simple embedding layer with sinusoidal positional encoding
    inputs = keras.Input((100, ), dtype=tf.float32)
    embedding = keras.layers.Embedding(input_dim=1000, output_dim=32)(inputs)
    positional_encoding = keras_nlp.layers.SinePositionEncoding()(embedding)
    outputs = embedding + positional_encoding
    ```

    References:
      [Attention is All You Need](https://arxiv.org/abs/1706.03762)
    """

    def __init__(
        self,
        min_frequency: float = 1.0e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_frequency = min_frequency

    def call(self, inputs):
        """Forward pass of the SinePositionEncoding.

        Args:
            inputs: a Tensor.

        Returns:
            A Tensor of the same shape as the `inputs` containing the sinusoidal
            positional encoding weights for the input sequence.
        """
        input_shape = tf.shape(inputs)
        # length of sequence is the second last dimension of the inputs
        seq_length = input_shape[-2]
        hidden_size = input_shape[-1]
        position = tf.cast(tf.range(seq_length), self.compute_dtype)
        min_freq = tf.cast(self.min_frequency, dtype=self.compute_dtype)
        timescales = tf.pow(
            min_freq,
            tf.cast(2 * (tf.range(hidden_size) // 2), self.compute_dtype)
            / tf.cast(hidden_size, self.compute_dtype),
        )
        angles = tf.expand_dims(position, 1) * tf.expand_dims(timescales, 0)
        # even indices are sine, odd are cosine
        cos_mask = tf.cast(tf.range(hidden_size) % 2, self.compute_dtype)
        sin_mask = 1 - cos_mask
        # embedding shape is [seq_length, hidden_size]
        positional_encodings = (
            tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
        )

        return tf.broadcast_to(positional_encodings, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "min_frequency": self.min_frequency,
            }
        )
        return config
