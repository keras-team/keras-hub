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


class RotaryEmbedding(keras.layers.Layer):
    def __init__(self, percentage, max_wavelength=10000):
        super().__init__()
        self.percentage = percentage
        self.max_wavelength = max_wavelength

    def _compute_cos_sin_embedding(self, x, rotary_ndims, seq_len):
        range = tf.range(start=0, limit=rotary_ndims, delta=2, dtype="float32")
        inverse_freq = 1.0 / (self.max_wavelength ** (range / rotary_ndims))
        tensor = tf.range(seq_len, dtype=inverse_freq.dtype)
        freqs = tf.einsum("i, j -> ij", tensor, inverse_freq)
        embedding = tf.concat((freqs, freqs), axis=-1)[None, :, None, :]
        return tf.cos(embedding), tf.sin(embedding)

    def call(self, inputs):

        shape = tf.shape(inputs)
        attn_head_size = shape[-1]
        seq_len = shape[1]

        rotary_ndims = tf.cast(attn_head_size, self.compute_dtype) * self.percentage

        cos_emb, sin_emb = self._compute_cos_sin_embedding(
            inputs, rotary_ndims, seq_len
        )

        return cos_emb, sin_emb

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "percentage": self.percentage,
                "max_wavelength": self.max_wavelength,
            }
        )
        return config
