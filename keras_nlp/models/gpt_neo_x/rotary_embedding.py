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
    def __init__(self, max_wavelength=10000, **kwargs):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength

    def _apply_rotary_pos_emb(self, tensor, cos_emb, sin_emb):
        cos_emb = cos_emb[:, : tf.shape(tensor)[1], :, :]
        sin_emb = sin_emb[:, : tf.shape(tensor)[1], :, :]

        x1, x2 = tf.split(tensor, 2, axis=-1)
        half_rot_tensor = tf.concat((-x2, x1), axis=-1)

        return (tensor * cos_emb) + (half_rot_tensor * sin_emb)

    def _compute_cos_sin_embedding(self, x, rotary_dim, seq_dim=1):
        seq_len = tf.shape(x)[seq_dim]
        rotary_dim = tf.cast(rotary_dim, "float32")

        range = tf.range(start=0, limit=rotary_dim, delta=2, dtype="float32")
        inverse_freq = 1.0 / (self.max_wavelength ** (range / rotary_dim))

        tensor = tf.range(seq_len, dtype=inverse_freq.dtype)
        freqs = tf.einsum("i, j -> ij", tensor, inverse_freq)
        embedding = tf.concat((freqs, freqs), axis=-1)[None, :, None, :]

        return tf.cos(embedding), tf.sin(embedding)

    def call(self, query, key):
        rotary_dim = tf.shape(query)[-1]

        cos_emb, sin_emb = self._compute_cos_sin_embedding(
            query, rotary_dim, seq_dim=1
        )

        query = self._apply_rotary_pos_emb(query, cos_emb, sin_emb)
        key = self._apply_rotary_pos_emb(key, cos_emb, sin_emb)

        return query, key

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
            }
        )
        return config
