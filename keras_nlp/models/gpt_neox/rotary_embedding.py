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
    def __init__(self, rotary_ndims, max_wavelength=10000):
        super().__init__()
        self.rotary_ndims = int(rotary_ndims)
        self.max_wavelength = max_wavelength

    def build(self, input_shape):
        super().build(input_shape)
        self.inverse_freq = self.add_weight(
            "inverse_freq", shape=(self.rotary_ndims // 2,), dtype=tf.float32
        )
        range = tf.range(
            start=0, limit=self.rotary_ndims, delta=2, dtype="float32"
        )

        self.inverse_freq.assign(
            1.0 / (self.max_wavelength ** (range / self.rotary_ndims))
        )

    @staticmethod
    def _apply_rotary_pos_emb(tensor, cos_emb, sin_emb):
        cos_emb = cos_emb[:, : tf.shape(tensor)[1], :, :]
        sin_emb = sin_emb[:, : tf.shape(tensor)[1], :, :]
        x1, x2 = tf.split(tensor, 2, axis=-1)
        half_rot_tensor = tf.concat((-x2, x1), axis=-1)
        ret = (tensor * cos_emb) + (half_rot_tensor * sin_emb)
        return ret

    def _compute_cos_sin_embedding(self, x, seq_dim=1):
        seq_len = tf.shape(x)[seq_dim]
        tensor = tf.range(seq_len, dtype=self.inverse_freq.dtype)
        freqs = tf.einsum("i, j -> ij", tensor, self.inverse_freq)
        embedding = tf.concat((freqs, freqs), axis=-1)[None, :, None, :]
        return tf.cos(embedding), tf.sin(embedding)

    def call(self, query, key):
        query_rot, query_pass = (
            query[..., : self.rotary_ndims],
            query[..., self.rotary_ndims :],
        )
        key_rot, key_pass = (
            key[..., : self.rotary_ndims],
            key[..., self.rotary_ndims :],
        )

        cos_emb, sin_emb = self._compute_cos_sin_embedding(key_rot, seq_dim=1)
        query_emb = self._apply_rotary_pos_emb(query_rot, cos_emb, sin_emb)
        key_emb = self._apply_rotary_pos_emb(key_rot, cos_emb, sin_emb)

        query = tf.concat((query_emb, query_pass), axis=-1)
        key = tf.concat((key_emb, key_pass), axis=-1)

        return query, key

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rotary_ndims": self.rotary_ndims,
                "max_wavelength": self.max_wavelength,
                "inverse_freq": self.inverse_freq,
            }
        )
