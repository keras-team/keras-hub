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
    def __init__(self, rotary_percentage, max_wavelength=10000):
        super().__init__()
        self.rotary_percentage = rotary_percentage
        self.max_wavelength = max_wavelength

    def _apply_rotary_pos_emb(self, tensor, cos_emb, sin_emb):
        cos_emb = cos_emb[:, : tf.shape(tensor)[1], :, :]
        sin_emb = sin_emb[:, : tf.shape(tensor)[1], :, :]
        x1, x2 = tf.split(tensor, 2, axis=-1)
        half_rot_tensor = tf.concat((-x2, x1), axis=-1)
        ret = (tensor * cos_emb) + (half_rot_tensor * sin_emb)
        return ret

    def _compute_cos_sin_embedding(self, x, rotary_ndims, seq_dim=1):
        seq_len = tf.shape(x)[seq_dim]
        rotary_ndims = tf.cast(rotary_ndims, tf.float32)
        range = tf.range(start=0, limit=rotary_ndims, delta=2, dtype="float32")
        inverse_freq = 1.0 / (self.max_wavelength ** (range / rotary_ndims))
        tensor = tf.range(seq_len, dtype=inverse_freq.dtype)
        freqs = tf.einsum("i, j -> ij", tensor, inverse_freq)
        embedding = tf.concat((freqs, freqs), axis=-1)[None, :, None, :]
        return tf.cos(embedding), tf.sin(embedding)

    def call(self, query, key):
        attn_head_size = tf.shape(query)[-1]
        rotary_ndims = tf.cast(
            tf.cast(attn_head_size, self.compute_dtype)
            * self.rotary_percentage,
            tf.int32,
        )

        query_rot, query_pass = (
            query[..., :rotary_ndims],
            query[..., rotary_ndims:],
        )
        key_rot, key_pass = (
            key[..., :rotary_ndims],
            key[..., rotary_ndims:],
        )

        cos_emb, sin_emb = self._compute_cos_sin_embedding(
            key_rot, rotary_ndims, seq_dim=1
        )
        query_emb = self._apply_rotary_pos_emb(query_rot, cos_emb, sin_emb)
        key_emb = self._apply_rotary_pos_emb(key_rot, cos_emb, sin_emb)

        query = tf.concat((query_emb, query_pass), axis=-1)
        key = tf.concat((key_emb, key_pass), axis=-1)

        return query, key

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rotary_percentage": self.rotary_percentage,
                "max_wavelength": self.max_wavelength,
            }
        )
        return config
