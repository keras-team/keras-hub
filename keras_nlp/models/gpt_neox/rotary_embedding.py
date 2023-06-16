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
    def __init__(self, dim, rotary_emb_base=10000):
        super().__init__()
        self.dim = int(dim)
        self.rotary_emb_base = rotary_emb_base

    def build(self, input_shape):
        super().build(input_shape)
        self.inverse_freq = self.add_weight(
            "inverse_freq", shape=(self.dim // 2,), dtype=tf.float32
        )
        range = tf.range(start=0, limit=self.dim, delta=2, dtype="float32")

        self.inverse_freq.assign(
            1.0 / (self.rotary_emb_base ** (range / self.dim))
        )

    @staticmethod
    def _apply_rotary_pos_emb(tensor, cos_emb, sin_emb):
        cos_emb = cos_emb[:, : tf.shape(tensor)[1], :, :]
        sin_emb = sin_emb[:, : tf.shape(tensor)[1], :, :]
        x1, x2 = tf.split(tensor, 2, axis=-1)
        half_rot_tensor = tf.concat((-x2, x1), axis=-1)
        # Incompatible shapes: [32,256,8,2] vs. [1,256,1,16] [Op:Mul]
        ret = (tensor * cos_emb) + (half_rot_tensor * sin_emb)
        return ret

    def _compute_cos_sin_embedding(self, x, seq_dim=1):
        seq_len = tf.shape(x)[seq_dim]
        tensor = tf.range(seq_len, dtype=self.inverse_freq.dtype)
        freqs = tf.einsum("i, j -> ij", tensor, self.inverse_freq)
        embedding = tf.concat((freqs, freqs), axis=-1)[None, :, None, :]
        return tf.cos(embedding), tf.sin(embedding)

    def call(self, query, key):
        cos_emb, sin_emb = self._compute_cos_sin_embedding(key, seq_dim=1)
        q_emb = self._apply_rotary_pos_emb(query, cos_emb, sin_emb)
        k_emb = self._apply_rotary_pos_emb(key, cos_emb, sin_emb)
        return q_emb, k_emb

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "rotary_emb_base": self.rotary_emb_base,
                "inverse_freq": self.inverse_freq,
            }
        )
