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
    """Rotary positional encoding layer.

        Tbjs layer encodes absolute positional information with rotation matrix and naturally
        incorporates explicit relative position dependency in self-attention formulation.
        It layer calculates the rotary encoding with a mix of sine and cosine
        functions with geometrically increasing wavelengths. Defined and formulized
        in [RoFormer: Enhanced Transformer with Rotary Position Embedding
    ](https://arxiv.org/abs/2104.09864v4).

        Takes as input the query and key tensors. The input must have shape
        [batch_size, num_heads, sequence_length, query_length]. This layer will return
        new query and key tensors after applying rotational encoding.

        Args:
            percentage: float. The percentage of attn_head_size over which rotation
                should be applied. Defaults to 0
            max_wavelength: int. The maximum angular wavelength of the sine/cosine
                curves, as described in Attention is All You Need. Defaults to
                `10000`.

        Examples:
        ```python
        rotary_percentage = 0.25
        batch_size = 16
        num_heads = 8
        sequence_length = 256
        query_length = key_length = 256
        query = tf.ones((batch_size, num_heads, sequence_length, query_length))
        key = tf.ones((batch_size, num_heads, sequence_length, query_length))

        rot_emb = RotaryEmbedding(rotary_percentage)
        query_rot, key_rot = rot_emb(query, key)
        ```

        References:
         - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4)
    """

    def __init__(self, percentage=0, max_wavelength=10000):
        super().__init__()
        self.percentage = percentage
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
            tf.cast(attn_head_size, self.compute_dtype) * self.percentage,
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
                "percentage": self.percentage,
                "max_wavelength": self.max_wavelength,
            }
        )
        return config
