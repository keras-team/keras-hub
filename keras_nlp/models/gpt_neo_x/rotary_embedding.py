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
from keras_nlp.backend import keras
from keras_nlp.backend import ops


class RotaryEmbedding(keras.layers.Layer):
    def __init__(self, percentage, max_wavelength=10000):
        super().__init__()
        self.percentage = percentage
        self.max_wavelength = max_wavelength

    def _apply_rotary_pos_emb(self, tensor, cos_emb, sin_emb):
        cos_emb = cos_emb[:, : ops.shape(tensor)[1], :, :]
        sin_emb = sin_emb[:, : ops.shape(tensor)[1], :, :]
        x1, x2 = ops.split(tensor, 2, axis=-1)
        half_rot_tensor = ops.concatenate((-x2, x1), axis=-1)
        ret = (tensor * cos_emb) + (half_rot_tensor * sin_emb)
        return ret

    def _compute_cos_sin_embedding(self, x, seq_dim=1):
        seq_len = ops.shape(x)[seq_dim]
        range = ops.arange(0, self.rotary_ndims, 2, "float32")
        inverse_freq = 1.0 / (
            self.max_wavelength ** (range / self.rotary_ndims)
        )
        tensor = ops.arange(seq_len, dtype=inverse_freq.dtype)
        freqs = ops.einsum("i, j -> ij", tensor, inverse_freq)
        embedding = ops.concatenate((freqs, freqs), axis=-1)[None, :, None, :]
        return ops.cos(embedding), ops.sin(embedding)

    def build(self, attn_head_size):
        self.attn_head_size = attn_head_size
        self.rotary_ndims = int(self.attn_head_size * self.percentage)
        self.built = True

    def __call__(self, query, key):
        if not self.built:
            attn_head_size = query.shape[-1]
            self.build(attn_head_size)
        return super().__call__(query, key)

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

        query = ops.concatenate((query_emb, query_pass), axis=-1)
        key = ops.concatenate((key_emb, key_pass), axis=-1)

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
