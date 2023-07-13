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
    def __init__(self, max_wavelength=10000, **kwargs):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength

    def call(self, query, key):
        rotary_dim = ops.shape(query)[-1]

        cos_emb, sin_emb = self._compute_cos_sin_embedding(
            key, rotary_dim, seq_dim=1
        )
        query = self._apply_rotary_pos_emb(query, cos_emb, sin_emb)
        key = self._apply_rotary_pos_emb(key, cos_emb, sin_emb)

        return query, key

    def _apply_rotary_pos_emb(self, tensor, cos_emb, sin_emb):
        cos_emb = cos_emb[:, : ops.shape(tensor)[1], :, :]
        sin_emb = sin_emb[:, : ops.shape(tensor)[1], :, :]

        x1, x2 = ops.split(tensor, 2, axis=-1)
        half_rot_tensor = ops.concatenate((-x2, x1), axis=-1)

        return (tensor * cos_emb) + (half_rot_tensor * sin_emb)

    def _compute_cos_sin_embedding(self, x, rotary_dim, seq_dim=1):
        seq_len = ops.shape(x)[seq_dim]

        range = ops.arange(0, rotary_dim, 2, self.compute_dtype)
        inverse_freq = 1.0 / (
            self.max_wavelength
            ** (range / ops.cast(rotary_dim, self.compute_dtype))
        )

        tensor = ops.arange(seq_len, dtype=inverse_freq.dtype)
        freqs = ops.einsum("i, j -> ij", tensor, inverse_freq)
        embedding = ops.concatenate((freqs, freqs), axis=-1)[None, :, None, :]

        return ops.cos(embedding), ops.sin(embedding)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
            }
        )
        return config
