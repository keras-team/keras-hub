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
    """Rotary positional encoding layer.

    Tbjs layer encodes absolute positional information with rotation
    matrix and naturally incorporates explicit relative position
    dependency in self-attention formulation. This layer calculates
    the rotary encoding with a mix of sine and cosine functions with
    geometrically increasing wavelengths. Defined and formulized
    in [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4).
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
    batch_size = 16
    num_heads = 8
    sequence_length = 256
    query_length = 256
    query = tf.ones((batch_size, num_heads, sequence_length, query_length))
    rot_emb = RotaryEmbedding()
    query_rot = rot_emb(query)
    ```

    References:
     - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4)
    """
    def __init__(self, max_wavelength=10000, **kwargs):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength

    def call(self, inputs):
        rotary_dim = ops.shape(inputs)[-1]

        cos_emb, sin_emb = self._compute_cos_sin_embedding(
            inputs, rotary_dim, seq_dim=1
        )
        outputs = self._apply_rotary_pos_emb(inputs, cos_emb, sin_emb)

        return outputs

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

    def compute_output_shape(self, input_shape):
        return input_shape
