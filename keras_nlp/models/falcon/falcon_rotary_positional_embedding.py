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

"""Position embedding implementation based on `keras.layers.Layer`."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export


@keras_nlp_export("keras_nlp.layers.RotaryPositionalEmbedding")
class FalconRotaryPositionalEmbedding(keras.layers.Layer):
    """Rotary Positional Embedding layer has been created for falcon models.


    This layer computes rotary positional embeddings that can be used in Transformer models
    to incorporate positional information into the input sequences.

    Args:
        head_dim (int): The dimensionality of the positional embeddings.
        base (int): The base value used in the exponential calculation. Defaults to 10000.

    Input Shape:
        - `q` and `k`: Tensor of shape `(batch_size, seq_len, head_dim)` representing queries and keys.

    Output Shape:
        - `output_q` and `output_k`: Tensor of shape `(batch_size, seq_len, head_dim)` representing the transformed queries and keys.

    Examples:
        ```python
        embedding = FalconRotaryPositionalEmbedding(head_dim)
        output_q, output_k = embedding(q, k)
        ```

    Reference:
        - [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
        - [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
    """

    def __init__(self,
                 head_dim: int,
                 base=10000):
        super(FalconRotaryPositionalEmbedding, self).__init__()
        inv_freq = 1.0 / (base ** (tf.range(0, head_dim, 2, dtype=tf.float32) / head_dim))
        self.inv_freq = tf.Variable(inv_freq, trainable=False)
        self.head_dim = head_dim
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def cos_sin(self, seq_len):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = tf.range(seq_len, dtype=tf.float32)
            freqs = tf.einsum("i,j->ij", t, self.inv_freq)
            emb = tf.concat((freqs, freqs), axis=-1)

            self.cos_cached = tf.cos(emb)[None, :, :]
            self.sin_cached = tf.sin(emb)[None, :, :]

        return self.cos_cached, self.sin_cached

    def call(self, q, k):
        batch, seq_len, head_dim = q.shape
        cos, sin = self.cos_sin(seq_len)
        rotated_q = tf.concat((-q[..., head_dim // 2 :], q[..., : head_dim // 2]), axis=q.ndim - 1)
        return (q * cos) + (rotated_q * sin), (k * cos) + (rotated_q * sin)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_dim": self.head_dim,
            }
        )
        return config
