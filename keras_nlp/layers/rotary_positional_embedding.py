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
class RotaryPositionalEmbedding(keras.layers.Layer):
    """Rotary Positional Embedding layer for Transformers.

    This layer computes rotary positional embeddings that can be used in Transformer models
    to incorporate positional information into the input sequences.

    Args:
        dim: The dimensionality of the positional embeddings.

    Input Shape:
        - A scalar tensor representing the maximum sequence length.

    Output Shape:
        - A tensor of shape `(max_seq_len, 2 * dim)` representing the rotary positional embeddings.

    Examples:
        ```python
        rotary_emb = RotaryPositionalEmbedding(dim)
        pos_emb = rotary_emb(n, device=device)
        q, k = map(lambda t: apply_rotary_pos_emb(pos_emb, t), (q, k))
        ```

    Reference:
        - [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
    """

    def __init__(self, dim):
        super(RotaryPositionalEmbedding, self).__init__()
        inv_freq = 1.0 / (10000 ** (tf.range(0, dim, 2, dtype=tf.float32) / dim))
        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=(dim,),
            initializer=tf.keras.initializers.Constant(inv_freq),
            trainable=False,
        )

    def call(self, max_seq_len, device=None):
        """Compute rotary positional embeddings.

        Args:
            max_seq_len: The maximum length of the input sequence.
            device: Optional device to place the tensors.

        Returns:
            A tensor of shape `(max_seq_len, 2 * dim)` representing the rotary positional embeddings.
        """
        seq = tf.range(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = tf.einsum("i,j->ij", seq, self.inv_freq)
        return tf.concat((freqs, freqs), axis=-1)


def rotate_half(x):
    """Rotate the input tensor by half.

    Args:
        x: The input tensor.

    Returns:
        The rotated tensor.
    """
    x = tf.reshape(x, (-1, 2))
    x1, x2 = tf.unstack(x, axis=-1)
    return tf.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(pos, t):
    """Apply rotary positional embedding to the input tensor.

    Args:
        pos: The rotary positional embeddings.
        t: The input tensor.

    Returns:
        The tensor with applied rotary positional embedding.
    """
    return (t * tf.cos(pos)) + (rotate_half(t) * tf.sin(pos))
