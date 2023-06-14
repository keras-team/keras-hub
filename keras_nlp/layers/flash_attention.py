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
from keras_nlp.api_export import keras_nlp_export


@keras_nlp_export("keras_nlp.layers.FlashAttention")
class FlashAttention(keras.layers.Layer):
    """
    FlashAttention layer implementation.

    Args:
        blocksizes (tuple): The block sizes (Br, Bc) as mentioned in the paper.
        dim (int): The dimension of the q, k, v vectors.

    Input Shapes:
        - q: 3D tensor with shape `(batch_size, seq_len, d)`.
        - k: 3D tensor with shape `(batch_size, seq_len, d)`.
        - v: 3D tensor with shape `(batch_size, seq_len, d)`.

    Output Shape:
        - output: 3D tensor with shape `(batch_size, seq_len, d)`.

    Note:
        This implementation assumes that q, k, and v vectors are already calculated.

    Example:
    ```python
        flash_attn = FlashAttention(blocksizes=(Br, Bc), dim=d)
        output = flash_attn([q, k, v])
    ```
    References:
        - [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness  (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
    """

    def __init__(self, blocksizes, dim, **kwargs):
        super(FlashAttention, self).__init__(**kwargs)
        self.blocksizes = blocksizes
        self.dim = dim

    def call(self, inputs):
        """
        Perform FlashAttention calculation on the input tensors.

        Args:
            inputs (list): List of input tensors [q, k, v].

        Returns:
            output (tf.Tensor): FlashAttention output tensor.
        """
        # Get the input tensors.
        q, k, v = inputs

        # Get the lengths of the input tensors.
        q_len, k_len = q.shape[1], k.shape[1]

        # Get the block sizes.
        Br, Bc = self.blocksizes

        # Calculate the number of blocks in each dimension.
        num_blocks_row = q_len // Br + q_len % Br
        num_blocks_col = k_len // Bc + k_len % Bc

        # Split the input tensors into blocks.
        q_blocks = tf.split(q, num_blocks_row, axis=1)  # (Br, d) x Tr
        k_blocks = tf.split(k, num_blocks_col, axis=1)  # (Bc, d) x Tc
        v_blocks = tf.split(v, num_blocks_col, axis=1)  # (Bc, d) x Tc

        # Initialize the output tensor.
        output_blocks = [tf.zeros((q_len, self.dim))] * num_blocks_row  # (Br, seq_len, d) x Tr

        # Calculate the attention weights for each block.
        for j in range(num_blocks_col):
            # Get the current block of keys and values.
            K_j, V_j = k_blocks[j], v_blocks[j]

            # Calculate the scores for each query-key pair.
            scores = tf.matmul(q_blocks, tf.transpose(K_j, perm=[1, 0]))  # (Br, Bc)

            # Calculate the attention weights.
            attention_weights = tf.exp(scores)  # (Br, Bc)

            # Update the output tensor.
            for i in range(num_blocks_row):
                # Get the current block of queries.
                Q_i = q_blocks[i]

                # Calculate the output for the current block.
                output_blocks[i] += tf.matmul(attention_weights, V_j)

        # Concatenate the output blocks.
        output = tf.concat(output_blocks, axis=0)

        return output

