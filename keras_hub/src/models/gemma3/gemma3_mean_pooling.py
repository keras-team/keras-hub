# Copyright 2024 The KerasCV Authors
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

import keras
from keras import ops


class MeanPooling(keras.layers.Layer):
    """Mean pooling layer that computes the average of token embeddings.

    This layer correctly handles variable-length sequences by ignoring
    padded tokens in the mean calculation, using a `padding_mask`.

    Call arguments:
        sequence_output: A tensor of shape `(batch_size, seq_len, hidden_dim)`.
        padding_mask: A tensor of shape `(batch_size, seq_len)` with `1`
            for valid tokens and `0` for padded tokens.

    Returns:
        A tensor of shape `(batch_size, hidden_dim)`.

    Example:
    ```python
    sequence_output = np.random.rand(2, 4, 8).astype("float32")
    padding_mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]])
    mean_pool_layer = MeanPooling()
    pooled = mean_pool_layer(
        sequence_output=sequence_output,
        padding_mask=padding_mask
    )
    # pooled.shape -> (2, 8)
    ```
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, sequence_output, padding_mask):
        """
        Computes the masked mean pooling.

        Args:
            sequence_output: The tensor of token embeddings.
            padding_mask: The mask indicating which tokens to consider.
        """
        # Expand the mask to match the dimensions of the sequence output
        mask = ops.expand_dims(
            ops.cast(padding_mask, sequence_output.dtype), axis=-1
        )

        # Apply the mask to zero out the padded tokens
        masked_output = sequence_output * mask

        # Sum the embeddings of the valid tokens
        sum_embeddings = ops.sum(masked_output, axis=1)

        # Count the number of valid tokens for each sequence
        num_tokens = ops.sum(
            ops.cast(padding_mask, sequence_output.dtype), axis=1
        )
        num_tokens = ops.expand_dims(num_tokens, axis=1)

        # Add a small epsilon to avoid division by zero for empty sequences
        num_tokens = ops.maximum(num_tokens, 1e-9)

        # Compute the mean by dividing the sum by the count of valid tokens
        mean_embeddings = sum_embeddings / num_tokens
        return mean_embeddings

    def compute_output_shape(self, sequence_output_shape, padding_mask_shape):
        """Computes the output shape of the layer."""
        return (sequence_output_shape[0], sequence_output_shape[2])

    def get_config(self):
        """Returns the config of the layer."""
        return super().get_config()