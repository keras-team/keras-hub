import keras
from keras import ops


class MeanPooling(keras.layers.Layer):
    """Mean pooling layer that computes the average of token embeddings.

    This layer correctly handles variable-length sequences by ignoring
    padded tokens in the mean calculation, using a `padding_mask`.

    Example:
    ```python
    import numpy as np

    sequence_output = np.random.rand(2, 4, 8).astype("float32")
    padding_mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype="int32")
    mean_pool_layer = MeanPooling()
    pooled = mean_pool_layer([sequence_output, padding_mask])
    # pooled.shape -> (2, 8)
    ```
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        """Performs masked mean pooling on the token embeddings.

        Args:
            inputs: A list or tuple of two tensors:
                - sequence_output: The sequence of embeddings to pool, with a
                  shape of `(batch_size, seq_len, hidden_dim)`.
                - padding_mask: The mask indicating valid tokens, with a shape
                  of `(batch_size, seq_len)`.

        Returns:
            A tensor representing the pooled embeddings, with a shape of
            `(batch_size, hidden_dim)`.
        """
        sequence_output, padding_mask = inputs
        mask = ops.expand_dims(
            ops.cast(padding_mask, sequence_output.dtype), axis=-1
        )

        masked_output = sequence_output * mask

        sum_embeddings = ops.sum(masked_output, axis=1)

        num_tokens = ops.sum(
            ops.cast(padding_mask, sequence_output.dtype), axis=1
        )
        num_tokens = ops.expand_dims(num_tokens, axis=1)
        num_tokens = ops.maximum(num_tokens, 1e-9)

        mean_embeddings = sum_embeddings / num_tokens
        return mean_embeddings

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Args:
            input_shape: A tuple of input shapes.

        Returns:
            A tuple representing the output shape.
        """
        sequence_output_shape, _ = input_shape
        return (sequence_output_shape[0], sequence_output_shape[2])

    def get_config(self):
        """Returns the config of the layer."""
        return super().get_config()
