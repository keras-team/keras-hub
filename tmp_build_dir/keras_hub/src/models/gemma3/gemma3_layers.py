import keras
from keras import ops


class RMSNormalization(keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(input_shape[-1],),
            initializer="zeros",
        )
        self.built = True

    def call(self, x):
        # Always compute normalization in float32.
        x = ops.cast(x, "float32")
        scale = ops.cast(self.scale, "float32")
        var = ops.mean(ops.square(x), axis=-1, keepdims=True)
        normed_inputs = x * ops.reciprocal(ops.sqrt(var + self.epsilon))
        normed_inputs = normed_inputs * (1 + scale)
        return ops.cast(normed_inputs, self.compute_dtype)


class Gemma3MeanPooling(keras.layers.Layer):
    """Mean pooling layer that computes the average of token embeddings.

    This layer correctly handles variable-length sequences by ignoring
    padded tokens in the mean calculation, using a `padding_mask`.

    Example:
    ```python
    import numpy as np

    sequence_output = np.random.rand(2, 4, 8).astype("float32")
    padding_mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype="int32")
    mean_pool_layer = Gemma3MeanPooling()
    pooled = mean_pool_layer([sequence_output, padding_mask])
    # pooled.shape -> (2, 8)
    ```
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, padding_mask=None):
        """Performs masked mean pooling on the token embeddings.

        Args:
            inputs: The sequence of embeddings to pool, with a shape of
                `(batch_size, seq_len, hidden_dim)`.
            padding_mask: The mask indicating valid tokens, with a shape of
                `(batch_size, seq_len)`.

        Returns:
            A tensor representing the pooled embeddings, with a shape of
            `(batch_size, hidden_dim)`.
        """
        if padding_mask is None:
            inputs, padding_mask = inputs

        sequence_output = inputs
        mask = ops.expand_dims(
            ops.cast(padding_mask, sequence_output.dtype), axis=-1
        )

        masked_output = sequence_output * mask

        sum_embeddings = ops.sum(masked_output, axis=1)

        num_tokens = ops.sum(
            ops.cast(padding_mask, sequence_output.dtype), axis=1
        )
        num_tokens = ops.expand_dims(num_tokens, axis=1)
        # Avoid division by zero
        num_tokens = ops.maximum(num_tokens, 1e-9)

        mean_embeddings = sum_embeddings / num_tokens
        return ops.cast(mean_embeddings, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Args:
            input_shape: A tuple or list of tuples representing input shapes.

        Returns:
            A tuple representing the output shape.
        """
        if isinstance(input_shape, list):
            sequence_output_shape = input_shape[0]
        else:
            sequence_output_shape = input_shape
        return sequence_output_shape[:-2] + (sequence_output_shape[-1],)

    def get_config(self):
        """Returns the config of the layer."""
        return super().get_config()


class Gemma3InterleaveEmbeddings(keras.layers.Layer):
    """Places image embeddings in the correct position in an embedding sequence.

    For Gemma3, images can be in any position in the input sequence. In order
    to do accomplish this, we have image placeholder tokens in the input
    sequence. We fill up these positions with the image embeddings as returned
    by the vision encoder.

    Args:
        num_vision_tokens_per_image: int. Number of soft tokens per image.
    """

    def __init__(self, num_vision_tokens_per_image, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        self.num_vision_tokens_per_image = num_vision_tokens_per_image

    def call(self, image_embeddings, text_embeddings, vision_indices):
        """
        Integrates image embeddings into a text embedding sequence.

        Args:
            image_embeddings: tensor. Image embeddings as returned by the
                vision encoder (`Gemma3VisionEncoder`, usually). Shape:
                `(batch_size * num_images_per_prompt, `
                `num_vision_tokens_per_image, embedding_dim)`.
            text_embeddings: tensor. Embeddings returned by the text embedding
                layer. Shape: `(batch_size, seq_length, embedding_dim)`.
            vision_indices:  tensor. Indexes into `text_embeddings`, used to
                identify which places are supposed to be replaced by
                `image_embeddings`. Shape:
                `(batch_size,`
                `num_images_per_prompt * num_vision_tokens_per_image)`.

        Returns:
            Tensor of shape `(batch_size, seq_length, embedding_dim)`
            representing the reconstructed embeddings.
        """

        batch_size, seq_length, embedding_dim = ops.shape(text_embeddings)
        # `num_images` will be 0 for text only inputs, and
        # `batch_size * max_images_per_prompt` if images are passed.
        num_images = ops.shape(image_embeddings)[0]

        # Flatten text embeddings, image embeddings and indices.
        flat_text_embeddings = ops.reshape(
            text_embeddings, (batch_size * seq_length, embedding_dim)
        )
        # `flat_image_embeddings` is the `updates` tensor and should be of shape
        # `(num_updates, embedding_dim)`.
        flat_image_embeddings = ops.reshape(
            image_embeddings,
            (
                num_images * self.num_vision_tokens_per_image,
                embedding_dim,
            ),
        )

        # For vision indices, we need to add values such that the indices
        # index into a flattened `text_embeddings`.
        to_add = ops.multiply(
            keras.ops.arange(batch_size, dtype="int32"), seq_length
        )
        to_add = ops.cast(ops.expand_dims(to_add, axis=-1), "int32")
        vision_indices = ops.add(vision_indices, to_add)

        # indices should be of shape `(num_updates, 1)`. `num_updates` is
        # how many vision tokens there are to update.
        vision_indices_shape = ops.shape(vision_indices)
        flat_vision_indices = ops.reshape(
            vision_indices,
            (vision_indices_shape[0] * vision_indices_shape[1], 1),
        )
        indices = ops.cast(flat_vision_indices, "int32")

        # Before reconstructing, store the 0th index so that we can restore it
        # later.
        zeroth_index_text_embeddings = ops.take(
            flat_text_embeddings,
            indices=ops.squeeze(to_add, axis=-1),
            axis=0,
        )

        # Reconstruct embeddings
        reconstructed_embedding = ops.scatter_update(
            inputs=flat_text_embeddings,
            indices=indices,
            updates=flat_image_embeddings,
        )

        # Remember that we pad `vision_indices` with the 0th index. We need to
        # restore the original value in the reconstructed embedding tensor.
        reconstructed_embedding = ops.scatter_update(
            inputs=reconstructed_embedding,
            indices=to_add,
            updates=zeroth_index_text_embeddings,
        )

        # Reshape to original dimensions
        reconstructed_embedding = ops.reshape(
            reconstructed_embedding, (batch_size, seq_length, embedding_dim)
        )
        return reconstructed_embedding

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_vision_tokens_per_image": self.num_vision_tokens_per_image,
            }
        )
        return config
