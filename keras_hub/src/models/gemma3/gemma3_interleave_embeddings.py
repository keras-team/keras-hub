import keras
from jax import numpy as jnp
from keras import ops


class Gemma3InterleaveEmbeddings(keras.layers.Layer):
    """Places image embeddings in the correct position in a text embedding
    sequence.

    Args:
        image_max_length: int. The maximum number of images per sample (padded).
            Defaults to `None`.
    """

    def __init__(self, image_max_length, num_vision_tokens_per_image, **kwargs):
        super().__init__(**kwargs)

        self.image_max_length = image_max_length
        self.num_vision_tokens_per_image = num_vision_tokens_per_image

    def call(self, image_embeddings, text_embeddings, text_mask):
        """
        Integrates image embeddings into a text embedding sequence.

        Args:
            image_embeddings: Tensor of shape (batch_size * image_max_length,
                num_vision_tokens_per_image, embedding_dim).
            text_embeddings: Tensor of shape (batch_size, seq_length,
                embedding_dim).
            text_mask: Boolean tensor of shape (batch_size, seq_length).

        Returns:
            Tensor of shape (batch_size, seq_length, embedding_dim) representing
            the reconstructed embeddings.
        """

        batch_size, seq_length, embedding_dim = ops.shape(text_embeddings)

        # Flatten text embeddings, text mask and image embeddings.
        flat_text_embeddings = ops.reshape(
            text_embeddings, (batch_size * seq_length, embedding_dim)
        )
        flat_text_mask = ops.reshape(text_mask, (batch_size * seq_length,))

        # The image batch size might be different when we pass only text.
        image_batch_size = ops.shape(image_embeddings)[0]
        flat_image_embeddings = ops.reshape(
            image_embeddings,
            (
                image_batch_size * self.num_vision_tokens_per_image,
                embedding_dim,
            ),
        )

        # Reconstruct embeddings.
        if keras.backend.backend() == "jax":
            indices = jnp.where(
                jnp.logical_not(flat_text_mask),
                size=image_batch_size * self.num_vision_tokens_per_image,
            )
        else:
            indices = ops.where(
                ops.logical_not(flat_text_mask),
            )
        indices = ops.cast(indices, "int32")
        indices = ops.transpose(indices)
        reconstructed_embedding = ops.scatter_update(
            flat_text_embeddings, indices, flat_image_embeddings
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
        config["image_max_length"] = self.image_max_length
        return config
