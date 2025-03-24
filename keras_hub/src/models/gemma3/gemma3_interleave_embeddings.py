import keras
from keras import ops


class Gemma3InterleaveEmbeddings(keras.layers.Layer):
    """Places image embeddings in the correct position in an embedding sequence.

    Args:
        num_vision_tokens_per_image: int. Number of soft tokens per image.
    """

    def __init__(self, num_vision_tokens_per_image, **kwargs):
        super().__init__(**kwargs)

        self.num_vision_tokens_per_image = num_vision_tokens_per_image

    def call(self, image_embeddings, text_embeddings, vision_indices):
        """
        Integrates image embeddings into a text embedding sequence.

        Args:
            image_embeddings: Tensor of shape
                `(batch_size * num_images_per_prompt,
                num_vision_tokens_per_image, embedding_dim)`.
            text_embeddings: Tensor of shape
            `(batch_size, seq_length, embedding_dim)`.
            text_mask: Boolean tensor of shape `(batch_size, seq_length)`.

        Returns:
            Tensor of shape `(batch_size, seq_length, embedding_dim)`
            representing the reconstructed embeddings.
        """

        batch_size, seq_length, embedding_dim = ops.shape(text_embeddings)

        # Flatten text embeddings, text mask and image embeddings.
        flat_text_embeddings = ops.reshape(
            text_embeddings, (batch_size * seq_length, embedding_dim)
        )

        # The image batch size might be different when we pass only text, i.e,
        # it will be 0 for text-only.
        image_batch_size = ops.shape(image_embeddings)[0]
        flat_image_embeddings = ops.reshape(
            image_embeddings,
            (
                image_batch_size * self.num_vision_tokens_per_image,
                embedding_dim,
            ),
        )

        # Reconstruct embeddings.
        vision_indices_shape = ops.shape(vision_indices)
        flat_vision_indices = ops.reshape(
            vision_indices,
            (vision_indices_shape[0] * vision_indices_shape[1], 1),
        )
        indices = ops.cast(flat_vision_indices, "int32")
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
        config.update(
            {
                "num_vision_tokens_per_image": self.num_vision_tokens_per_image,
            }
        )
        return config
