import keras
from keras import ops


class Gemma3nInterleaveEmbeddings(keras.layers.Layer):
    """Places image embeddings in the correct position in an embedding sequence.

    For Gemma3, images can be in any position in the input sequence. In order
    to do accomplish this, we have image placeholder tokens in the input
    sequence. We fill up these positions with the image embeddings as returned
    by the vision encoder.

    Args:
        num_vision_tokens_per_image: int. Number of soft tokens per image.
    """

    def __init__(
        self,
        num_vision_tokens_per_image,
        num_audio_tokens,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.num_vision_tokens_per_image = num_vision_tokens_per_image
        self.num_audio_tokens = num_audio_tokens

    def call(
        self,
        text_embeddings,
        image_embeddings=None,
        vision_indices=None,
        audio_embeddings=None,
        audio_indices=None,
    ):
        """Integrates modality embeddings into a text embedding sequence."""
        batch_size, seq_length, embedding_dim = ops.shape(text_embeddings)
        reconstructed_embedding = ops.reshape(
            text_embeddings, (batch_size * seq_length, embedding_dim)
        )

        # Helper function to scatter embeddings for a given modality
        def scatter_modality(
            flat_embeddings,
            modality_embeddings,
            modality_indices,
            num_tokens_per_item,
        ):
            if modality_embeddings is None or modality_indices is None:
                return flat_embeddings

            num_items = ops.shape(modality_embeddings)[0]
            if num_items == 0:
                return flat_embeddings

            flat_modality_embeddings = ops.reshape(
                modality_embeddings,
                (num_items * num_tokens_per_item, embedding_dim),
            )

            to_add = ops.multiply(
                ops.arange(batch_size, dtype="int32"), seq_length
            )
            to_add = ops.cast(ops.expand_dims(to_add, axis=-1), "int32")
            modality_indices = ops.add(modality_indices, to_add)

            modality_indices_shape = ops.shape(modality_indices)
            flat_modality_indices = ops.reshape(
                modality_indices,
                (modality_indices_shape[0] * modality_indices_shape[1], 1),
            )
            indices = ops.cast(flat_modality_indices, "int32")

            return ops.scatter_update(
                inputs=flat_embeddings,
                indices=indices,
                updates=flat_modality_embeddings,
            )

        # Scatter images first
        reconstructed_embedding = scatter_modality(
            reconstructed_embedding,
            image_embeddings,
            vision_indices,
            self.num_vision_tokens_per_image,
        )

        # Then scatter audio on top of the result
        reconstructed_embedding = scatter_modality(
            reconstructed_embedding,
            audio_embeddings,
            audio_indices,
            self.num_audio_tokens,
        )

        # Reshape to original 3D tensor
        return ops.reshape(
            reconstructed_embedding, (batch_size, seq_length, embedding_dim)
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_vision_tokens_per_image": self.num_vision_tokens_per_image,
                "num_audio_tokens": self.num_audio_tokens,
            }
        )
        return config
