import keras
from keras import ops


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
