import keras
from keras import ops


class Qwen2VLInterleaveEmbeddings(keras.layers.Layer):
    """Scatter visual token embeddings into the text embedding sequence.

    Given a (batch, seq_len, hidden) text embedding tensor and a flat list of
    visual token embeddings, this layer replaces positions indicated by
    `vision_indices` with the corresponding visual embeddings.

    This is the KerasHub equivalent of the HF
    `inputs_embeds.masked_scatter(image_mask, image_embeds)` pattern.

    Args:
        hidden_dim: int. The embedding dimension (must match both text and
            visual embeddings after the vision projection).
    """

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def call(self, image_embeddings, text_embeddings, vision_indices):
        """Interleave vision tokens into the text embedding sequence.

        Accepts both batched and unbatched image embeddings:
        - Unbatched (from imperative call):
            image_embeddings ``(total_vision_tokens, hidden_dim)``,
            vision_indices ``(total_vision_tokens,)``.
        - Batched (from backbone functional graph):
            image_embeddings ``(batch, total_vision_tokens, hidden_dim)``,
            vision_indices ``(batch, total_vision_tokens)``.

        Args:
            image_embeddings: Tensor with visual token embeddings.
            text_embeddings: Tensor ``(batch, seq_len, hidden_dim)``.
            vision_indices: int32 Tensor with flat indices into the
                concatenated ``(batch x seq_len)`` sequence.

        Returns:
            Tensor ``(batch, seq_len, hidden_dim)`` with visual tokens
            inserted at the specified positions.
        """
        batch_size = ops.shape(text_embeddings)[0]
        seq_len = ops.shape(text_embeddings)[1]

        # Handle batched image_embeddings from the functional graph.
        # Squeeze batch dim: (batch, N, hidden) -> (N, hidden)
        if len(ops.shape(image_embeddings)) == 3:
            image_embeddings = ops.reshape(
                image_embeddings, (-1, self.hidden_dim)
            )
        if len(ops.shape(vision_indices)) == 2:
            vision_indices = ops.reshape(vision_indices, (-1,))

        # Flatten the text embedding to (batch * seq_len, hidden_dim).
        flat_text = ops.reshape(text_embeddings, (-1, self.hidden_dim))

        # Cast vision indices to int32 and reshape to (N, 1) for
        # scatter_update.
        vision_indices = ops.cast(vision_indices, "int32")
        vision_indices = ops.expand_dims(vision_indices, axis=-1)

        flat_out = ops.scatter_update(
            flat_text, vision_indices, image_embeddings
        )

        return ops.reshape(flat_out, (batch_size, seq_len, self.hidden_dim))

    def compute_output_spec(
        self, image_embeddings, text_embeddings, vision_indices
    ):
        """The output shape is identical to ``text_embeddings``."""
        return keras.KerasTensor(
            shape=text_embeddings.shape,
            dtype=text_embeddings.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config
