"""Qwen3.5 Layers.

Contains shared sub-layers for the Qwen3.5 multimodal backbone:
  - Qwen3_5InterleaveEmbeddings: scatter vision tokens into text sequence.
"""

import keras
from keras import ops


class Qwen3_5InterleaveEmbeddings(keras.layers.Layer):
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

        Args:
            image_embeddings: Tensor (total_vision_tokens, hidden_dim).
                All visual tokens for all images in the batch, concatenated
                along axis 0.
            text_embeddings: Tensor (batch, seq_len, hidden_dim).
            vision_indices: int32 Tensor (total_vision_tokens,).
                Flat indices into the *concatenated* (batch × seq_len)
                sequence indicating where each visual token should be placed.
                Produced by the preprocessor to be consistent across the batch
                by treating the sequence as batch×seq_len flat.

        Returns:
            Tensor (batch, seq_len, hidden_dim) with visual tokens inserted
            at the specified positions.
        """
        batch_size = ops.shape(text_embeddings)[0]
        seq_len = ops.shape(text_embeddings)[1]

        # Flatten the text embedding to (batch * seq_len, hidden_dim).
        flat_text = ops.reshape(text_embeddings, (-1, self.hidden_dim))

        # Cast vision indices to int32 and reshape to (N, 1) for scatter_update.
        vision_indices = ops.cast(vision_indices, "int32")
        vision_indices = ops.expand_dims(vision_indices, axis=-1)

        # Scatter vision embeddings into the flat text tensor.
        # ops.scatter_update replaces rows at the given indices.
        flat_out = ops.scatter_update(
            flat_text, vision_indices, image_embeddings
        )

        # Reshape back to (batch, seq_len, hidden_dim).
        return ops.reshape(flat_out, (batch_size, seq_len, self.hidden_dim))

    def compute_output_spec(
        self, image_embeddings, text_embeddings, vision_indices
    ):
        """Return the output shape spec for functional model tracing.

        The output shape is identical to text_embeddings — we just replace
        some rows in-place.
        """
        return keras.KerasTensor(
            shape=text_embeddings.shape,
            dtype=text_embeddings.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config
