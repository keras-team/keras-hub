import keras
from keras import ops


class Qwen3OmniVisualPosMask(keras.layers.Layer):
    """Build a ``(batch, seq_len)`` bool mask from flat vision indices.

    A thin layer wrapper so the scatter runs inside ``call`` (dynamic
    shapes) instead of during functional-graph build.
    """

    def call(self, vision_indices, reference_tensor):
        flat = ops.reshape(
            ops.cast(ops.zeros_like(reference_tensor[..., 0]), "int32"), (-1,)
        )
        indices = ops.reshape(ops.cast(vision_indices, "int32"), (-1,))
        flat = ops.scatter_update(
            flat, ops.expand_dims(indices, axis=-1), ops.ones_like(indices)
        )
        return ops.cast(
            ops.reshape(
                flat,
                (
                    ops.shape(reference_tensor)[0],
                    ops.shape(reference_tensor)[1],
                ),
            ),
            "bool",
        )

    def compute_output_spec(self, vision_indices, reference_tensor):
        return keras.KerasTensor(shape=reference_tensor.shape[:2], dtype="bool")


class Qwen3OmniInterleaveEmbeddings(keras.layers.Layer):
    """Scatter vision / audio embeddings into a text sequence.

    Replaces positions named by ``vision_indices`` / ``audio_indices``
    in ``text_embeddings`` with the matching encoder outputs. Either
    pair may be empty (vision-only / audio-only / fully-multimodal).

    Args:
        hidden_dim: int. Common embedding dimension.
    """

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def _scatter(self, flat_text, source_embeddings, source_indices):
        """Scatter ``source_embeddings`` into ``flat_text``; empty no-op."""
        if len(ops.shape(source_embeddings)) == 3:
            source_embeddings = ops.reshape(
                source_embeddings, (-1, self.hidden_dim)
            )
        if len(ops.shape(source_indices)) == 2:
            source_indices = ops.reshape(source_indices, (-1,))
        return ops.scatter_update(
            flat_text,
            ops.expand_dims(ops.cast(source_indices, "int32"), axis=-1),
            source_embeddings,
        )

    def call(
        self,
        text_embeddings,
        vision_embeddings=None,
        vision_indices=None,
        audio_embeddings=None,
        audio_indices=None,
    ):
        batch_size = ops.shape(text_embeddings)[0]
        seq_len = ops.shape(text_embeddings)[1]
        flat = ops.reshape(text_embeddings, (-1, self.hidden_dim))
        if vision_embeddings is not None and vision_indices is not None:
            flat = self._scatter(flat, vision_embeddings, vision_indices)
        if audio_embeddings is not None and audio_indices is not None:
            flat = self._scatter(flat, audio_embeddings, audio_indices)
        return ops.reshape(flat, (batch_size, seq_len, self.hidden_dim))

    def compute_output_spec(
        self,
        text_embeddings,
        vision_embeddings=None,
        vision_indices=None,
        audio_embeddings=None,
        audio_indices=None,
    ):
        return keras.KerasTensor(
            shape=text_embeddings.shape, dtype=text_embeddings.dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config
