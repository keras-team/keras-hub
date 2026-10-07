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
                concatenated ``(batch × seq_len)`` sequence.

        Returns:
            Tensor ``(batch, seq_len, hidden_dim)`` with visual tokens
            inserted at the specified positions.
        """
        batch_size = ops.shape(text_embeddings)[0]
        seq_len = ops.shape(text_embeddings)[1]

        # Handle batched image_embeddings from the functional graph.
        # Squeeze batch dim: (batch, N, hidden) → (N, hidden)
        if len(ops.shape(image_embeddings)) == 3:
            image_embeddings = ops.reshape(
                image_embeddings, (-1, self.hidden_dim)
            )
        if len(ops.shape(vision_indices)) == 2:
            vision_indices = ops.reshape(vision_indices, (-1,))

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


class Qwen3_5LayerNorm(keras.layers.Layer):
    """RMS normalization layer for Qwen3.5.

    Uses a ``(1 + weight)``-centered RMSNorm. Weights are initialized
    to zero so the effective scale starts at 1.0. Used for standard
    layer norms and Q/K norms throughout the model.

    Args:
        head_dim: int or None. Explicit dimension override.
        epsilon: float. Epsilon for numerical stability.
    """

    def __init__(self, head_dim=None, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = self.head_dim if self.head_dim else input_shape[-1]
        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(dim,),
            initializer="zeros",
            dtype=self.variable_dtype,
        )
        self.built = True

    def call(self, x):
        input_dtype = x.dtype
        x = ops.cast(x, "float32")
        var = ops.mean(ops.power(x, 2), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.epsilon)
        scale = ops.cast(self.scale, "float32")
        return ops.cast(x * (1.0 + scale), input_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_dim": self.head_dim,
                "epsilon": self.epsilon,
            }
        )
        return config


class Qwen3_5RMSNormGated(keras.layers.Layer):
    """Gated RMS normalization for GatedDeltaNet output.

    Unlike ``Qwen3_5LayerNorm`` which uses ``(1 + weight) * x``, this
    layer uses ``weight * x`` with weights initialized to ones, matching
    HF's ``Qwen3_5RMSNormGated``.

    Args:
        head_dim: int or None. Explicit dimension override.
        epsilon: float. Epsilon for numerical stability.
    """

    def __init__(self, head_dim=None, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = self.head_dim if self.head_dim else input_shape[-1]
        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(dim,),
            initializer="ones",
            dtype=self.variable_dtype,
        )
        self.built = True

    def call(self, x):
        input_dtype = x.dtype
        x = ops.cast(x, "float32")
        var = ops.mean(ops.power(x, 2), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.epsilon)
        scale = ops.cast(self.scale, "float32")
        return ops.cast(scale * x, input_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_dim": self.head_dim,
                "epsilon": self.epsilon,
            }
        )
        return config
