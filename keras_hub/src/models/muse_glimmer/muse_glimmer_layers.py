import keras
from keras import ops


class MuseGlimmerRMSNorm(keras.layers.Layer):
    """Scaleless (or optionally scaled) RMSNorm.

    Matches HF's `MuseGlimmerRMSNorm`: used, with `with_scale=False`, for
    QK-norm, the normed token embedding, and the final sequence norm; and,
    with `with_scale=True`, as a plain scaled RMSNorm elsewhere.

    Args:
        eps: float. Epsilon added inside the reciprocal sqrt.
        with_scale: bool. Whether to learn a per-channel scale. Defaults to
            `True`.
    """

    def __init__(self, eps=1e-6, with_scale=True, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.with_scale = with_scale

    def build(self, input_shape):
        if self.with_scale:
            self.scale = self.add_weight(
                name="scale",
                shape=(input_shape[-1],),
                initializer="ones",
                dtype=self.variable_dtype,
            )
        self.built = True

    def call(self, x):
        input_dtype = x.dtype
        x = ops.cast(x, "float32")
        mean_squared = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.power(mean_squared + self.eps, -0.5)
        if self.with_scale:
            x = x * ops.cast(self.scale, "float32")
        return ops.cast(x, input_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps, "with_scale": self.with_scale})
        return config


class MuseGlimmerCenteredRMSNorm(keras.layers.Layer):
    """ "Centered" RMSNorm: `norm(x) * (1 + weight)`, weight init at zero.

    Matches HF's `MuseGlimmerTextCenteredRMSNorm`. Used for the four
    per-decoder-layer sandwich norms.

    Args:
        eps: float. Epsilon added inside the reciprocal sqrt.
    """

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(input_shape[-1],),
            initializer="zeros",
            dtype=self.variable_dtype,
        )
        self.built = True

    def call(self, x):
        input_dtype = x.dtype
        x = ops.cast(x, "float32")
        mean_squared = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(mean_squared + self.eps)
        x = x * (1.0 + ops.cast(self.scale, "float32"))
        return ops.cast(x, input_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


class MuseGlimmerInterleaveEmbeddings(keras.layers.Layer):
    """Scatter vision token embeddings into the text embedding sequence.

    KerasHub equivalent of HF's
    `inputs_embeds.masked_scatter(image_mask, image_embeds)`, using
    precomputed flat `vision_indices` (built by the preprocessor) so the
    scatter is representable inside the Functional API graph.

    Args:
        hidden_dim: int. The embedding dimension.
    """

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def call(self, image_embeddings, text_embeddings, vision_indices):
        batch_size = ops.shape(text_embeddings)[0]
        seq_len = ops.shape(text_embeddings)[1]

        if len(ops.shape(image_embeddings)) == 3:
            image_embeddings = ops.reshape(
                image_embeddings, (-1, self.hidden_dim)
            )
        if len(ops.shape(vision_indices)) == 2:
            vision_indices = ops.reshape(vision_indices, (-1,))

        flat_text = ops.reshape(text_embeddings, (-1, self.hidden_dim))
        vision_indices = ops.cast(vision_indices, "int32")
        vision_indices = ops.expand_dims(vision_indices, axis=-1)

        flat_out = ops.scatter_update(
            flat_text, vision_indices, image_embeddings
        )
        return ops.reshape(flat_out, (batch_size, seq_len, self.hidden_dim))

    def compute_output_spec(
        self, image_embeddings, text_embeddings, vision_indices
    ):
        return keras.KerasTensor(
            shape=text_embeddings.shape, dtype=text_embeddings.dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config
