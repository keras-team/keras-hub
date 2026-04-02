import keras
from keras import ops


class RMSNormalization(keras.layers.Layer):
    """RMS Normalization layer.

    Normalizes inputs using their root mean square, then applies a learned
    scale to the normalized output.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(input_shape[-1],),
            initializer="ones",
        )
        self.built = True

    def call(self, x):
        # Always compute normalization in float32.
        x = ops.cast(x, "float32")
        scale = ops.cast(self.scale, "float32")
        var = ops.mean(ops.square(x), axis=-1, keepdims=True)
        normed_inputs = x * ops.power(var + self.epsilon, -0.5)
        normed_inputs = normed_inputs * scale
        return ops.cast(normed_inputs, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


class Gemma4VNorm(keras.layers.Layer):
    """Pure L2 normalization layer used for value vectors.

    This is a parameter-free RMS normalization (no learnable scale).
    Used for the v_norm in Gemma4 attention and the multimodal embedder
    post-projection norms.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.built = True

    def call(self, x):
        # Always compute normalization in float32.
        x = ops.cast(x, "float32")
        var = ops.mean(ops.square(x), axis=-1, keepdims=True)
        normed = x * ops.power(var + self.epsilon, -0.5)
        return ops.cast(normed, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


class Gemma4FrozenNorm(keras.layers.Layer):
    """RMS normalization with a frozen (non-trainable) scale.

    Matches HuggingFace's ``Gemma4RMSNorm(requires_grad=False,
    scale_shift=0.0)``: forward is ``normed * scale`` where *scale*
    is a non-trainable weight initialized to zeros (loaded from
    checkpoint).  Used in the audio conformer blocks.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            trainable=False,
            shape=(input_shape[-1],),
            initializer="zeros",
        )
        self.built = True

    def call(self, x):
        x = ops.cast(x, "float32")
        scale = ops.cast(self.scale, "float32")
        var = ops.mean(ops.square(x), axis=-1, keepdims=True)
        normed = x * ops.power(var + self.epsilon, -0.5)
        return ops.cast(normed * scale, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


class Gemma4MeanPooling(keras.layers.Layer):
    """Mean pooling layer that computes the average of token embeddings.

    This layer correctly handles variable-length sequences by ignoring
    padded tokens in the mean calculation, using a `padding_mask`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, padding_mask=None):
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
        if isinstance(input_shape, list):
            sequence_output_shape = input_shape[0]
        else:
            sequence_output_shape = input_shape
        return sequence_output_shape[:-2] + (sequence_output_shape[-1],)

    def get_config(self):
        return super().get_config()


class Gemma4InterleaveEmbeddings(keras.layers.Layer):
    """Places image embeddings in the correct position in an embedding sequence.

    For Gemma4, images can be in any position in the input sequence. In order
    to accomplish this, we have image placeholder tokens in the input
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
                vision encoder. Shape:
                `(batch_size, num_images_per_prompt,
                num_vision_tokens_per_image, embedding_dim)`.
            text_embeddings: tensor. Embeddings returned by the text embedding
                layer. Shape: `(batch_size, seq_length, embedding_dim)`.
            vision_indices: tensor. Indexes into `text_embeddings`, used to
                identify which places are supposed to be replaced by
                `image_embeddings`. Shape:
                `(batch_size,
                num_images_per_prompt * num_vision_tokens_per_image)`.

        Returns:
            Tensor of shape `(batch_size, seq_length, embedding_dim)`
            representing the reconstructed embeddings.
        """

        batch_size, seq_length, embedding_dim = ops.shape(text_embeddings)
        max_images = ops.shape(image_embeddings)[1]

        num_patches = ops.shape(image_embeddings)[2]

        # Fast path for text-only generation where max_images is 0
        if max_images == 0:
            return text_embeddings

        flat_text_embeddings = ops.reshape(
            text_embeddings, (batch_size * seq_length, embedding_dim)
        )
        to_add = ops.multiply(
            keras.ops.arange(batch_size, dtype="int32"), seq_length
        )
        to_add = ops.cast(ops.expand_dims(to_add, axis=-1), "int32")

        # Slice vision_indices to match the valid number of patches generated
        # (the pooler may output up to `max_images * num_patches` tokens, but
        # vision_indices only covers actual – possibly fewer – soft tokens).
        valid_vision_indices = vision_indices[:, : max_images * num_patches]

        # Trim image_embeddings to the same token count as valid_vision_indices
        # so that scatter_update receives matching updates and indices tensors.
        # We use the static shape attribute (which is concrete for numpy-backed
        # inputs and therefore a static constant in JAX JIT) rather than a
        # dynamic ops.shape() call, so no dynamic-slice restriction applies.
        num_actual = valid_vision_indices.shape[1]
        if num_actual is None:  # fallback when shape is not static
            num_actual = ops.shape(valid_vision_indices)[1]
        # Reshape to (B, all_tokens, H), clip, then flatten.
        all_img = ops.reshape(
            image_embeddings,
            (batch_size, max_images * num_patches, embedding_dim),
        )
        flat_image_embeddings = ops.reshape(
            all_img[:, :num_actual, :],
            (-1, embedding_dim),
        )
        vision_indices = ops.add(valid_vision_indices, to_add)

        vision_indices_shape = ops.shape(vision_indices)
        flat_vision_indices = ops.reshape(
            vision_indices,
            (vision_indices_shape[0] * vision_indices_shape[1], 1),
        )
        indices = ops.cast(flat_vision_indices, "int32")

        # Before reconstructing, store the 0th index so we can restore it.
        zeroth_index_text_embeddings = ops.take(
            flat_text_embeddings,
            indices=ops.squeeze(to_add, axis=-1),
            axis=0,
        )

        num_images_static = getattr(image_embeddings, "shape", [None])[0]
        if num_images_static is None or num_images_static > 0:
            # Reconstruct embeddings.
            reconstructed_embedding = ops.scatter_update(
                inputs=flat_text_embeddings,
                indices=indices,
                updates=flat_image_embeddings,
            )

            # Restore the original value at the 0th index (since vision_indices
            # are padded with 0 for samples with fewer images).
            reconstructed_embedding = ops.scatter_update(
                inputs=reconstructed_embedding,
                indices=to_add,
                updates=zeroth_index_text_embeddings,
            )
        else:
            reconstructed_embedding = flat_text_embeddings

        return ops.reshape(
            reconstructed_embedding, (batch_size, seq_length, embedding_dim)
        )

    def compute_output_shape(
        self,
        image_embeddings_shape,
        text_embeddings_shape,
        vision_indices_shape=None,
    ):
        # Output has the same shape as `text_embeddings`.
        return text_embeddings_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_vision_tokens_per_image": self.num_vision_tokens_per_image,
            }
        )
        return config


class Gemma4ClippableEinsumDense(keras.layers.Layer):
    """EinsumDense with clipping for inputs and outputs.

    Matches HF's `Gemma4ClippableLinear` when `use_clipped_linears=True`.
    Uses learnable (but non-trainable in standard usage, loaded from checkpoint)
    buffers for min/max values.
    """

    def __init__(
        self,
        equation,
        output_shape,
        use_clipped_linears=True,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.equation = equation
        self.output_shape = output_shape
        self.use_clipped_linears = use_clipped_linears
        self.kernel_initializer = kernel_initializer
        self.dense = keras.layers.EinsumDense(
            equation,
            output_shape=output_shape,
            kernel_initializer=kernel_initializer,
            dtype=self.dtype_policy,
            name="dense",
        )

    def build(self, input_shape):
        self.dense.build(input_shape)
        if self.use_clipped_linears:
            # We use add_weight to create these as non-trainable weights.
            # They will be loaded from the checkpoint.
            # Initialized to very large values so they don't clip unless loaded.
            self.input_min = self.add_weight(
                name="input_min",
                shape=(),
                initializer=keras.initializers.Constant(-65504.0),
                trainable=False,
            )
            self.input_max = self.add_weight(
                name="input_max",
                shape=(),
                initializer=keras.initializers.Constant(65504.0),
                trainable=False,
            )
            self.output_min = self.add_weight(
                name="output_min",
                shape=(),
                initializer=keras.initializers.Constant(-65504.0),
                trainable=False,
            )
            self.output_max = self.add_weight(
                name="output_max",
                shape=(),
                initializer=keras.initializers.Constant(65504.0),
                trainable=False,
            )
        self.built = True

    def call(self, x):
        if self.use_clipped_linears:
            x = ops.clip(
                x,
                ops.cast(self.input_min, x.dtype),
                ops.cast(self.input_max, x.dtype),
            )
        x = self.dense(x)
        if self.use_clipped_linears:
            x = ops.clip(
                x,
                ops.cast(self.output_min, x.dtype),
                ops.cast(self.output_max, x.dtype),
            )
        return x

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "equation": self.equation,
                "output_shape": self.output_shape,
                "use_clipped_linears": self.use_clipped_linears,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config
