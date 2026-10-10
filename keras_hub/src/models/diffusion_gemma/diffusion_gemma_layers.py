import keras
from keras import ops

from keras_hub.src.models.gemma4.gemma4_layers import Gemma4InterleaveEmbeddings
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4VNorm
from keras_hub.src.models.gemma4.gemma4_layers import RMSNormalization


class DiffusionGemmaSelfConditioning(keras.layers.Layer):
    """Self-conditioning layer for DiffusionGemma decoder.

    This layer refines the canvas embeddings at the start of each decoder
    denoising step by incorporating information from the logits predicted at
    the previous step.  It is the only parameter that exists in the decoder
    but NOT in the encoder (all other backbone weights are shared).

    On the first denoising step (when `prev_logits` is `None`), the layer
    is skipped and `canvas_embeds` is returned unchanged.

    Architecture:
        soft_embeds = softmax(prev_logits) @ embed_tokens_weight * embed_scale
        x           = pre_norm(soft_embeds)
        gate        = gelu(gate_proj(x), approximate=True)
        out         = down_proj(gate * up_proj(x))
        return post_norm(canvas_embeds + out)

    `pre_norm` has a learnable scale (standard RMSNorm).
    `post_norm` has NO learnable scale (pure L2 normalisation via
    `Gemma4VNorm`), matching the HF checkpoint which stores no
    `post_norm.weight` tensor.

    Args:
        hidden_dim: int. Dimensionality of the model's hidden representations.
        intermediate_dim: int. Intermediate dimension of the gated MLP.
        epsilon: float. Epsilon for RMS normalization layers. Defaults to
            `1e-6`.

    Call arguments:
        canvas_embeds: float tensor of shape `(B, canvas_length, hidden_dim)`.
            Raw canvas token embeddings from the current step.
        prev_logits: float tensor of shape `(B, canvas_length, vocab_size)`
            from the previous denoising step, or `None` on the first step.

    Returns:
        Float tensor of shape `(B, canvas_length, hidden_dim)`.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.epsilon = epsilon

        self.pre_norm = RMSNormalization(
            epsilon=epsilon,
            dtype=self.dtype_policy,
            name="pre_norm",
        )
        self.gate_proj = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="gate_proj",
        )
        self.up_proj = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="up_proj",
        )
        self.down_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="down_proj",
        )
        # post_norm has no learnable scale — matches HF which stores no
        # post_norm.weight tensor for this module.
        self.post_norm = Gemma4VNorm(
            epsilon=epsilon,
            dtype=self.dtype_policy,
            name="post_norm",
        )

    def build(self, input_shape):
        # input_shape is the shape of
        # canvas_embeds: (B, canvas_length, hidden_dim)
        self.pre_norm.build(input_shape)
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)

        gate_out_shape = self.gate_proj.compute_output_shape(input_shape)
        self.down_proj.build(gate_out_shape)

        down_out_shape = self.down_proj.compute_output_shape(gate_out_shape)
        self.post_norm.build(down_out_shape)

        self.built = True

    def call(self, canvas_embeds, prev_logits):
        if prev_logits is None:
            return self.post_norm(canvas_embeds)

        # Soft token embeddings: weighted combination of embedding rows.
        embed_tokens_weight = self._token_embedding_layer.embeddings
        embed_scale = ops.cast(
            ops.sqrt(ops.cast(self.hidden_dim, "float32")),
            embed_tokens_weight.dtype,
        )
        prev_logits = ops.cast(prev_logits, embed_tokens_weight.dtype)
        probs = ops.softmax(ops.cast(prev_logits, "float32"), axis=-1)
        probs = ops.cast(probs, embed_tokens_weight.dtype)
        # (B, canvas_length, vocab_size) x (vocab_size, hidden_dim)
        soft_embeds = ops.matmul(probs, embed_tokens_weight)
        soft_embeds = soft_embeds * embed_scale
        soft_embeds = ops.cast(soft_embeds, self.compute_dtype)

        x = self.pre_norm(soft_embeds)
        gate = keras.activations.gelu(self.gate_proj(x), approximate=True)
        out = self.down_proj(gate * self.up_proj(x))

        return self.post_norm(canvas_embeds + out)

    def compute_output_shape(self, input_shape):
        # Output shape matches canvas_embeds shape.
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            return input_shape[0]
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "epsilon": self.epsilon,
            }
        )
        return config


class DiffusionGemmaInterleaveEmbeddings(Gemma4InterleaveEmbeddings):
    """Places image embeddings at the image placeholder positions.

    Each image gets a number of placeholders equal to its real soft-token
    count. The count depends on the image aspect ratio. The vision encoder
    always returns `num_vision_tokens_per_image` pooled bins per image. The
    bins after the real count are padding bins. This layer moves all padding
    bins after the real bins of every image, so that the real bins fill the
    placeholders in order. HF drops the padding bins with `pooler_mask`
    before the scatter.

    Args:
        num_vision_tokens_per_image: int. Number of pooled bins per image.
        pool_size: int. Spatial pooling factor of the vision encoder.

    Call arguments:
        image_embeddings: float tensor of shape
            `(batch_size, num_images, num_vision_tokens_per_image, dim)`.
        text_embeddings: float tensor of shape `(batch_size, seq_len, dim)`.
        vision_indices: int tensor of shape `(batch_size, num_placeholders)`.
        pixel_position_ids: int tensor of shape
            `(batch_size, num_images, num_patches, 2)`. Padding patches have
            the position `(-1, -1)`.
    """

    def __init__(
        self, num_vision_tokens_per_image, pool_size, dtype=None, **kwargs
    ):
        super().__init__(
            num_vision_tokens_per_image=num_vision_tokens_per_image,
            dtype=dtype,
            **kwargs,
        )
        self.pool_size = pool_size

    def build(
        self,
        image_embeddings_shape,
        text_embeddings_shape=None,
        vision_indices_shape=None,
        pixel_position_ids_shape=None,
    ):
        self.built = True

    def _pack_real_bins(self, image_embeddings, pixel_position_ids):
        """Move the real bins of all images ahead of all padding bins."""
        batch_size, num_images, num_bins, dim = ops.shape(image_embeddings)
        # The real bins of an image are its first `real_patches //
        # pool_size**2` bins. This holds because the image converter makes
        # each patch-grid side a multiple of `pool_size`, and the vision
        # pooler places the pooled `w * h` grid of each image in bins
        # `[0, w * h)`.
        is_real_patch = ops.any(ops.not_equal(pixel_position_ids, -1), axis=-1)
        real_counts = ops.sum(ops.cast(is_real_patch, "int32"), axis=-1) // (
            self.pool_size**2
        )
        bin_ids = ops.arange(num_bins, dtype="int32")
        is_real_bin = ops.less(
            bin_ids[None, None, :], ops.expand_dims(real_counts, axis=-1)
        )
        is_real_bin = ops.reshape(
            is_real_bin, (batch_size, num_images * num_bins)
        )
        # Unique sort keys keep the order stable on every backend.
        positions = ops.arange(num_images * num_bins, dtype="int32")
        sort_keys = ops.where(
            is_real_bin, positions, positions + num_images * num_bins
        )
        order = ops.argsort(sort_keys, axis=-1)
        flat_embeddings = ops.reshape(
            image_embeddings, (batch_size, num_images * num_bins, dim)
        )
        flat_embeddings = ops.take_along_axis(
            flat_embeddings, ops.expand_dims(order, axis=-1), axis=1
        )
        return ops.reshape(
            flat_embeddings, (batch_size, num_images, num_bins, dim)
        )

    def call(
        self,
        image_embeddings,
        text_embeddings,
        vision_indices,
        pixel_position_ids,
    ):
        if ops.shape(image_embeddings)[1] != 0:
            image_embeddings = self._pack_real_bins(
                image_embeddings, pixel_position_ids
            )
        return super().call(image_embeddings, text_embeddings, vision_indices)

    def compute_output_shape(
        self,
        image_embeddings_shape,
        text_embeddings_shape,
        vision_indices_shape=None,
        pixel_position_ids_shape=None,
    ):
        return text_embeddings_shape

    def get_config(self):
        config = super().get_config()
        config.update({"pool_size": self.pool_size})
        return config
