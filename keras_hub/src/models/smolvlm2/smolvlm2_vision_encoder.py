from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


class SmolVLM2VisionEmbedding(layers.Layer):
    """Patch + position embedding for the SmolVLM2 vision encoder.

    Uses a Conv2D for patch extraction and fractional-coordinate bucket
    position embeddings to handle variable aspect ratios.

    Args:
        hidden_dim: int. Dimensionality of the embedding output.
        image_size: int. Expected image size (square).
        patch_size: int. Patch size for Conv2D extraction.
        num_channels: int. Number of input image channels.
        dtype: string or keras DTypePolicy.
    """

    def __init__(
        self,
        hidden_dim,
        image_size,
        patch_size,
        num_channels=3,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches

    def build(self, input_shape):
        self.patch_embedding = layers.Conv2D(
            filters=self.hidden_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            use_bias=True,
            dtype=self.dtype_policy,
            name="patch_embedding",
        )
        self.patch_embedding.build((None, None, None, self.num_channels))

        self.position_embedding = layers.Embedding(
            input_dim=self.num_positions,
            output_dim=self.hidden_dim,
            dtype=self.dtype_policy,
            name="position_embedding",
        )

        super().build(input_shape)

    def call(self, pixel_values, patch_attention_mask=None):
        """Compute patch embeddings with position encoding.

        Args:
            pixel_values: Tensor of shape (batch, height, width, channels).
            patch_attention_mask: Optional bool tensor of shape
                (batch, num_patches_h, num_patches_w) indicating
                valid (non-padded) patches.

        Returns:
            Tensor of shape (batch, num_patches, hidden_dim).
        """
        batch_size = ops.shape(pixel_values)[0]

        # Extract patches: (batch, H', W', hidden_dim)
        patch_embeds = self.patch_embedding(pixel_values)
        # Flatten spatial dims: (batch, H'*W', hidden_dim)
        ph = ops.shape(patch_embeds)[1]
        pw = ops.shape(patch_embeds)[2]
        embeddings = ops.reshape(
            patch_embeds, (batch_size, ph * pw, self.hidden_dim)
        )

        # Use pre-computed num_patches for position IDs. For standard
        # fixed-resolution input, all patches get simple sequential
        # position IDs.
        position_ids = ops.broadcast_to(
            ops.arange(self.num_patches, dtype="int32")[None, :],
            (batch_size, self.num_patches),
        )

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_channels": self.num_channels,
            }
        )
        return config


class SmolVLM2VisionAttention(layers.Layer):
    """Multi-head attention for the SmolVLM2 vision encoder.

    Standard multi-head attention without GQA or RoPE.

    Args:
        hidden_dim: int. Dimensionality of the model.
        num_heads: int. Number of attention heads.
        dropout: float. Attention dropout rate.
        dtype: string or keras DTypePolicy.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

    def build(self, input_shape):
        self.q_proj = layers.Dense(
            self.hidden_dim,
            dtype=self.dtype_policy,
            name="q_proj",
        )
        self.q_proj.build(input_shape)

        self.k_proj = layers.Dense(
            self.hidden_dim,
            dtype=self.dtype_policy,
            name="k_proj",
        )
        self.k_proj.build(input_shape)

        self.v_proj = layers.Dense(
            self.hidden_dim,
            dtype=self.dtype_policy,
            name="v_proj",
        )
        self.v_proj.build(input_shape)

        self.out_proj = layers.Dense(
            self.hidden_dim,
            dtype=self.dtype_policy,
            name="out_proj",
        )
        self.out_proj.build(input_shape)

        self._softmax = layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        super().build(input_shape)

    def call(self, hidden_states, attention_mask=None):
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        queries = ops.reshape(
            self.q_proj(hidden_states),
            (batch_size, seq_len, self.num_heads, self.head_dim),
        )
        keys = ops.reshape(
            self.k_proj(hidden_states),
            (batch_size, seq_len, self.num_heads, self.head_dim),
        )
        values = ops.reshape(
            self.v_proj(hidden_states),
            (batch_size, seq_len, self.num_heads, self.head_dim),
        )

        # Transpose to (batch, heads, seq, head_dim)
        queries = ops.transpose(queries, (0, 2, 1, 3))
        keys = ops.transpose(keys, (0, 2, 1, 3))
        values = ops.transpose(values, (0, 2, 1, 3))

        attn_weights = ops.matmul(queries, ops.transpose(keys, (0, 1, 3, 2)))
        attn_weights = attn_weights * ops.cast(self.scale, self.compute_dtype)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = self._softmax(attn_weights)
        attn_weights = ops.cast(attn_weights, self.compute_dtype)

        attn_output = ops.matmul(attn_weights, values)

        # Transpose back to original dimension order
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output, (batch_size, seq_len, self.hidden_dim)
        )
        attn_output = self.out_proj(attn_output)

        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }
        )
        return config


class SmolVLM2VisionMLP(layers.Layer):
    """Two-layer MLP for the SmolVLM2 vision encoder.

    Uses approximate GELU activation, matching HF's Idefics3VisionMLP
    with hidden_act="gelu_pytorch_tanh".

    Args:
        hidden_dim: int. Input/output dimensionality.
        intermediate_dim: int. Inner dimensionality.
        dtype: string or keras DTypePolicy.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

    def build(self, input_shape):
        self.fc1 = layers.Dense(
            self.intermediate_dim,
            dtype=self.dtype_policy,
            name="fc1",
        )
        self.fc1.build(input_shape)

        self.fc2 = layers.Dense(
            self.hidden_dim,
            dtype=self.dtype_policy,
            name="fc2",
        )
        self.fc2.build((input_shape[0], input_shape[1], self.intermediate_dim))

        super().build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = ops.gelu(hidden_states, approximate=True)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
            }
        )
        return config


class SmolVLM2VisionEncoderBlock(layers.Layer):
    """Pre-norm encoder block for the SmolVLM2 vision encoder.

    Matches HF's Idefics3EncoderLayer: LayerNorm → Attention → Residual
    → LayerNorm → MLP → Residual.

    Args:
        hidden_dim: int. Dimensionality of the model.
        num_heads: int. Number of attention heads.
        intermediate_dim: int. Inner dimensionality of the MLP.
        layer_norm_epsilon: float. Epsilon for LayerNorm.
        attention_dropout: float. Dropout for attention.
        dtype: string or keras DTypePolicy.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        layer_norm_epsilon=1e-6,
        attention_dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.attention_dropout = attention_dropout

    def build(self, input_shape):
        self.self_attn = SmolVLM2VisionAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.attention_dropout,
            dtype=self.dtype_policy,
            name="self_attn",
        )
        self.self_attn.build(input_shape)

        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm1",
        )
        self.layer_norm1.build(input_shape)

        self.mlp = SmolVLM2VisionMLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.mlp.build(input_shape)

        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm2",
        )
        self.layer_norm2.build(input_shape)

        super().build(input_shape)

    def call(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "attention_dropout": self.attention_dropout,
            }
        )
        return config


@keras_hub_export("keras_hub.models.SmolVLM2VisionEncoder")
class SmolVLM2VisionEncoder(Backbone):
    """SmolVLM2 vision encoder (Idefics3-style SigLIP ViT).

    A vision transformer that processes images into patch embeddings
    using fractional-coordinate position encoding for variable aspect
    ratios, followed by a stack of encoder blocks and a final
    LayerNorm.

    Args:
        image_size: int. Expected input image size (square).
        patch_size: int. Size of each image patch.
        hidden_dim: int. Dimensionality of the encoder.
        intermediate_dim: int. Inner dimensionality of the MLP.
        num_layers: int. Number of encoder blocks.
        num_heads: int. Number of attention heads.
        num_channels: int. Number of input channels. Defaults to `3`.
        layer_norm_epsilon: float. Epsilon for LayerNorm.
            Defaults to `1e-6`.
        attention_dropout: float. Attention dropout rate.
            Defaults to `0.0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`.

    Examples:
    ```python
    # Randomly initialized vision encoder.
    encoder = keras_hub.models.SmolVLM2VisionEncoder(
        image_size=384,
        patch_size=14,
        hidden_dim=1152,
        intermediate_dim=4304,
        num_layers=27,
        num_heads=16,
    )
    images = np.random.rand(1, 384, 384, 3).astype("float32")
    output = encoder({"pixel_values": images})
    # output.shape == (1, 729, 1152)
    ```
    """

    def __init__(
        self,
        image_size,
        patch_size,
        hidden_dim,
        intermediate_dim,
        num_layers,
        num_heads,
        num_channels=3,
        layer_norm_epsilon=1e-6,
        attention_dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.vision_embeddings = SmolVLM2VisionEmbedding(
            hidden_dim=hidden_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            dtype=dtype,
            name="embeddings",
        )

        self.encoder_blocks = []
        for i in range(num_layers):
            block = SmolVLM2VisionEncoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                layer_norm_epsilon=layer_norm_epsilon,
                attention_dropout=attention_dropout,
                dtype=dtype,
                name=f"encoder_block_{i}",
            )
            self.encoder_blocks.append(block)

        self.post_layernorm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="post_layernorm",
        )

        # === Functional Model ===
        pixel_values_input = layers.Input(
            shape=(image_size, image_size, num_channels),
            dtype="float32",
            name="pixel_values",
        )

        x = self.vision_embeddings(pixel_values_input)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.post_layernorm(x)

        super().__init__(
            inputs={"pixel_values": pixel_values_input},
            outputs=x,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.layer_norm_epsilon = layer_norm_epsilon
        self.attention_dropout = attention_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_channels": self.num_channels,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "attention_dropout": self.attention_dropout,
            }
        )
        return config
