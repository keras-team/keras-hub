import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder


class GatedPositionalEmbedding(layers.Layer):
    """Gated positional embedding with tile support for Llama 3.2 Vision."""

    def __init__(self, num_patches, hidden_dim, max_num_tiles=4, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.max_num_tiles = max_num_tiles

        # Main position embedding
        self.embedding = self.add_weight(
            name="embedding",
            shape=(num_patches, hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        # Gate for positional embedding
        self.gate = self.add_weight(
            name="gate",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        # Tile embedding
        self.tile_embedding = layers.Embedding(
            input_dim=max_num_tiles,
            output_dim=hidden_dim,
            name="tile_embedding",
        )

    def call(self, x, tile_ids=None):
        pos_embed = self.embedding
        if tile_ids is not None:
            tile_embed = self.tile_embedding(tile_ids)
            pos_embed = pos_embed + tile_embed
        gate = ops.tanh(self.gate)
        return x + gate * pos_embed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "hidden_dim": self.hidden_dim,
                "max_num_tiles": self.max_num_tiles,
            }
        )
        return config


class AspectRatioEmbedding(layers.Layer):
    """Aspect ratio embedding with gating for tile positions."""

    def __init__(self, max_num_tiles, num_patches, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_num_tiles = max_num_tiles
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim

        self.embedding = layers.Embedding(
            input_dim=max_num_tiles,
            output_dim=num_patches * hidden_dim,
            name="embedding",
        )
        self.gate = self.add_weight(
            name="gate",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x, aspect_ratio_ids=None):
        if aspect_ratio_ids is None:
            return x
        embed = self.embedding(aspect_ratio_ids)
        embed = ops.reshape(embed, (-1, self.num_patches, self.hidden_dim))
        gate = ops.tanh(self.gate)
        return x + gate * embed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_num_tiles": self.max_num_tiles,
                "num_patches": self.num_patches,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


@keras_hub_export("keras_hub.models.Llama3VisionEncoder")
class Llama3VisionEncoder(keras.layers.Layer):
    """Vision encoder for the Llama 3.2 Vision model.

    This layer implements the MllamaVisionModel architecture with support for
    multi-tile images, gated positional embeddings, and two-stage encoding
    (local + global transformer layers).

    Args:
        hidden_dim: int. The size of the transformer hidden state.
        num_layers: int. The number of local transformer layers.
        num_heads: int. The number of attention heads.
        intermediate_dim: int. The output dimension of the feedforward network.
        patch_size: int. The size of each square image patch.
        image_size: int. The input image resolution. Defaults to `560`.
        num_channels: int. The number of input channels. Defaults to `3`.
        global_layers: int. Number of global encoder layers. Defaults to `8`.
        max_num_tiles: int. Maximum number of image tiles. Defaults to `4`.
        activation: str. The activation function. Defaults to `"gelu"`.
        dropout: float. Dropout rate. Defaults to `0.0`.
        layer_norm_epsilon: float. Layer norm epsilon. Defaults to `1e-6`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.

    Example:
    ```python
    encoder = keras_hub.models.Llama3VisionEncoder(
        hidden_dim=1280,
        num_layers=32,
        num_heads=16,
        intermediate_dim=5120,
        patch_size=14,
        image_size=560,
    )
    images = np.random.uniform(size=(1, 560, 560, 3))
    output = encoder(images)  # Shape: (1, num_patches, hidden_dim)
    ```
    """

    def __init__(
        self,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        patch_size,
        image_size=560,
        num_channels=3,
        global_layers=8,
        max_num_tiles=4,
        activation="gelu",
        dropout=0.0,
        layer_norm_epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        # === Config ===
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.global_layers_count = global_layers
        self.max_num_tiles = max_num_tiles
        self.activation = activation
        self.dropout_rate = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        self.num_patches = (image_size // patch_size) ** 2

        # === Layers ===
        # Patch embedding (Conv2D)
        self.patch_embedding = layers.Conv2D(
            filters=hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            use_bias=False,
            name="patch_embedding",
        )

        # Class embedding (learnable CLS token)
        self.class_embedding = self.add_weight(
            name="class_embedding",
            shape=(hidden_dim,),
            initializer="zeros",
            trainable=True,
        )

        # Gated positional embedding with tile support
        self.gated_positional_embedding = GatedPositionalEmbedding(
            num_patches=self.num_patches + 1,  # +1 for CLS token
            hidden_dim=hidden_dim,
            max_num_tiles=max_num_tiles,
            name="gated_positional_embedding",
        )

        # Pre/Post tile positional embeddings
        self.pre_tile_positional_embedding = AspectRatioEmbedding(
            max_num_tiles=max_num_tiles,
            num_patches=self.num_patches + 1,
            hidden_dim=hidden_dim,
            name="pre_tile_positional_embedding",
        )
        self.post_tile_positional_embedding = AspectRatioEmbedding(
            max_num_tiles=max_num_tiles,
            num_patches=self.num_patches + 1,
            hidden_dim=hidden_dim,
            name="post_tile_positional_embedding",
        )

        # Local transformer layers
        self.transformer_layers = [
            TransformerEncoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
                layer_norm_epsilon=layer_norm_epsilon,
                normalize_first=True,
                name=f"transformer_layer_{i}",
            )
            for i in range(num_layers)
        ]

        # Global transformer layers
        self.global_transformer_layers = [
            TransformerEncoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
                layer_norm_epsilon=layer_norm_epsilon,
                normalize_first=True,
                name=f"global_transformer_layer_{i}",
            )
            for i in range(global_layers)
        ]

        # Layer norms (pre and post)
        self.layernorm_pre = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name="layernorm_pre",
        )
        self.layernorm_post = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name="layernorm_post",
        )

    def build(self, input_shape):
        batch_shape = (input_shape[0],)
        self.patch_embedding.build(input_shape)

        # Build transformers with CLS token included
        transformer_shape = batch_shape + (
            self.num_patches + 1,
            self.hidden_dim,
        )
        for layer in self.transformer_layers:
            layer.build(transformer_shape)
        for layer in self.global_transformer_layers:
            layer.build(transformer_shape)

        self.layernorm_pre.build(transformer_shape)
        self.layernorm_post.build(transformer_shape)
        super().build(input_shape)

    def call(self, images, aspect_ratio_ids=None, tile_ids=None):
        """Forward pass of the vision encoder.

        Args:
            images: Tensor of shape `(batch, height, width, channels)`.
            aspect_ratio_ids: Optional tensor for aspect ratio embeddings.
            tile_ids: Optional tensor for tile embeddings.

        Returns:
            Tensor of shape `(batch, num_patches + 1, hidden_dim)`.
        """
        # Patch embedding
        embeddings = self.patch_embedding(images)
        batch_size = ops.shape(embeddings)[0]
        embeddings = ops.reshape(embeddings, (batch_size, -1, self.hidden_dim))

        # Add CLS token
        cls_token = ops.broadcast_to(
            self.class_embedding, (batch_size, 1, self.hidden_dim)
        )
        embeddings = ops.concatenate([cls_token, embeddings], axis=1)

        # Pre-tile positional embedding
        embeddings = self.pre_tile_positional_embedding(
            embeddings, aspect_ratio_ids
        )

        # Gated positional embedding
        embeddings = self.gated_positional_embedding(embeddings, tile_ids)

        # Local transformer layers
        for layer in self.transformer_layers:
            embeddings = layer(embeddings)

        # Post-tile positional embedding
        embeddings = self.post_tile_positional_embedding(
            embeddings, aspect_ratio_ids
        )

        # Pre layer norm for global layers
        embeddings = self.layernorm_pre(embeddings)

        # Global transformer layers
        for layer in self.global_transformer_layers:
            embeddings = layer(embeddings)

        # Post layer norm
        embeddings = self.layernorm_post(embeddings)

        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "num_channels": self.num_channels,
                "global_layers": self.global_layers_count,
                "max_num_tiles": self.max_num_tiles,
                "activation": self.activation,
                "dropout": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def freeze_local_encoder(self):
        """Freeze local encoder layers."""
        self.patch_embedding.trainable = False
        for layer in self.transformer_layers:
            layer.trainable = False

    def freeze_global_encoder(self):
        """Freeze global encoder layers."""
        for layer in self.global_transformer_layers:
            layer.trainable = False
        self.layernorm_pre.trainable = False
        self.layernorm_post.trainable = False

    def freeze_all(self):
        """Freeze all encoder weights."""
        self.trainable = False

    def unfreeze_all(self):
        """Unfreeze all encoder weights."""
        self.trainable = True
