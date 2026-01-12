import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder


@keras_hub_export("keras_hub.models.Llama3VisionEncoder")
class Llama3VisionEncoder(keras.layers.Layer):
    """Vision encoder for the Llama 3.2 Vision model.

    This layer implements a SigLIP-style Vision Transformer with support for
    both single-stage and two-stage (local + global) architectures.

    Args:
        hidden_dim: int. The size of the transformer hidden state.
        num_layers: int. The number of transformer layers (single-stage mode).
        num_heads: int. The number of attention heads.
        intermediate_dim: int. The output dimension of the feedforward network.
        patch_size: int. The size of each square image patch.
        image_size: int. The input image resolution. Defaults to `560`.
        num_channels: int. The number of input channels. Defaults to `3`.
        local_layers: int. Number of local encoder layers (two-stage mode).
            Defaults to `None`.
        global_layers: int. Number of global encoder layers (two-stage mode).
            Defaults to `None`.
        activation: str. The activation function. Defaults to `"gelu"`.
        dropout: float. Dropout rate. Defaults to `0.0`.
        attention_dropout: float. Attention dropout rate. Defaults to `0.0`.
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
    output = encoder(images)  # Shape: (1, 1600, 1280)
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
        local_layers=None,
        global_layers=None,
        activation="gelu",
        dropout=0.0,
        attention_dropout=0.0,
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
        self.local_layers = local_layers
        self.global_layers = global_layers
        self.activation = activation
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        self.is_two_stage = (
            local_layers is not None and global_layers is not None
        )
        self.num_patches = (image_size // patch_size) ** 2

        # === Layers ===
        self.patch_embedding = layers.Conv2D(
            filters=hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name="patch_embedding",
        )
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=hidden_dim,
            name="position_embedding",
        )

        if self.is_two_stage:
            self.local_transformer_layers = [
                TransformerEncoder(
                    intermediate_dim=intermediate_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_epsilon=layer_norm_epsilon,
                    normalize_first=True,
                    name=f"local_transformer_layer_{i}",
                )
                for i in range(local_layers)
            ]
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
        else:
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

        self.layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name="post_layer_norm",
        )

    def build(self, input_shape):
        self.patch_embedding.build(input_shape)
        self.position_embedding.build((self.num_patches,))

        transformer_input_shape = (
            input_shape[0],
            self.num_patches,
            self.hidden_dim,
        )

        if self.is_two_stage:
            for layer in self.local_transformer_layers:
                layer.build(transformer_input_shape)
            for layer in self.global_transformer_layers:
                layer.build(transformer_input_shape)
        else:
            for layer in self.transformer_layers:
                layer.build(transformer_input_shape)

        self.layer_norm.build(transformer_input_shape)
        super().build(input_shape)

    def call(self, images):
        """Forward pass of the vision encoder.

        Args:
            images: Tensor of shape `(batch, height, width, channels)`.

        Returns:
            Tensor of shape `(batch, num_patches, hidden_dim)`.
        """
        embeddings = self.patch_embedding(images)
        batch_size = ops.shape(embeddings)[0]
        embeddings = ops.reshape(embeddings, (batch_size, -1, self.hidden_dim))

        positions = ops.arange(start=0, stop=self.num_patches, step=1)
        pos_embeddings = self.position_embedding(positions)
        x = embeddings + pos_embeddings

        if self.is_two_stage:
            for layer in self.local_transformer_layers:
                x = layer(x)
            for layer in self.global_transformer_layers:
                x = layer(x)
        else:
            for layer in self.transformer_layers:
                x = layer(x)

        x = self.layer_norm(x)
        return x

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
                "local_layers": self.local_layers,
                "global_layers": self.global_layers,
                "activation": self.activation,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def freeze_local_encoder(self):
        """Freeze local encoder layers (two-stage mode only)."""
        if not self.is_two_stage:
            raise ValueError(
                "freeze_local_encoder() requires two-stage mode. "
                "Set local_layers and global_layers."
            )
        self.patch_embedding.trainable = False
        self.position_embedding.trainable = False
        for layer in self.local_transformer_layers:
            layer.trainable = False

    def freeze_global_encoder(self):
        """Freeze global encoder layers (two-stage mode only)."""
        if not self.is_two_stage:
            raise ValueError(
                "freeze_global_encoder() requires two-stage mode. "
                "Set local_layers and global_layers."
            )
        for layer in self.global_transformer_layers:
            layer.trainable = False
        self.layer_norm.trainable = False

    def freeze_all(self):
        """Freeze all encoder weights."""
        self.trainable = False

    def unfreeze_all(self):
        """Unfreeze all encoder weights."""
        self.trainable = True
        self.patch_embedding.trainable = True
        self.position_embedding.trainable = True
        self.layer_norm.trainable = True

        if self.is_two_stage:
            for layer in self.local_transformer_layers:
                layer.trainable = True
            for layer in self.global_transformer_layers:
                layer.trainable = True
        else:
            for layer in self.transformer_layers:
                layer.trainable = True
