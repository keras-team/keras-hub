"""Llama 3.2 Vision Encoder with optional two-stage architecture."""

import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder


@keras_hub_export("keras_hub.models.Llama3VisionEncoder")
class Llama3VisionEncoder(keras.layers.Layer):
    """Vision Encoder for the Llama 3.2 Vision model.

    This encoder implements SigLIP-style Vision Transformer architecture
    with support for both single-stage and two-stage (local + global)
    processing as used in Llama 3.2 Vision.

    Single-stage mode:
        - All transformer layers process patches uniformly.

    Two-stage mode (Meta's architecture):
        - Stage 1 (Local): Processes patches with local attention.
        - Stage 2 (Global): Aggregates information globally.

    Args:
        hidden_dim: int. The size of the transformer hidden state.
        num_layers: int. Total number of transformer layers (single-stage mode).
        num_heads: int. The number of attention heads.
        intermediate_dim: int. The output dimension of the feedforward network.
        patch_size: int. The size of each square image patch.
        num_channels: int. The number of input channels. Defaults to 3.
        image_size: int. The width and height of input images. Defaults to 560.
        local_layers: int. Number of local encoder layers (two-stage mode).
            If provided with global_layers, enables two-stage processing.
        global_layers: int. Number of global encoder layers (two-stage mode).
        activation: str. The activation function. Defaults to "gelu".
        dropout: float. Dropout rate. Defaults to 0.0.
        attention_dropout: float. Attention dropout rate. Defaults to 0.0.
        layer_norm_epsilon: float. Layer norm epsilon. Defaults to 1e-6.
        dtype: Data type for computations and weights.
    """

    def __init__(
        self,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        patch_size,
        num_channels=3,
        image_size=560,
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
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.local_layers = local_layers
        self.global_layers = global_layers
        self.activation = activation
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        # Determine if using two-stage architecture
        self.is_two_stage = (
            local_layers is not None and global_layers is not None
        )

        # 1. Patch Embedding
        self.patch_embedding = layers.Conv2D(
            filters=hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name="patch_embedding",
        )

        # 2. Positional Embedding
        self.num_patches = (image_size // patch_size) ** 2
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=hidden_dim,
            name="position_embedding",
        )

        # 3. Transformer Layers
        if self.is_two_stage:
            # Two-stage architecture: Local + Global
            self.local_transformer_layers = []
            for i in range(local_layers):
                self.local_transformer_layers.append(
                    TransformerEncoder(
                        intermediate_dim=intermediate_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        activation=activation,
                        layer_norm_epsilon=layer_norm_epsilon,
                        normalize_first=True,
                        name=f"local_transformer_layer_{i}",
                    )
                )

            self.global_transformer_layers = []
            for i in range(global_layers):
                self.global_transformer_layers.append(
                    TransformerEncoder(
                        intermediate_dim=intermediate_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        activation=activation,
                        layer_norm_epsilon=layer_norm_epsilon,
                        normalize_first=True,
                        name=f"global_transformer_layer_{i}",
                    )
                )
        else:
            # Single-stage architecture
            self.transformer_layers = []
            for i in range(num_layers):
                self.transformer_layers.append(
                    TransformerEncoder(
                        intermediate_dim=intermediate_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        activation=activation,
                        layer_norm_epsilon=layer_norm_epsilon,
                        normalize_first=True,
                        name=f"transformer_layer_{i}",
                    )
                )

        # 4. Final Layer Norm
        self.layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name="post_layer_norm",
        )

    def build(self, input_shape):
        # Build patch embedding
        self.patch_embedding.build(input_shape)

        # Build position embedding
        self.position_embedding.build((self.num_patches,))

        # Build transformer layers
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

        # Build final layer norm
        self.layer_norm.build(transformer_input_shape)

        super().build(input_shape)

    def call(self, images):
        """Forward pass of the Vision Encoder.

        Args:
            images: Tensor of shape (batch, height, width, channels).

        Returns:
            Tensor of shape (batch, num_patches, hidden_dim).
        """
        # 1. Create Patches
        embeddings = self.patch_embedding(images)

        # 2. Flatten patches into sequence
        batch_size = ops.shape(embeddings)[0]
        embeddings = ops.reshape(embeddings, (batch_size, -1, self.hidden_dim))

        # 3. Add Positional Information
        positions = ops.arange(start=0, stop=self.num_patches, step=1)
        pos_embeddings = self.position_embedding(positions)
        x = embeddings + pos_embeddings

        # 4. Process through Transformer Stack
        if self.is_two_stage:
            # Stage 1: Local processing
            for layer in self.local_transformer_layers:
                x = layer(x)

            # Stage 2: Global processing
            for layer in self.global_transformer_layers:
                x = layer(x)
        else:
            # Single-stage processing
            for layer in self.transformer_layers:
                x = layer(x)

        # 5. Final Normalization
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
                "num_channels": self.num_channels,
                "image_size": self.image_size,
                "local_layers": self.local_layers,
                "global_layers": self.global_layers,
                "activation": self.activation,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    # =========================================================================
    # Fine-Tuning Utilities
    # =========================================================================

    def freeze_local_encoder(self):
        """Freeze local encoder layers while keeping global encoder trainable.

        This is a common fine-tuning strategy where the lower-level local
        feature extraction is frozen while the global aggregation layers
        are fine-tuned for the specific task.

        Only applicable in two-stage mode.

        Example:
        ```python
        encoder = Llama3VisionEncoder(
            hidden_dim=1280, num_layers=40,
            local_layers=32, global_layers=8, ...
        )
        encoder.freeze_local_encoder()
        # Now only global_transformer_layers are trainable
        ```
        """
        if not self.is_two_stage:
            raise ValueError(
                "freeze_local_encoder() is only available in two-stage mode. "
                "Set local_layers and global_layers in the config."
            )

        # Freeze patch and position embeddings
        self.patch_embedding.trainable = False
        self.position_embedding.trainable = False

        # Freeze local transformer layers
        for layer in self.local_transformer_layers:
            layer.trainable = False

    def freeze_global_encoder(self):
        """Freeze global encoder layers while keeping local encoder trainable.

        This is useful when you want to fine-tune lower-level features
        while keeping the global aggregation fixed.

        Only applicable in two-stage mode.
        """
        if not self.is_two_stage:
            raise ValueError(
                "freeze_global_encoder() is only available in two-stage mode. "
                "Set local_layers and global_layers in the config."
            )

        # Freeze global transformer layers
        for layer in self.global_transformer_layers:
            layer.trainable = False

        # Freeze final layer norm (part of global processing)
        self.layer_norm.trainable = False

    def freeze_all(self):
        """Freeze the entire vision encoder.

        Useful when fine-tuning only the language model while keeping
        the vision encoder fixed.
        """
        self.trainable = False

    def unfreeze_all(self):
        """Unfreeze the entire vision encoder.

        Restores all layers to trainable state.
        """
        self.trainable = True

        # Explicitly set all sublayers in case of nested freezing
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

    def get_trainable_layers_summary(self):
        """Get a summary of trainable vs frozen layers.

        Returns:
            Dict with layer counts and trainable status.
        """
        summary = {
            "patch_embedding": self.patch_embedding.trainable,
            "position_embedding": self.position_embedding.trainable,
            "layer_norm": self.layer_norm.trainable,
        }

        if self.is_two_stage:
            local_trainable = sum(
                1 for l in self.local_transformer_layers if l.trainable
            )
            global_trainable = sum(
                1 for l in self.global_transformer_layers if l.trainable
            )
            summary["local_layers"] = (
                f"{local_trainable}/{len(self.local_transformer_layers)}"
            )
            summary["global_layers"] = (
                f"{global_trainable}/{len(self.global_transformer_layers)}"
            )
        else:
            layers_trainable = sum(
                1 for l in self.transformer_layers if l.trainable
            )
            summary["transformer_layers"] = (
                f"{layers_trainable}/{len(self.transformer_layers)}"
            )

        return summary
