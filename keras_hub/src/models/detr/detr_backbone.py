from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.detr.detr_layers import DetrSinePositionEmbedding
from keras_hub.src.models.detr.detr_layers import DetrTransformerEncoder
from keras_hub.src.models.detr.detr_layers import ExpandMaskLayer


@keras_hub_export("keras_hub.models.DETRBackbone")
class DETRBackbone(Backbone):
    """DETR backbone for feature extraction.

    Combines a CNN backbone (ResNet) with a transformer encoder to extract
    rich features for object detection. The backbone:
    1. Extracts features using ResNet
    2. Projects to hidden dimension via 1x1 convolution
    3. Adds 2D positional embeddings
    4. Processes through transformer encoder

    Args:
        image_encoder: keras.Model. CNN backbone, typically
            ResNetBackbone.from_preset("resnet_50_imagenet")
        hidden_dim: int. Model dimension (d_model), default 256
        num_encoder_layers: int. Number of encoder layers, default 6
        num_heads: int. Number of attention heads, default 8
        intermediate_size: int. FFN intermediate size, default 2048
        dropout: float. Dropout rate, default 0.1
        activation: str. Activation function, default "relu"
        image_shape: tuple. Input image shape, default (None, None, 3)

    Example:
    ```python
    # Create backbone
    resnet = keras_hub.models.ResNetBackbone.from_preset(
        "resnet_50_imagenet"
    )
    backbone = keras_hub.models.DETRBackbone(
        image_encoder=resnet,
        hidden_dim=256,
        num_encoder_layers=6,
    )

    # Extract features
    features = backbone(images)  # Returns dict with encoded_features, etc
    ```
    """

    def __init__(
        self,
        image_encoder,
        hidden_dim=256,
        num_encoder_layers=6,
        num_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        activation="relu",
        image_shape=(None, None, 3),
        **kwargs,
    ):
        # === Layers ===
        image_input = layers.Input(shape=image_shape, name="images")

        # Output shape: (batch, H/32, W/32, 2048) for ResNet-50
        features = image_encoder(image_input)

        input_proj = layers.Conv2D(hidden_dim, kernel_size=1, name="input_proj")
        projected = input_proj(features)

        # Get static shape for mask generation
        _, h, w, _ = projected.shape

        pos_embed_layer = DetrSinePositionEmbedding(
            embedding_dim=hidden_dim // 2,
            normalize=True,
        )
        encoder = DetrTransformerEncoder(
            num_layers=num_encoder_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            activation=activation,
            dropout_rate=dropout,
            attentiondropout_rate=dropout,
            norm_first=False,
            norm_epsilon=1e-5,
            use_bias=True,
            name="encoder",
        )
        expand_mask = ExpandMaskLayer(name="expand_mask")

        # === Functional Model ===
        # Generate mask (1 for valid, 0 for padding)
        # Resize input mask to feature map size
        mask = layers.Lambda(
            lambda x: ops.image.resize(
                ops.expand_dims(
                    ops.cast(ops.not_equal(ops.sum(x, axis=-1), 0), x.dtype),
                    axis=-1,
                ),
                (h, w),
                interpolation="nearest",
            ),
            name="generate_mask",
        )(image_input)

        # Generate position embeddings
        pos_embed = pos_embed_layer(mask[:, :, :, 0])
        # pos_embed shape: (batch, hidden_dim, h, w) -> (batch, h, w, hidden_dim)
        pos_embed = layers.Permute((2, 3, 1), name="transpose_pos_embed")(
            pos_embed
        )

        # Flatten spatial dimensions
        projected_flat = layers.Reshape(
            (-1, hidden_dim), name="flatten_projected"
        )(projected)
        pos_embed_flat = layers.Reshape(
            (-1, hidden_dim), name="flatten_pos_embed"
        )(pos_embed)
        mask_flat = layers.Reshape((-1,), name="flatten_mask")(mask)

        # Process through transformer encoder
        attention_mask = expand_mask(mask_flat)
        encoded_features = encoder(
            projected_flat,
            attention_mask=attention_mask,
            pos_embed=pos_embed_flat,
        )

        # Output: Dictionary containing all necessary info for decoder
        outputs = {
            "encoded_features": encoded_features,
            "pos_embed": pos_embed_flat,
            "mask": mask_flat,
        }

        super().__init__(inputs=image_input, outputs=outputs, **kwargs)

        # Store config
        self.image_encoder = image_encoder
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.activation = activation
        self.image_shape = image_shape

    def _generate_image_mask(self, images, target_shape):
        """Generate binary mask from images (1 for valid, 0 for padding).

        Args:
            images: Input images (batch, height, width, channels)
            target_shape: Target (height, width) for mask

        Returns:
            Binary mask (batch, height, width, 1)
        """
        # Sum across channels, mark non-zero as valid
        # This handles black padding (all zeros)
        mask = ops.not_equal(ops.sum(images, axis=-1), 0)
        mask = ops.cast(mask, images.dtype)
        mask = ops.expand_dims(mask, axis=-1)

        # Resize to feature map size
        mask = ops.image.resize(mask, target_shape, interpolation="nearest")
        return mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_encoder": keras.saving.serialize_keras_object(
                    self.image_encoder
                ),
                "hidden_dim": self.hidden_dim,
                "num_encoder_layers": self.num_encoder_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "dropout": self.dropout,
                "activation": self.activation,
                "image_shape": self.image_shape,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "image_encoder" in config and isinstance(
            config["image_encoder"], dict
        ):
            config["image_encoder"] = keras.layers.deserialize(
                config["image_encoder"]
            )
        return super().from_config(config)
