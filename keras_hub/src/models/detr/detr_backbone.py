import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.detr.detr_layers import DetrSinePositionEmbedding
from keras_hub.src.models.detr.detr_layers import DetrTransformerEncoder
from keras_hub.src.models.detr.detr_layers import ExpandMaskLayer
from keras_hub.src.models.detr.detr_layers import ResizeMaskLayer


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

        # === Functional Model ===
        image_sum = ops.sum(image_input, axis=-1)
        mask_binary = ops.cast(ops.not_equal(image_sum, 0), image_input.dtype)
        mask_expanded = ops.expand_dims(mask_binary, axis=-1)

        # Resize mask to feature map size (need layer for dynamic shapes)
        resize_mask = ResizeMaskLayer()
        mask = resize_mask([mask_expanded, projected])

        # Generate position embeddings
        pos_embed = pos_embed_layer(mask[:, :, :, 0])
        pos_embed = layers.Permute((2, 3, 1))(pos_embed)

        projected_flat = layers.Reshape((-1, hidden_dim))(projected)
        pos_embed_flat = layers.Reshape((-1, hidden_dim))(pos_embed)
        mask_flat = layers.Reshape((-1,))(mask)

        # Expand mask for attention using helper layer
        expand_mask = ExpandMaskLayer()
        attention_mask = expand_mask(mask_flat)

        encoded_features = encoder(
            projected_flat,
            attention_mask=attention_mask,
            pos_embed=pos_embed_flat,
        )

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
