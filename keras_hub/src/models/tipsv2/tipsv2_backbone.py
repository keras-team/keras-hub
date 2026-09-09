"""TIPSv2 Backbone.

A dual-encoder contrastive vision-language backbone that combines a
DINOv2-style ViT vision encoder with a custom text transformer encoder.
"""

from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.TIPSv2Backbone")
class TIPSv2Backbone(Backbone):
    """TIPSv2 dual-encoder vision-language backbone.

    This backbone wraps a `TIPSv2VisionEncoder` and a `TIPSv2TextEncoder`
    to produce aligned vision and text embeddings for contrastive learning.
    The vision encoder outputs spatially-rich per-patch features that enable
    zero-shot segmentation in addition to global image understanding.

    The default constructor gives a fully customizable, randomly initialized
    model. To load preset architectures and weights, use `from_preset`.

    Args:
        vision_encoder: `TIPSv2VisionEncoder`. The vision encoder.
        text_encoder: `TIPSv2TextEncoder`. The text encoder.
        temperature: float. Learned contrastive temperature. Defaults to
            `0.01`.
        dtype: str or Policy. Dtype for computations. Defaults to `None`.

    Example:
    ```python
    import numpy as np
    import keras_hub

    vision_encoder = keras_hub.models.TIPSv2VisionEncoder(
        patch_size=14,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        image_shape=(448, 448, 3),
    )
    text_encoder = keras_hub.models.TIPSv2TextEncoder(
        vocabulary_size=32000,
        embedding_dim=768,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=3072,
    )
    backbone = keras_hub.models.TIPSv2Backbone(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
    )

    images = np.random.rand(1, 448, 448, 3).astype("float32")
    token_ids = np.ones((1, 64), dtype="int32")
    padding_mask = np.ones((1, 64), dtype="int32")

    outputs = backbone({
        "images": images,
        "token_ids": token_ids,
        "padding_mask": padding_mask,
    })
    print(outputs["vision_cls_embedding"].shape)    # (1, 768)
    print(outputs["vision_patch_embeddings"].shape)  # (1, 1024, 768)
    print(outputs["text_embedding"].shape)           # (1, 768)
    ```
    """

    def __init__(
        self,
        vision_encoder,
        text_encoder,
        temperature=0.01,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        # === Functional Model ===
        image_input = layers.Input(
            shape=vision_encoder.image_shape, name="images"
        )
        token_id_input = layers.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = layers.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Vision path.
        vision_outputs = self.vision_encoder({"images": image_input})
        cls_token = vision_outputs["cls_token"]  # (B, 1, D)
        patch_tokens = vision_outputs["patch_tokens"]  # (B, N, D)

        # Flatten CLS to (B, D).
        vision_cls_embedding = cls_token[:, 0, :]

        # Text path.
        text_embedding = self.text_encoder(
            {
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            }
        )  # (B, D)

        super().__init__(
            inputs={
                "images": image_input,
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs={
                "vision_cls_embedding": vision_cls_embedding,
                "vision_patch_embeddings": patch_tokens,
                "text_embedding": text_embedding,
            },
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.temperature = temperature

    def get_vision_embeddings(self, images):
        """Encode images and return CLS embedding.

        Args:
            images: Input tensor of shape `(B, H, W, C)` in `[0, 1]`.

        Returns:
            Vision CLS embedding of shape `(B, D)`.
        """
        outputs = self.vision_encoder({"images": images})
        return outputs["cls_token"][:, 0, :]

    def get_text_embeddings(self, token_ids, padding_mask):
        """Encode text and return text embedding.

        Args:
            token_ids: Input int tensor of shape `(B, seq_len)`.
            padding_mask: Input int tensor of shape `(B, seq_len)`.

        Returns:
            Text embedding of shape `(B, D)`.
        """
        return self.text_encoder(
            {"token_ids": token_ids, "padding_mask": padding_mask}
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder": layers.serialize(self.vision_encoder),
                "text_encoder": layers.serialize(self.text_encoder),
                "temperature": self.temperature,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()

        # Propagate dtype to submodels if needed.
        if "dtype" in config and config["dtype"] is not None:
            dtype_config = config["dtype"]
            if "dtype" not in config["vision_encoder"]["config"]:
                config["vision_encoder"]["config"]["dtype"] = dtype_config
            if "dtype" not in config["text_encoder"]["config"]:
                config["text_encoder"]["config"]["dtype"] = dtype_config

        config["vision_encoder"] = layers.deserialize(
            config["vision_encoder"], custom_objects=custom_objects
        )
        config["text_encoder"] = layers.deserialize(
            config["text_encoder"], custom_objects=custom_objects
        )
        return cls(**config)
