import keras
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.llama3.llama3_backbone import Llama3BackboneConfig


@keras_hub_export("keras_hub.models.Llama3VisionConfig")
class Llama3VisionConfig(Llama3BackboneConfig):
    """Configuration for the Llama 3.2 Vision Backbone.

    This class extends `Llama3BackboneConfig` to include parameters for the
    vision encoder and the gated cross-attention mechanism used in
    Llama 3.2 multimodal models (11B and 90B).

    Args:
        vision_encoder_config: dict or config instance. The configuration
            for the vision encoder (ViT-like architecture).
        vision_projection_dim: int. The dimension of the projection layer
            that maps vision features to the text embedding space.
        cross_attention_layers: list of int. The indices of the transformer
            layers that should include gated cross-attention blocks.
            For Llama 3.2 11B, this is typically every 4th layer.
        **kwargs: Arguments for the parent `Llama3BackboneConfig`.
    """

    def __init__(
        self,
        vision_encoder_config=None,
        vision_projection_dim=4096,
        cross_attention_layers=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_encoder_config = vision_encoder_config
        self.vision_projection_dim = vision_projection_dim
        # Default to empty list if generic Llama3 is initialized without vision
        self.cross_attention_layers = cross_attention_layers or []

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder_config": self.vision_encoder_config,
                "vision_projection_dim": self.vision_projection_dim,
                "cross_attention_layers": self.cross_attention_layers,
            }
        )
        return config