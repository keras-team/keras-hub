import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone


@keras_hub_export("keras_hub.models.Llama3VisionBackbone")
class Llama3VisionBackbone(Backbone):
    """Llama 3.2 Vision Backbone model.

    This model combines a Vision Encoder (ViT) and a Llama 3 Text Decoder
    interleaved with Gated Cross-Attention layers.

    Args:
        config: `Llama3VisionConfig` instance.
    """

    def __init__(self, config, **kwargs):
        # TODO(Vivek1106-04): Implement the Vision Encoder integration.
        # This will initialize the vision tower and the text backbone.

        # Placeholder for input validation
        if config.vision_encoder_config is None:
            raise ValueError("`vision_encoder_config` must be provided.")

        super().__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        # TODO(Vivek1106-04): Implement the forward pass.
        # 1. Process images through Vision Encoder.
        # 2. Process text through Embedding.
        # 3. Pass through Decoder layers with Cross-Attention.
        return inputs

    def get_config(self):
        # serialization_lib requires a python dict, not a custom object
        config = super().get_config()
        config.update({"config": self.config.get_config()})
        return config

    @classmethod
    def from_config(cls, config):
        # We must manually deserialize the nested config object
        from keras_hub.src.models.llama3.llama3_vision_config import \
            Llama3VisionConfig

        config_data = config.pop("config")
        vision_config = Llama3VisionConfig(**config_data)
        return cls(config=vision_config, **config)
