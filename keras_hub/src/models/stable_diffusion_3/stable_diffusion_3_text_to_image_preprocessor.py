import keras
from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (  # noqa: E501
    StableDiffusion3Backbone,
)
from keras_hub.src.models.text_to_image_preprocessor import (
    TextToImagePreprocessor,
)


@keras_hub_export("keras_hub.models.StableDiffusion3TextToImagePreprocessor")
class StableDiffusion3TextToImagePreprocessor(TextToImagePreprocessor):
    """Stable Diffusion 3 text-to-image model preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.StableDiffusion3TextToImage`.

    For use with generation, the layer exposes one methods
    `generate_preprocess()`.

    Args:
        clip_l_preprocessor: A `keras_hub.models.CLIPPreprocessor` instance.
        clip_g_preprocessor: A `keras_hub.models.CLIPPreprocessor` instance.
        t5_preprocessor: A optional `keras_hub.models.T5Preprocessor` instance.
    """

    backbone_cls = StableDiffusion3Backbone

    def __init__(
        self,
        clip_l_preprocessor,
        clip_g_preprocessor,
        t5_preprocessor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clip_l_preprocessor = clip_l_preprocessor
        self.clip_g_preprocessor = clip_g_preprocessor
        self.t5_preprocessor = t5_preprocessor

    @property
    def sequence_length(self):
        """The padded length of model input sequences."""
        return self.clip_l_preprocessor.sequence_length

    def build(self, input_shape):
        self.built = True

    def generate_preprocess(self, x):
        token_ids = {}
        token_ids["clip_l"] = self.clip_l_preprocessor(x)["token_ids"]
        token_ids["clip_g"] = self.clip_g_preprocessor(x)["token_ids"]
        if self.t5_preprocessor is not None:
            token_ids["t5"] = self.t5_preprocessor(x)["token_ids"]
        return token_ids

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "clip_l_preprocessor": layers.serialize(
                    self.clip_l_preprocessor
                ),
                "clip_g_preprocessor": layers.serialize(
                    self.clip_g_preprocessor
                ),
                "t5_preprocessor": layers.serialize(self.t5_preprocessor),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        for layer_name in (
            "clip_l_preprocessor",
            "clip_g_preprocessor",
            "t5_preprocessor",
        ):
            if layer_name in config and isinstance(config[layer_name], dict):
                config[layer_name] = keras.layers.deserialize(
                    config[layer_name]
                )
        return cls(**config)
