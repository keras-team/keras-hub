import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.blip2.blip2_custom_opt import Blip2CustomOPT
from keras_hub.src.models.blip2.blip2_qformer import Blip2QFormer
from keras_hub.src.models.blip2.blip2_vision_encoder import Blip2VisionEncoder


def _build_if_config(cls, arg):
    if isinstance(arg, dict):
        return keras.layers.deserialize(arg)
    return arg


@keras_hub_export("keras_hub.models.Blip2Backbone")
class Blip2Backbone(Backbone):
    """BLIP-2 core network with hyperparameters.

    Args:
        vision_encoder: A `Blip2VisionEncoder` instance or config dict.
        qformer: A `Blip2QFormer` instance or config dict.
        language_model: A `Blip2CustomOPT` instance or config dict.
        dtype: string or `keras.mixed_precision.DTypePolicy`.
        **kwargs: Standard Keras Model arguments.
    """

    def __init__(
        self,
        vision_encoder,
        qformer,
        language_model,
        dtype=None,
        **kwargs,
    ):
        # Accept config dicts or live instances.
        self.vision_encoder = _build_if_config(
            Blip2VisionEncoder, vision_encoder
        )
        self.qformer = _build_if_config(Blip2QFormer, qformer)
        self.language_model = _build_if_config(Blip2CustomOPT, language_model)

        # === Functional Graph ===
        token_ids_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="bool", name="padding_mask"
        )

        inputs = {
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }

        if self.vision_encoder is not None:
            image_size = self.vision_encoder.image_size
            images_input = keras.Input(
                shape=(image_size, image_size, 3), name="images"
            )
            inputs["images"] = images_input

            vision_features = self.vision_encoder(images_input)
            query_output = self.qformer(vision_features)
        else:
            query_output = None

        lm_inputs = {
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }
        if query_output is not None:
            lm_inputs["qformer_features"] = query_output

        output = self.language_model(lm_inputs)

        super().__init__(
            inputs=inputs,
            outputs=output,
            dtype=dtype,
            **kwargs,
        )

    @property
    def num_query_tokens(self):
        if self.qformer is not None:
            return self.qformer.num_query_tokens
        return 0

    @property
    def qformer_hidden_dim(self):
        if self.qformer is not None:
            return self.qformer.hidden_dim
        return 0

    @property
    def token_embedding(self):
        return self.language_model.token_embedding

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder": keras.layers.serialize(self.vision_encoder),
                "qformer": keras.layers.serialize(self.qformer),
                "language_model": keras.layers.serialize(self.language_model),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if config.get("vision_encoder") is not None:
            config["vision_encoder"] = keras.layers.deserialize(
                config["vision_encoder"]
            )
        if config.get("qformer") is not None:
            config["qformer"] = keras.layers.deserialize(config["qformer"])
        config["language_model"] = keras.layers.deserialize(
            config["language_model"]
        )
        return cls(**config)
