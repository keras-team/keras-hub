import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.MoondreamBackbone")
class MoondreamBackbone(Backbone):
    """
    The Moondream Backbone model.

    This model connects a vision encoder (SigLIP) and a text decoder (Phi-1.5)
    using a projection layer. It is designed for vision-language tasks where
    image features are projected into the text embedding space.

    Args:
        vision_encoder: A Keras model (e.g., SigLIP). The vision encoder
            responsible for processing input images.
        text_decoder: A Keras model (e.g., Phi-1.5). The text decoder
            responsible for generating text tokens.
        projection_dim: int. The dimension to project image features into.
            Defaults to `2048`.
        **kwargs: Standard Keras keyword arguments.

    Example:
    ```python
    import keras
    import numpy as np
    from keras_hub.src.models.moondream.moondream_backbone import (
        MoondreamBackbone
    )

    # 1. Create Mock Encoders
    # Vision Encoder: Maps (378, 378, 3) -> (729, 1152)
    image_input = keras.Input(shape=(378, 378, 3))
    vision_output = keras.layers.Lambda(
        lambda x: keras.ops.ones((keras.ops.shape(x)[0], 729, 1152))
    )(image_input)
    vision_encoder = keras.Model(inputs=image_input, outputs=vision_output)

    # Text Decoder: Maps (Seq,) -> (Seq, 2048)
    text_input = keras.Input(shape=(None,), dtype="int32")
    text_output = keras.layers.Lambda(
        lambda x: keras.ops.ones(
            (keras.ops.shape(x)[0], keras.ops.shape(x)[1], 2048)
        )
    )(text_input)
    text_decoder = keras.Model(inputs=text_input, outputs=text_output)

    # Helper for embeddings
    text_decoder.get_input_embeddings = lambda x: keras.layers.Embedding(
        50000, 2048
    )(x)

    # 2. Instantiate Backbone
    backbone = MoondreamBackbone(
        vision_encoder=vision_encoder,
        text_decoder=text_decoder,
        projection_dim=2048
    )

    # 3. Run Forward Pass
    inputs = {
        "images": np.random.rand(2, 378, 378, 3),
        "token_ids": np.random.randint(0, 50000, (2, 10)),
        "padding_mask": np.ones((2, 10))
    }
    outputs = backbone(inputs)
    ```
    """

    def __init__(
        self, vision_encoder, text_decoder, projection_dim=2048, **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.projection_dim = projection_dim

        self.vision_projection = keras.layers.Dense(
            projection_dim, name="vision_projection"
        )

        images = keras.Input(shape=(None, None, 3), name="images")
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        inputs = {
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        image_features = self.vision_encoder(images)
        projected_images = self.vision_projection(image_features)

        text_embeddings = self.text_decoder.get_input_embeddings(token_ids)

        combined_embeddings = ops.concatenate(
            [projected_images, text_embeddings], axis=1
        )

        batch_size = ops.shape(images)[0]
        num_patches = ops.shape(projected_images)[1]

        image_mask = ops.ones((batch_size, num_patches), dtype="int32")
        combined_mask = ops.concatenate([image_mask, padding_mask], axis=1)

        outputs = self.text_decoder(
            inputs=None,
            decoder_inputs_embeds=combined_embeddings,
            padding_mask=combined_mask,
        )

        super(MoondreamBackbone, self).__init__(
            inputs=inputs, outputs=outputs, **kwargs
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder": keras.saving.serialize_keras_object(
                    self.vision_encoder
                ),
                "text_decoder": keras.saving.serialize_keras_object(
                    self.text_decoder
                ),
                "projection_dim": self.projection_dim,
            }
        )
        return config
