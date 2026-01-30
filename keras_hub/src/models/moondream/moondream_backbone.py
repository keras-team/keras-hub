import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.MoondreamBackbone")
class MoondreamBackbone(Backbone):
    def __init__(self, vision_encoder, text_decoder, projection_dim=2048, **kwargs):
        super().__init__(**kwargs)

        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder

        # The Connector
        self.vision_projection = keras.layers.Dense(
            projection_dim, name="vision_projection"
        )

    def call(self, inputs):
        images = inputs["images"]
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]

        # 1. Image Features
        image_features = self.vision_encoder(images)

        # 2. Project
        projected_images = self.vision_projection(image_features)

        # 3. Text Embeddings
        text_embeddings = self.text_decoder.get_input_embeddings(token_ids)

        # 4. Concatenate
        combined_embeddings = ops.concatenate(
            [projected_images, text_embeddings], axis=1
        )

        # 5. Masking
        batch_size = ops.shape(images)[0]
        num_patches = ops.shape(projected_images)[1]

        image_mask = ops.ones((batch_size, num_patches), dtype="bool")
        combined_mask = ops.concatenate([image_mask, padding_mask], axis=1)

        # 6. Decoder Pass
        # Now compatible with our Subclass Mock Decoder
        outputs = self.text_decoder(
            inputs=None,
            decoder_inputs_embeds=combined_embeddings,
            padding_mask=combined_mask,
        )

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder": keras.saving.serialize_keras_object(
                    self.vision_encoder
                ),
                "text_decoder": keras.saving.serialize_keras_object(self.text_decoder),
                "projection_dim": self.vision_projection.units,
            }
        )
        return config
