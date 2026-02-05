# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import keras
from keras import ops

from keras_hub.src.models.backbone import Backbone


class Qwen2VLBackbone(Backbone):
    """Qwen2-VL architecture backbone.

    This class implements the Qwen2-VL architecture, which combines a Vision
    Transformer (ViT) encoder with a Qwen2 text backbone. Images are encoded,
    projected, and then concatenated with text embeddings before passing through
    the Qwen2 transformer layers.

    Args:
        vision_encoder: A Keras model or layer. The vision encoder (ViT).
        projector: A Keras model or layer. The projector (C-Abstractor).
        text_backbone: A Keras model. The existing Qwen2 backbone.
        dtype: string or keras.mixed_precision.DTypePolicy. The dtype to use
            for the model computations and weights.
        **kwargs: Standard Keras keyword arguments.

    Example:
    ```python
    vision_encoder = Qwen2VLVisionEncoder(...)
    projector = Qwen2VLProjector(...)
    text_backbone = Qwen2Backbone(...)

    model = Qwen2VLBackbone(
        vision_encoder=vision_encoder,
        projector=projector,
        text_backbone=text_backbone,
    )
    ```
    """

    def __init__(
        self,
        vision_encoder,
        projector,
        text_backbone,
        dtype=None,
        **kwargs,
    ):
        image_shape = getattr(vision_encoder, "input_shape", (None, None, None, 3))
        # Remove batch dim if present
        if len(image_shape) == 5:
            image_shape = image_shape[1:]

        images = keras.Input(shape=image_shape, name="images")
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(shape=(None,), dtype="bool", name="padding_mask")

        # Vision feature extraction and projection
        image_features = vision_encoder(images)
        image_embeddings = projector(image_features)

        # Text embeddings
        text_embeddings = text_backbone.token_embedding(token_ids)

        # Concatenate image and text embeddings
        image_embeddings = ops.cast(image_embeddings, text_embeddings.dtype)
        x = ops.concatenate([image_embeddings, text_embeddings], axis=1)

        # Update padding mask to include image tokens
        batch_size = ops.shape(token_ids)[0]
        image_seq_len = ops.shape(image_embeddings)[1]
        image_mask = ops.ones((batch_size, image_seq_len), dtype="bool")
        combined_mask = ops.concatenate([image_mask, padding_mask], axis=1)

        # Pass through Qwen2 transformer layers
        for layer in text_backbone.transformer_layers:
            x = layer(x, padding_mask=combined_mask)

        x = text_backbone.layer_norm(x)

        super().__init__(
            inputs={
                "images": images,
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=x,
            dtype=dtype,
            **kwargs,
        )

        self.vision_encoder = vision_encoder
        self.projector = projector
        self.text_backbone = text_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder": keras.saving.serialize_keras_object(
                    self.vision_encoder
                ),
                "projector": keras.saving.serialize_keras_object(self.projector),
                "text_backbone": keras.saving.serialize_keras_object(
                    self.text_backbone
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["vision_encoder"] = keras.saving.deserialize_keras_object(
            config["vision_encoder"]
        )
        config["projector"] = keras.saving.deserialize_keras_object(config["projector"])
        config["text_backbone"] = keras.saving.deserialize_keras_object(
            config["text_backbone"]
        )
        return cls(**config)
