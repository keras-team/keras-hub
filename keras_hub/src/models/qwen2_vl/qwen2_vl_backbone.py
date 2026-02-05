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
    Transformer (ViT) encoder with a Qwen2 text backbone.
    """

    def __init__(
        self,
        vision_encoder,
        projector,
        text_backbone,
        dtype=None,
        **kwargs,
    ):
        # 1. Inspect vision encoder for input shape
        image_shape = getattr(
            vision_encoder, "input_shape", (None, None, None, 3)
        )
        if len(image_shape) == 5:
            image_shape = image_shape[1:]

        # 2. Define Inputs
        images = keras.Input(shape=image_shape, name="images")
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="bool", name="padding_mask"
        )

        # 3. Vision Branch
        image_features = vision_encoder(images)
        image_embeddings = projector(image_features)

        # 4. Text Branch
        text_embeddings = text_backbone.token_embedding(token_ids)

        # 5. Fusion (Concatenation)
        image_embeddings = ops.cast(image_embeddings, text_embeddings.dtype)
        x = ops.concatenate([image_embeddings, text_embeddings], axis=1)

        # 6. Update Mask (The Fix)
        # Instead of manually building a shape tuple (which causes the None crash),
        # we generate a mask directly from the existing image_embeddings tensor.
        # image_embeddings shape: (Batch, Seq_Len, Hidden)
        # Slice [:, :, 0] gives us shape (Batch, Seq_Len).
        image_mask = ops.ones_like(image_embeddings[:, :, 0], dtype="bool")

        combined_mask = ops.concatenate([image_mask, padding_mask], axis=1)

        # 7. Pass through Transformer Layers
        for layer in text_backbone.transformer_layers:
            x = layer(x, padding_mask=combined_mask)

        # 8. Final Norm
        x = text_backbone.layer_norm(x)

        # 9. Initialize Backbone
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
                "projector": keras.saving.serialize_keras_object(
                    self.projector
                ),
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
        config["projector"] = keras.saving.deserialize_keras_object(
            config["projector"]
        )
        config["text_backbone"] = keras.saving.deserialize_keras_object(
            config["text_backbone"]
        )
        return cls(**config)
