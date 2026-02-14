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

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.Qwen2VLBackbone")
class Qwen2VLBackbone(Backbone):
    """Qwen2-VL Backbone model.

    This backbone combines the Vision Encoder and the Text Backbone.
    It follows the KerasHub Functional API pattern.
    """

    def __init__(
        self,
        vision_encoder,
        text_backbone,
        image_converter=None,
        **kwargs,
    ):
        # --- Inputs ---
        # 1. Image Input: 5D (Batch, Time, H, W, Channels)
        # We use flexible shapes (None) to support dynamic resizing
        images = keras.Input(shape=(None, None, None, 3), name="images")

        # 2. Text Input: (Batch, Seq_Len)
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # --- Forward Pass ---
        # 1. Vision Branch
        # The encoder outputs (Batch, Time, H, W, Hidden)
        vision_features = vision_encoder(images)

        # 2. Projection
        # We assume the projector is attached to the vision encoder or separate.
        # Ideally, we define the projector here if it's not part of the encoder.
        # For this implementation, we assume the vision_encoder returns
        # projected features OR we leave the merging logic to the CausalLM.

        # NOTE: In the Functional API style for KerasHub, the Backbone usually
        # just exposes the sub-models.

        # Let's wrap the outputs.
        # Since Qwen2-VL is complex (token replacement), we return the features
        # separately so the CausalLM can merge them.

        outputs = {
            "vision_features": vision_features,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        # --- Initialize Super ---
        super().__init__(
            inputs={
                "images": images,
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=outputs,
            **kwargs,
        )

        self.vision_encoder = vision_encoder
        self.text_backbone = text_backbone
        self.image_converter = image_converter

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder": keras.saving.serialize_keras_object(
                    self.vision_encoder
                ),
                "text_backbone": keras.saving.serialize_keras_object(
                    self.text_backbone
                ),
                "image_converter": keras.saving.serialize_keras_object(
                    self.image_converter
                ),
            }
        )
        return config
