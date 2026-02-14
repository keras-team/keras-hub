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
from keras_hub.src.models.causal_lm import CausalLM


@keras_hub_export("keras_hub.models.Qwen2VLCausalLM")
class Qwen2VLCausalLM(CausalLM):
    """Qwen2-VL Causal LM model."""

    def __init__(self, backbone, preprocessor=None, **kwargs):
        super().__init__(backbone=backbone, preprocessor=preprocessor, **kwargs)
        self.backbone = backbone

    def call(self, inputs, training=False, mask=None):
        images = inputs["images"]
        token_ids = inputs["token_ids"]

        vision_encoder = self.backbone.vision_encoder
        text_backbone = self.backbone.text_backbone

        image_embeds = vision_encoder(images, training=training)
        text_embeds = text_backbone.token_embedding(token_ids)

        x = keras.ops.concatenate([image_embeds, text_embeds], axis=1)

        for layer in text_backbone.transformer_layers:
            x = layer(x, training=training)

        if hasattr(text_backbone, "layer_norm"):
            x = text_backbone.layer_norm(x)

        x = self.backbone.text_backbone.token_embedding(x, reverse=True)
        return x

    def get_config(self):
        return super().get_config()
