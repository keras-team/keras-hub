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

from keras_hub.src.models.causal_lm import CausalLM


class Qwen2VLCausalLM(CausalLM):
    """Qwen2-VL Causal LM model."""

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # 1. Do NOT pass backbone/preprocessor to super().
        #    The traceback proved that parent classes just forward them to Layer,
        #    causing a crash. We send only the remaining kwargs up.
        super().__init__(**kwargs)

        # 2. Manually attach the components.
        #    Keras will still automatically track the weights of self.backbone.
        self.backbone = backbone
        self.preprocessor = preprocessor

    def call(self, inputs, training=False):
        x = self.backbone(inputs)
        embedding_weights = (
            self.backbone.text_backbone.token_embedding.embeddings
        )
        logits = keras.ops.matmul(x, keras.ops.transpose(embedding_weights))
        return logits

    def get_config(self):
        # Since we didn't pass backbone to super, we must serialize it manually.
        config = super().get_config()
        config.update(
            {
                "backbone": keras.saving.serialize_keras_object(self.backbone),
                "preprocessor": keras.saving.serialize_keras_object(
                    self.preprocessor
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # And deserialize it manually.
        config["backbone"] = keras.saving.deserialize_keras_object(
            config["backbone"]
        )
        config["preprocessor"] = keras.saving.deserialize_keras_object(
            config["preprocessor"]
        )
        return cls(**config)
