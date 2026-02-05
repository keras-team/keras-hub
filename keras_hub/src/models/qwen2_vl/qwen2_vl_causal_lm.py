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
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone


class Qwen2VLCausalLM(CausalLM):
    """Qwen2-VL Causal LM model.

    This model can be used for image captioning, visual question answering,
    and text generation. It wraps a `Qwen2VLBackbone` and adds a generic
    text generation `generate()` method.

    Args:
        backbone: A `Qwen2VLBackbone` instance.
        preprocessor: A `Qwen2VLCausalLMPreprocessor` or `None`.
            If `None`, this model will not be able to process raw strings/images
            directly and will require pre-tokenized inputs.
        **kwargs: Standard Keras keyword arguments.
    """

    def __init__(self, backbone, preprocessor=None, **kwargs):
        super().__init__(
            backbone=backbone,
            preprocessor=preprocessor,
            **kwargs,
        )

    def call(self, inputs, training=False):
        x = self.backbone(inputs)

        # Qwen2-VL ties the weights of the output projection layer to the
        # input embeddings of the text backbone.
        embedding_weights = self.backbone.text_backbone.token_embedding.embeddings
        logits = keras.ops.matmul(x, keras.ops.transpose(embedding_weights))

        return logits

    def get_config(self):
        config = super().get_config()
        return config
