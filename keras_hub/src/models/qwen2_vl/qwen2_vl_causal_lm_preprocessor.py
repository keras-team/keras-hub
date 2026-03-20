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
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor


@keras_hub_export("keras_hub.models.Qwen2VLCausalLMPreprocessor")
class Qwen2VLCausalLMPreprocessor(CausalLMPreprocessor):
    """Qwen2-VL Causal LM Preprocessor.

    This class handles the preprocessing of inputs for the Qwen2-VL model.
    It combines text tokenization with image preprocessing for the vision
    encoder.

    Args:
        tokenizer: A `keras_hub.models.Tokenizer` instance.
        image_converter: A callable or layer that converts raw images
            to tensors. If `None`, image inputs will pass through unchanged.
    """

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            **kwargs,
        )
        self.image_converter = image_converter

    def generate_preprocess(self, x, sequence_length=None):
        if isinstance(x, dict):
            text = x.get("text", "")
            images = x.get("images", None)
        else:
            text = x
            images = None

        token_ids = self.tokenizer(text)

        if images is not None and self.image_converter:
            images = self.image_converter(images)

        return {
            "token_ids": token_ids,
            "images": images,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_converter": keras.saving.serialize_keras_object(
                    self.image_converter
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "image_converter" in config:
            config["image_converter"] = keras.saving.deserialize_keras_object(
                config["image_converter"]
            )
        return cls(**config)
