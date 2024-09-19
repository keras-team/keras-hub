# Copyright 2024 The kerasCV Authors
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


@keras_hub_export("keras_hub.models.SAMBackbone")
class SAMBackbone(Backbone):
    """A backbone for the Segment Anything Model (SAM).

    Args:
        image_encoder: keras_hub.models.ViTDetBackbone. A feature extractor for
            the input images.
        prompt_encoder: keras_hub.layers.SAMPromptEncoder. A Keras layer to
            compute embeddings for points, box, and mask prompt.
        mask_decoder: keras_hub.layers.SAMMaskDecoder. A Keras layer to
            generate segmentation masks given the embeddings generated by the
            backbone and the prompt encoder.
        dtype: The dtype of the layer weights.
    """

    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        image_shape,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        # === Functional model
        image_input = self.image_encoder.input

        inputs = {
            "images": image_input,
            "points": keras.Input(shape=[None, 2], name="points"),
            "labels": keras.Input(shape=[None], name="labels"),
            "boxes": keras.Input(shape=[None, 2, 2], name="boxes"),
            "masks": keras.Input(shape=[None, None, None, 1], name="masks"),
        }
        image_embeddings = self.image_encoder.output
        prompt_embeddings = self.prompt_encoder(inputs)
        outputs = {
            "image_embeddings": image_embeddings,
        }
        outputs.update(prompt_embeddings)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )
        # === Config ===
        self.image_shape = image_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_encoder": keras.layers.serialize(self.image_encoder),
                "prompt_encoder": keras.layers.serialize(self.prompt_encoder),
                "mask_decoder": keras.layers.serialize(self.mask_decoder),
                "image_shape": self.image_shape,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config.update(
            {
                "image_encoder": keras.layers.deserialize(
                    config["image_encoder"]
                ),
                "prompt_encoder": keras.layers.deserialize(
                    config["prompt_encoder"]
                ),
                "mask_decoder": keras.layers.deserialize(
                    config["mask_decoder"]
                ),
            }
        )

        return super().from_config(config)