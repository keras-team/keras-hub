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
from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.preprocessor import Preprocessor


@keras_hub_export("keras_hub.models.StableDiffusion3TextToImagePreprocessor")
class StableDiffusion3TextToImagePreprocessor(Preprocessor):
    """Stable Diffusion 3 text-to-image model preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.StableDiffusion3TextToImage`.

    For use with generation, the layer exposes one methods
    `generate_preprocess()`.

    Args:
        clip_l_preprocessor: A `keras_hub.models.CLIPPreprocessor` instance.
        clip_g_preprocessor: A `keras_hub.models.CLIPPreprocessor` instance.
        t5_preprocessor: A optional `keras_hub.models.T5Preprocessor` instance.
    """

    def __init__(
        self,
        clip_l_preprocessor,
        clip_g_preprocessor,
        t5_preprocessor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clip_l_preprocessor = clip_l_preprocessor
        self.clip_g_preprocessor = clip_g_preprocessor
        self.t5_preprocessor = t5_preprocessor

    def build(self, input_shape):
        self.built = True

    def generate_preprocess(self, x):
        token_ids = {}
        token_ids["clip_l"] = self.clip_l_preprocessor(x)["token_ids"]
        token_ids["clip_g"] = self.clip_g_preprocessor(x)["token_ids"]
        if self.t5_preprocessor is not None:
            token_ids["t5"] = self.t5_preprocessor(x)["token_ids"]
        return token_ids

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "clip_l_preprocessor": layers.serialize(
                    self.clip_l_preprocessor
                ),
                "clip_g_preprocessor": layers.serialize(
                    self.clip_g_preprocessor
                ),
                "t5_preprocessor": layers.serialize(self.t5_preprocessor),
            }
        )
        return config

    @property
    def sequence_length(self):
        """The padded length of model input sequences."""
        return self.clip_l_preprocessor.sequence_length
