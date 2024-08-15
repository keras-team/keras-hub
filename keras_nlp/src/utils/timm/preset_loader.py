# Copyright 2024 The KerasNLP Authors
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
"""Convert timm models to KerasNLP."""

from keras_nlp.src.utils.preset_utils import PresetLoader
from keras_nlp.src.utils.preset_utils import jax_memory_cleanup
from keras_nlp.src.utils.timm import convert_resnet
from keras_nlp.src.utils.transformers.safetensor_utils import SafetensorLoader


class TimmPresetLoader(PresetLoader):
    def __init__(self, preset, config):
        super().__init__(preset, config)
        architecture = self.config["architecture"]
        if "resnet" in architecture:
            self.converter = convert_resnet
        else:
            raise ValueError(
                "KerasNLP has no converter for timm models "
                f"with architecture `'{architecture}'`."
            )

    def check_backbone_class(self):
        return self.converter.backbone_cls

    def load_backbone(self, cls, load_weights, **kwargs):
        keras_config = self.converter.convert_backbone_config(self.config)
        backbone = cls(**{**keras_config, **kwargs})
        if load_weights:
            jax_memory_cleanup(backbone)
            # Use prefix="" to avoid using `get_prefixed_key`.
            with SafetensorLoader(self.preset, prefix="") as loader:
                self.converter.convert_weights(backbone, loader, self.config)
        return backbone

    def load_image_converter(self, cls, **kwargs):
        # TODO.
        return None
