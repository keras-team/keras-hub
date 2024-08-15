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

from keras_nlp.src.utils.timm.convert_resnet import load_resnet_backbone


def load_timm_backbone(cls, preset, load_weights, **kwargs):
    """Load a timm model config and weights as a KerasNLP backbone.

    Args:
        cls (class): Keras model class.
        preset (str): Preset configuration name.
        load_weights (bool): Whether to load the weights.

    Returns:
        backbone: Initialized Keras model backbone.
    """
    if cls is None:
        raise ValueError("Backbone class is None")
    if cls.__name__ == "ResNetBackbone":
        return load_resnet_backbone(cls, preset, load_weights, **kwargs)
    raise ValueError(
        f"{cls} has not been ported from the Hugging Face format yet. "
        "Please check Hugging Face Hub for the Keras model. "
    )
