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
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.resizing_image_converter import (
    ResizingImageConverter,
)
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone


@keras_hub_export("keras_hub.layers.ResNetImageConverter")
class ResNetImageConverter(ResizingImageConverter):
    backbone_cls = ResNetBackbone
