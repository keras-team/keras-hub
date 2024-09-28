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
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)
from keras_hub.src.models.sam.sam_backbone import SAMBackbone
from keras_hub.src.models.sam.sam_image_converter import SamImageConverter


@keras_hub_export("keras_hub.models.SamImageSegmenterPreprocessor")
class SamImageSegmenterPreprocessor(ImageSegmenterPreprocessor):
    backbone_cls = SAMBackbone
    image_converter_cls = SamImageConverter
