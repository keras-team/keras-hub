# Copyright 2023 The KerasNLP Authors
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

from keras_nlp.src.models.t5.t5_backbone import T5Backbone
from keras_nlp.src.models.t5.t5_presets import backbone_presets
from keras_nlp.src.models.t5.t5_tokenizer import T5Tokenizer
from keras_nlp.src.utils.preset_utils import register_presets

register_presets(backbone_presets, (T5Backbone, T5Tokenizer))
