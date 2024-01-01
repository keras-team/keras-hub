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

import copy

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_nlp.models.electra.electra_presets import backbone_presets
from keras_nlp.models.electra.electra_tokenizer import (
    ElectraTokenizer,
)
from keras_nlp.models.preprocessor import Preprocessor
from keras_nlp.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight
from keras_nlp.utils.python_utils import classproperty

@keras_nlp_export("keras_nlp.models.ElectraPreprocessor")
class ElectraPreprocessor(Preprocessor):

    def __init__(
            self,
            tokenizer,
            sequence_length=512,
            truncate="round_robin",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.cls_token_id,
            end_value=self.tokenizer.sep_token_id,
            pad_value=self.tokenizer.pad_token_id,
            truncate=truncate,
            sequence_length=sequence_length,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.packer.sequence_length,
                "truncate": self.packer.truncate,
            }
        )
        return config

    def call(self, x, y=None, sample_weight=None):
        x = convert_inputs_to_list_of_tensor_segments(x)
        x = [self.tokenizer(segment) for segment in x]
        token_ids, segment_ids = self.packer(x)
        x = {
            "token_ids": token_ids,
            "segment_ids": segment_ids,
            "padding_mask": token_ids != self.tokenizer.pad_token_id,
        }
        return pack_x_y_sample_weight(x, y, sample_weight)

    @classproperty
    def tokenizer_cls(cls):
        return ElectraTokenizer

    @classproperty
    def presets(cls):
        return copy.deepcopy({**backbone_presets})