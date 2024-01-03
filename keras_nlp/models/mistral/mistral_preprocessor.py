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

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.preprocessing.start_end_packer import StartEndPacker
from keras_nlp.models.mistral.mistral_tokenizer import MistralTokenizer
from keras_nlp.models.preprocessor import Preprocessor
from keras_nlp.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.MistralPreprocessor")
class MistralPreprocessor(Preprocessor):
    def __init__(
        self,
        tokenizer,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token
        self.sequence_length = sequence_length
        self.packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            sequence_length=sequence_length,
            return_padding_mask=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "add_start_token": self.add_start_token,
                "add_end_token": self.add_end_token,
            }
        )
        return config

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        x = convert_inputs_to_list_of_tensor_segments(x)
        if len(x) != 1:
            raise ValueError(
                "Mistral requires each input feature to contain only "
                f"one segment, but received {len(x)}. If you are using Mistral"
                " for a multi-segment classification task, please refer to "
                "classification models like BERT or RoBERTa."
            )
        sequence_length = sequence_length or self.sequence_length
        token_ids, padding_mask = self.packer(
            self.tokenizer(x[0]),
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        return pack_x_y_sample_weight(x, y, sample_weight)

    @classproperty
    def tokenizer_cls(cls):
        return MistralTokenizer
