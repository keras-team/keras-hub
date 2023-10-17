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

import io

import pytest
import sentencepiece

from keras_nlp.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)
from keras_nlp.tests.test_case import TestCase


class XLMRobertaTokenizerTest(TestCase):
    def setUp(self):
        vocab_data = ["the quick brown fox", "the earth is round"]
        bytes_io = io.BytesIO()
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=iter(vocab_data),
            model_writer=bytes_io,
            vocab_size=11,
            model_type="WORD",
            unk_id=0,
            bos_id=1,
            eos_id=2,
            user_defined_symbols="[MASK]",
        )
        self.init_kwargs = {"proto": bytes_io.getvalue()}
        self.input_data = ["the quick brown fox.", "the earth is round."]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=XLMRobertaTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[5, 10, 6, 3], [5, 7, 9, 3]],
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=XLMRobertaTokenizer,
            preset="xlm_roberta_base_multi",
            input_data=["The quick brown fox."],
            expected_output=[[581, 63773, 119455, 6, 147797, 5]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in XLMRobertaTokenizer.presets:
            self.run_preset_test(
                cls=XLMRobertaTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
