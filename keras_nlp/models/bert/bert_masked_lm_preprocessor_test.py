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

import pytest

from keras_nlp.models.bert.bert_masked_lm_preprocessor import (
    BertMaskedLMPreprocessor,
)
from keras_nlp.models.bert.bert_tokenizer import BertTokenizer
from keras_nlp.tests.test_case import TestCase


class BertMaskedLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["THE", "QUICK", "BROWN", "FOX"]
        self.vocab += ["the", "quick", "brown", "fox"]
        self.tokenizer = BertTokenizer(vocabulary=self.vocab)
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            # Simplify our testing by masking every available token.
            "mask_selection_rate": 1.0,
            "mask_token_rate": 1.0,
            "random_token_rate": 0.0,
            "mask_selection_length": 4,
            "sequence_length": 12,
        }
        self.input_data = ["the quick brown fox"]

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=BertMaskedLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[2, 4, 4, 4, 4, 3, 0, 0, 0, 0, 0, 0]],
                    "segment_ids": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
                    "mask_positions": [[1, 2, 3, 4]],
                },
                [[9, 10, 11, 12]],
                [[1.0, 1.0, 1.0, 1.0]],
            ),
        )

    def test_no_masking_zero_rate(self):
        no_mask_preprocessor = BertMaskedLMPreprocessor(
            self.tokenizer,
            mask_selection_rate=0.0,
            mask_selection_length=4,
            sequence_length=12,
        )
        input_data = ["the quick brown fox"]
        self.assertAllClose(
            no_mask_preprocessor(input_data),
            (
                {
                    "token_ids": [[2, 9, 10, 11, 12, 3, 0, 0, 0, 0, 0, 0]],
                    "segment_ids": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
                    "mask_positions": [[0, 0, 0, 0]],
                },
                [[0, 0, 0, 0]],
                [[0.0, 0.0, 0.0, 0.0]],
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BertMaskedLMPreprocessor.presets:
            self.run_preset_test(
                cls=BertMaskedLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
