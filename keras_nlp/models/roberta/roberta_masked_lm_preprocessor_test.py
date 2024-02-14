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

from keras_nlp.models.roberta.roberta_masked_lm_preprocessor import (
    RobertaMaskedLMPreprocessor,
)
from keras_nlp.models.roberta.roberta_tokenizer import RobertaTokenizer
from keras_nlp.tests.test_case import TestCase


class RobertaMaskedLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["<s>", "<pad>", "</s>", "air", "Ġair", "plane", "Ġat"]
        self.vocab += ["port", "<mask>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.tokenizer = RobertaTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            # Simplify our testing by masking every available token.
            "mask_selection_rate": 1.0,
            "mask_token_rate": 1.0,
            "random_token_rate": 0.0,
            "mask_selection_length": 4,
            "sequence_length": 12,
        }
        self.input_data = [" airplane airport"]

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=RobertaMaskedLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[0, 8, 8, 8, 8, 2, 1, 1, 1, 1, 1, 1]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
                    "mask_positions": [[1, 2, 3, 4]],
                },
                [[4, 5, 4, 7]],
                [[1.0, 1.0, 1.0, 1.0]],
            ),
        )

    def test_no_masking_zero_rate(self):
        no_mask_preprocessor = RobertaMaskedLMPreprocessor(
            self.tokenizer,
            mask_selection_rate=0.0,
            mask_selection_length=4,
            sequence_length=12,
        )
        input_data = [" airplane airport"]
        self.assertAllClose(
            no_mask_preprocessor(input_data),
            (
                {
                    "token_ids": [[0, 4, 5, 4, 7, 2, 1, 1, 1, 1, 1, 1]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
                    "mask_positions": [[0, 0, 0, 0]],
                },
                [[0, 0, 0, 0]],
                [[0.0, 0.0, 0.0, 0.0]],
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in RobertaMaskedLMPreprocessor.presets:
            self.run_preset_test(
                cls=RobertaMaskedLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
