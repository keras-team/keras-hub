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

import os

import pytest

from keras_nlp.models.mistral.mistral_causal_lm_preprocessor import (
    MistralCausalLMPreprocessor,
)
from keras_nlp.models.mistral.mistral_tokenizer import MistralTokenizer
from keras_nlp.tests.test_case import TestCase


class MistralCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = MistralTokenizer(
            # Generated using create_mistral_test_proto.py
            proto=os.path.join(
                self.get_test_data_dir(), "mistral_test_vocab.spm"
            )
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (["the quick brown fox"],)

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=MistralCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 3, 8, 4, 6, 0, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 0, 0, 0]],
                },
                [[3, 8, 4, 6, 0, 0, 0, 0]],  # Pass through labels.
                [[1, 1, 1, 1, 0, 0, 0, 0]],  # Pass through sample_weights.
            ),
        )

    def test_no_start_end_token(self):
        input_data = ["the quick brown fox"] * 4

        preprocessor = MistralCausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[3, 8, 4, 6, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(y, [[8, 4, 6, 0, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 0, 0, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "the quick brown fox"
        preprocessor = MistralCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 3, 8, 4, 6, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 3, 8, 4, 6, 0, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = MistralCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "the quick brown fox")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MistralCausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=MistralCausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
