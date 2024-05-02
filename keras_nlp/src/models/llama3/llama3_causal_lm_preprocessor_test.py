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

import pytest

from keras_nlp.src.models.llama3.llama3_causal_lm_preprocessor import (
    Llama3CausalLMPreprocessor,
)
from keras_nlp.src.models.llama3.llama3_tokenizer import Llama3Tokenizer
from keras_nlp.src.tests.test_case import TestCase


class Llama3CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += ["<|begin_of_text|>", "<|end_of_text|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.tokenizer = Llama3Tokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = ["airplane at airport"]

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=Llama3CausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[6, 1, 3, 4, 2, 5, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [[1, 3, 4, 2, 5, 0, 0, 0]],  # Pass through labels.
                [[1, 1, 1, 1, 1, 0, 0, 0]],  # Pass through sample_weights.
            ),
        )

    def test_with_start_end_token(self):
        input_data = ["airplane at airport"] * 4

        preprocessor = Llama3CausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=True,
            add_end_token=True,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[6, 1, 3, 4, 2, 5, 7, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]] * 4)
        self.assertAllEqual(y, [[1, 3, 4, 2, 5, 7, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 1, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "airplane at airport"
        preprocessor = Llama3CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [6, 1, 3, 4, 2, 5, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 1, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [6, 1, 3, 4, 2, 5, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 1, 0, 0],
        }
        preprocessor = Llama3CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "airplane at airport")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Llama3CausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=Llama3CausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
