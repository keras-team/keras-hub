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

import os

import numpy as np
import pytest

from keras_nlp.src.models.pali_gemma.pali_gemma_causal_lm_preprocessor import (
    PaliGemmaCausalLMPreprocessor,
)
from keras_nlp.src.models.pali_gemma.pali_gemma_tokenizer import (
    PaliGemmaTokenizer,
)
from keras_nlp.src.tests.test_case import TestCase


class GemmaCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        # TODO make a pali gemma test vocab that inclues normal prompts?
        self.tokenizer = PaliGemmaTokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "gemma_test_vocab.spm"
            ),
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = {
            "prompts": ["the quick"],
            "responses": ["brown fox"],
            "images": [np.zeros([1, 224, 224, 3])],
        }

    def test_preprocessor_basics(self):
        self.run_preprocessing_layer_test(
            cls=PaliGemmaCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 4, 9, 5, 7, 2, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                    "response_mask": [[0, 0, 0, 1, 1, 1, 0, 0]],
                    "images": self.input_data["images"],
                },
                [[4, 9, 5, 7, 2, 0, 0, 0]],  # Labels shifted.
                [[0, 0, 1, 1, 1, 0, 0, 0]],  # Zero out unlabeled examples.
            ),
        )

    def test_no_start_end_token(self):
        input_data = {
            "prompts": ["the quick"] * 4,
            "responses": ["brown fox"] * 4,
            "images": [np.zeros([1, 224, 224, 3])] * 4,
        }
        preprocessor = PaliGemmaCausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[4, 9, 5, 7, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(x["response_mask"], [[0, 0, 1, 1, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(y, [[9, 5, 7, 0, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[0, 1, 1, 0, 0, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = {
            "prompts": "the quick",
            "images": np.zeros([1, 224, 224, 3]),
        }
        preprocessor = PaliGemmaCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 4, 9, 0, 0, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 0, 0, 0, 0, 0])
        self.assertAllEqual(x["response_mask"], [0, 0, 0, 0, 0, 0, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 4, 9, 5, 7, 2, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 1, 0, 0],
        }
        preprocessor = PaliGemmaCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "the quick brown fox")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in PaliGemmaCausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=PaliGemmaCausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
