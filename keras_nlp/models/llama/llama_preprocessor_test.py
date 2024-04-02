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

from keras_nlp.models.llama.llama_preprocessor import LlamaPreprocessor
from keras_nlp.models.llama.llama_tokenizer import LlamaTokenizer
from keras_nlp.tests.test_case import TestCase


class LlamaPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = LlamaTokenizer(
            # Generated using create_llama_test_proto.py
            proto=os.path.join(self.get_test_data_dir(), "llama_test_vocab.spm")
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (
            ["the quick brown fox"],
            [1],  # Pass through labels.
            [1.0],  # Pass through sample_weights.
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=LlamaPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 3, 8, 4, 6, 0, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 0, 0, 0]],
                },
                [1],  # Pass through labels.
                [1.0],  # Pass through sample_weights.
            ),
        )

    def test_errors_for_2d_list_input(self):
        preprocessor = LlamaPreprocessor(**self.init_kwargs)
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            preprocessor(ambiguous_input)
