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

from keras_nlp.models.phi3.phi3_preprocessor import Phi3Preprocessor
from keras_nlp.models.phi3.phi3_tokenizer import Phi3Tokenizer
from keras_nlp.tests.test_case import TestCase


class Phi3PreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = Phi3Tokenizer(
            # Generated using create_phi3_test_proto.py
            proto=os.path.join(self.get_test_data_dir(), "phi3_test_vocab.spm")
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 12,
        }
        self.input_data = (
            # Encoded to [3, 5, 6, 4, 3, 9, 7, 11, 3, 15]
            ["the fox <|endoftext|>"],
            [1],  # Pass through labels.
            [1.0],  # Pass through sample_weights.
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=Phi3Preprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 3, 5, 6, 4, 3, 9, 7, 11, 3, 15, 2]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]],
                },
                [1],  # Pass through labels.
                [1.0],  # Pass through sample_weights.
            ),
        )

    def test_errors_for_2d_list_input(self):
        preprocessor = Phi3Preprocessor(**self.init_kwargs)
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            preprocessor(ambiguous_input)

    # @pytest.mark.extra_large
    # def test_all_presets(self):
    #     for preset in Phi3Preprocessor.presets:
    #         self.run_preset_test(
    #             cls=Phi3Preprocessor,
    #             preset=preset,
    #             input_data=self.input_data,
    #         )
