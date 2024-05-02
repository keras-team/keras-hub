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

from keras_nlp.src.models.phi3.phi3_tokenizer import Phi3Tokenizer
from keras_nlp.src.tests.test_case import TestCase


class Phi3TokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_phi3_test_proto.py
            "proto": os.path.join(
                self.get_test_data_dir(), "phi3_test_vocab.spm"
            )
        }
        # `<|endoftext|>` id = vocab_size = 15
        self.input_data = [
            "the fox <|endoftext|>",
            "the earth <|endoftext|>",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Phi3Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[
                [3, 5, 6, 4, 3, 9, 7, 11, 3, 15],
                [3, 5, 6, 4, 3, 4, 8, 14, 5, 6, 3, 15],
            ],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            Phi3Tokenizer(
                # Generated using create_no_special_token_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )
        # Llama proto doesn't have `<|endoftext|>`
        with self.assertRaises(ValueError):
            Phi3Tokenizer(
                # Generated using create_no_special_token_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "llama_test_vocab.spm"
                )
            )

    # @pytest.mark.large
    # def test_smallest_preset(self):
    #     self.run_preset_test(
    #         cls=Phi3Tokenizer,
    #         preset="",
    #         input_data=["The quick brown fox."],
    #         expected_output=[[450, 4996, 17354, 1701, 29916, 29889]],
    #     )

    # @pytest.mark.extra_large
    # def test_all_presets(self):
    #     for preset in Phi3Tokenizer.presets:
    #         self.run_preset_test(
    #             cls=Phi3Tokenizer,
    #             preset=preset,
    #             input_data=self.input_data,
    #         )
