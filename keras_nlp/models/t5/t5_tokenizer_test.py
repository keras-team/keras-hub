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

import pytest

from keras_nlp.models.t5.t5_tokenizer import T5Tokenizer
from keras_nlp.tests.test_case import TestCase


class T5TokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_t5_test_proto.py
            "proto": os.path.join(self.get_test_data_dir(), "t5_test_vocab.spm")
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=T5Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[4, 9, 5, 7], [4, 6, 8, 10]],
        )

    def test_tokenizer_special_tokens(self):
        input_data = ["</s> the quick brown fox </s> <pad>"]
        tokenizer = T5Tokenizer(**self.init_kwargs)
        start_token_id = tokenizer.start_token_id
        end_token_id = tokenizer.end_token_id
        pad_token_id = tokenizer.pad_token_id
        expected_output = [
            [
                start_token_id,
                4,
                9,
                5,
                7,
                end_token_id,
                pad_token_id,
            ]
        ]
        self.assertAllEqual(tokenizer(input_data), expected_output)

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            T5Tokenizer(
                # Generated using create_no_special_token_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    @pytest.mark.large
    def test_smallest_preset(self):
        for preset in T5Tokenizer.presets:
            self.run_preset_test(
                cls=T5Tokenizer,
                preset=preset,
                input_data=["The quick brown fox."],
                expected_output=[[37, 1704, 4216, 3, 20400, 5]],
            )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in T5Tokenizer.presets:
            self.run_preset_test(
                cls=T5Tokenizer,
                preset=preset,
                input_data=self.input_data,
            )
