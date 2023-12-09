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

from keras_nlp.models.deberta_v3.deberta_v3_tokenizer import DebertaV3Tokenizer
from keras_nlp.tests.test_case import TestCase


class DebertaV3TokenizerTest(TestCase):
    def setUp(self):
        # Generated using create_deberta_v3_test_proto.py
        proto = os.path.join(
            self.get_test_data_dir(), "deberta_v3_test_vocab.spm"
        )
        self.tokenizer = DebertaV3Tokenizer(proto=proto)
        self.init_kwargs = {"proto": proto}
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=DebertaV3Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[5, 10, 6, 8], [5, 7, 9, 11]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            DebertaV3Tokenizer(
                # Generated using create_no_special_token_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    def test_mask_token_handling(self):
        tokenizer = DebertaV3Tokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.get_vocabulary()[4], "[MASK]")
        self.assertEqual(tokenizer.id_to_token(4), "[MASK]")
        self.assertEqual(tokenizer.token_to_id("[MASK]"), 4)
        input_data = [[5, 10, 6, 8, self.tokenizer.mask_token_id]]
        output = tokenizer.detokenize(input_data)
        self.assertEqual(output, ["the quick brown fox"])

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=DebertaV3Tokenizer,
            preset="deberta_v3_extra_small_en",
            input_data=["The quick brown fox."],
            expected_output=[[279, 1538, 3258, 16123, 260]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DebertaV3Tokenizer.presets:
            self.run_preset_test(
                cls=DebertaV3Tokenizer,
                preset=preset,
                input_data=self.input_data,
            )
