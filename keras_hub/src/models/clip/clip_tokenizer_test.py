# Copyright 2024 The KerasHub Authors
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

from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.tests.test_case import TestCase


class CLIPTokenizerTest(TestCase):
    def setUp(self):
        vocab = ["air", "plane</w>", "port</w>"]
        vocab += ["<|endoftext|>", "<|startoftext|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(vocab)])
        merges = ["a i", "p l", "n e</w>", "p o", "r t</w>", "ai r", "pl a"]
        merges += ["po rt</w>", "pla ne</w>"]
        self.merges = merges
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = ["airplane ", " airport"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=CLIPTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            # Whitespaces should be removed.
            expected_output=[[0, 1], [0, 2]],
            expected_detokenize_output=["airplane", "airport"],
        )

    def test_pad_with_end_token(self):
        init_kwargs = self.init_kwargs.copy()
        init_kwargs["pad_with_end_token"] = True
        tokenizer = CLIPTokenizer(**init_kwargs)
        self.assertEqual(tokenizer.pad_token_id, tokenizer.end_token_id)

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            CLIPTokenizer(vocabulary={"foo": 0, "bar": 1}, merges=["fo o"])

    @pytest.mark.large
    def test_smallest_preset(self):
        self.skipTest(
            "TODO: Add preset from `hf://openai/clip-vit-large-patch14`"
        )
        self.run_preset_test(
            cls=CLIPTokenizer,
            preset="llama3_8b_en",
            input_data=["The quick brown fox."],
            expected_output=[[791, 4062, 14198, 39935, 13]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        self.skipTest(
            "TODO: Add preset from `hf://openai/clip-vit-large-patch14`"
        )
        for preset in CLIPTokenizer.presets:
            self.run_preset_test(
                cls=CLIPTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
