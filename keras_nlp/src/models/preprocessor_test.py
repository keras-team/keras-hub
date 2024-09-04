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
from absl.testing import parameterized

from keras_nlp.src.models.albert.albert_text_classifier_preprocessor import (
    AlbertTextClassifierPreprocessor,
)
from keras_nlp.src.models.bert.bert_masked_lm_preprocessor import (
    BertMaskedLMPreprocessor,
)
from keras_nlp.src.models.bert.bert_text_classifier_preprocessor import (
    BertTextClassifierPreprocessor,
)
from keras_nlp.src.models.gpt2.gpt2_preprocessor import GPT2Preprocessor
from keras_nlp.src.models.preprocessor import Preprocessor
from keras_nlp.src.models.roberta.roberta_text_classifier_preprocessor import (
    RobertaTextClassifierPreprocessor,
)
from keras_nlp.src.tests.test_case import TestCase
from keras_nlp.src.utils.preset_utils import PREPROCESSOR_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import TOKENIZER_ASSET_DIR
from keras_nlp.src.utils.preset_utils import check_config_class
from keras_nlp.src.utils.preset_utils import load_json


class TestPreprocessor(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertTextClassifierPreprocessor.presets.keys())
        gpt2_presets = set(GPT2Preprocessor.presets.keys())
        all_presets = set(Preprocessor.presets.keys())
        self.assertContainsSubset(bert_presets, all_presets)
        self.assertContainsSubset(gpt2_presets, all_presets)

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            BertTextClassifierPreprocessor.from_preset("bert_tiny_en_uncased"),
            BertTextClassifierPreprocessor,
        )
        self.assertIsInstance(
            BertMaskedLMPreprocessor.from_preset("bert_tiny_en_uncased"),
            BertMaskedLMPreprocessor,
        )

    @pytest.mark.large
    def test_from_preset_with_sequence_length(self):
        preprocessor = BertTextClassifierPreprocessor.from_preset(
            "bert_tiny_en_uncased", sequence_length=16
        )
        self.assertEqual(preprocessor.sequence_length, 16)

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on a preprocessor directly (it is ambiguous).
            Preprocessor.from_preset("bert_tiny_en_uncased")
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            BertTextClassifierPreprocessor.from_preset("gpt2_base_en")
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            BertTextClassifierPreprocessor.from_preset(
                "hf://spacy/en_core_web_sm"
            )

    # TODO: Add more tests when we added a model that has `preprocessor.json`.

    @parameterized.parameters(
        (
            AlbertTextClassifierPreprocessor,
            "albert_base_en_uncased",
            "sentencepiece",
        ),
        (RobertaTextClassifierPreprocessor, "roberta_base_en", "bytepair"),
        (BertTextClassifierPreprocessor, "bert_tiny_en_uncased", "wordpiece"),
    )
    @pytest.mark.large
    def test_save_to_preset(self, cls, preset_name, tokenizer_type):
        save_dir = self.get_temp_dir()
        preprocessor = cls.from_preset(preset_name)
        preprocessor.save_to_preset(save_dir)

        if tokenizer_type == "bytepair":
            vocab_filename = "vocabulary.json"
            expected_assets = [
                "vocabulary.json",
                "merges.txt",
            ]
        elif tokenizer_type == "sentencepiece":
            vocab_filename = "vocabulary.spm"
            expected_assets = ["vocabulary.spm"]
        else:
            vocab_filename = "vocabulary.txt"
            expected_assets = ["vocabulary.txt"]

        # Check existence of vocab file.
        vocab_path = os.path.join(
            save_dir, os.path.join(TOKENIZER_ASSET_DIR, vocab_filename)
        )
        self.assertTrue(os.path.exists(vocab_path))

        # Check assets.
        self.assertEqual(
            set(preprocessor.tokenizer.file_assets),
            set(expected_assets),
        )

        # Check config class.
        preprocessor_config = load_json(save_dir, PREPROCESSOR_CONFIG_FILE)
        self.assertEqual(cls, check_config_class(preprocessor_config))
