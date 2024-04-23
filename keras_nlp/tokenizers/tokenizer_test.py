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

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use `keras_nlp`, please install Tensorflow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )
from absl.testing import parameterized

from keras_nlp.models.albert.albert_tokenizer import AlbertTokenizer
from keras_nlp.models.bert.bert_tokenizer import BertTokenizer
from keras_nlp.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_nlp.models.roberta.roberta_tokenizer import RobertaTokenizer
from keras_nlp.tests.test_case import TestCase
from keras_nlp.tokenizers.tokenizer import Tokenizer
from keras_nlp.utils.preset_utils import METADATA_FILE
from keras_nlp.utils.preset_utils import TOKENIZER_ASSET_DIR
from keras_nlp.utils.preset_utils import TOKENIZER_CONFIG_FILE
from keras_nlp.utils.preset_utils import check_config_class


class SimpleTokenizer(Tokenizer):
    __test__ = False  # for pytest

    def tokenize(self, inputs):
        return tf.strings.split(inputs).to_tensor()

    def detokenize(self, inputs):
        return tf.strings.reduce_join([inputs], separator=" ", axis=-1)


class TokenizerTest(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertTokenizer.presets.keys())
        gpt2_presets = set(GPT2Tokenizer.presets.keys())
        all_presets = set(Tokenizer.presets.keys())
        self.assertContainsSubset(bert_presets, all_presets)
        self.assertContainsSubset(gpt2_presets, all_presets)

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            Tokenizer.from_preset("bert_tiny_en_uncased"),
            BertTokenizer,
        )
        self.assertIsInstance(
            Tokenizer.from_preset("gpt2_base_en"),
            GPT2Tokenizer,
        )

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            GPT2Tokenizer.from_preset("bert_tiny_en_uncased")
        with self.assertRaisesRegex(
            FileNotFoundError, f"doesn't have a file named `{METADATA_FILE}`"
        ):
            # No loading on a non-keras model.
            Tokenizer.from_preset("hf://google-bert/bert-base-uncased")

    def test_tokenize(self):
        input_data = ["the quick brown fox"]
        tokenizer = SimpleTokenizer()
        tokenize_output = tokenizer.tokenize(input_data)
        call_output = tokenizer(input_data)
        self.assertAllEqual(tokenize_output, [["the", "quick", "brown", "fox"]])
        self.assertAllEqual(call_output, [["the", "quick", "brown", "fox"]])

    def test_detokenize(self):
        input_data = ["the", "quick", "brown", "fox"]
        tokenizer = SimpleTokenizer()
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, ["the quick brown fox"])

    def test_missing_tokenize_raises(self):
        with self.assertRaises(NotImplementedError):
            Tokenizer()(["the quick brown fox"])

    @parameterized.parameters(
        (AlbertTokenizer, "albert_base_en_uncased", "sentencepiece"),
        (RobertaTokenizer, "roberta_base_en", "bytepair"),
        (BertTokenizer, "bert_tiny_en_uncased", "wordpiece"),
    )
    @pytest.mark.keras_3_only
    @pytest.mark.large
    def test_save_to_preset(self, cls, preset_name, tokenizer_type):
        save_dir = self.get_temp_dir()
        tokenizer = cls.from_preset(preset_name)
        tokenizer.save_to_preset(save_dir)

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
        self.assertEqual(set(tokenizer.file_assets), set(expected_assets))

        # Check config class.
        self.assertEqual(
            cls, check_config_class(save_dir, TOKENIZER_CONFIG_FILE)
        )
