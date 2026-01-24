import os

import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_hub.src.models.albert.albert_tokenizer import AlbertTokenizer
from keras_hub.src.models.bert.bert_tokenizer import BertTokenizer
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer
from keras_hub.src.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_hub.src.models.roberta.roberta_tokenizer import RobertaTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.tokenizer import Tokenizer
from keras_hub.src.utils.preset_utils import TOKENIZER_ASSET_DIR
from keras_hub.src.utils.preset_utils import TOKENIZER_CONFIG_FILE
from keras_hub.src.utils.preset_utils import check_config_class
from keras_hub.src.utils.preset_utils import load_json


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
        self.assertIn("bert_tiny_en_uncased", bert_presets)
        self.assertNotIn("bert_tiny_en_uncased", gpt2_presets)
        self.assertIn("gpt2_base_en", gpt2_presets)
        self.assertNotIn("gpt2_base_en", bert_presets)
        self.assertIn("bert_tiny_en_uncased", all_presets)
        self.assertIn("gpt2_base_en", all_presets)

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
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            Tokenizer.from_preset("hf://spacy/en_core_web_sm")

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
        tokenizer_config = load_json(save_dir, TOKENIZER_CONFIG_FILE)
        self.assertEqual(cls, check_config_class(tokenizer_config))

    def test_export_supported_tokenizer(self):
        proto = os.path.join(self.get_test_data_dir(), "gemma_export_vocab.spm")
        tokenizer = GemmaTokenizer(proto=proto)
        export_path = os.path.join(self.get_temp_dir(), "export_tokenizer")
        tokenizer.export_to_transformers(export_path)
        # Basic check: tokenizer config exists
        self.assertTrue(
            os.path.exists(os.path.join(export_path, "tokenizer_config.json"))
        )

    def test_export_unsupported_tokenizer(self):
        proto = os.path.join(self.get_test_data_dir(), "gemma_export_vocab.spm")

        class UnsupportedTokenizer(GemmaTokenizer):
            pass

        tokenizer = UnsupportedTokenizer(proto=proto)
        export_path = os.path.join(self.get_temp_dir(), "unsupported_tokenizer")
        with self.assertRaises(ValueError):
            tokenizer.export_to_transformers(export_path)
