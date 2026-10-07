import os
import pathlib

import pytest
from absl.testing import parameterized

from keras_hub.src.models.albert.albert_text_classifier_preprocessor import (
    AlbertTextClassifierPreprocessor,
)
from keras_hub.src.models.bert.bert_masked_lm_preprocessor import (
    BertMaskedLMPreprocessor,
)
from keras_hub.src.models.bert.bert_text_classifier_preprocessor import (
    BertTextClassifierPreprocessor,
)
from keras_hub.src.models.gpt2.gpt2_preprocessor import GPT2Preprocessor
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.models.roberta.roberta_text_classifier_preprocessor import (
    RobertaTextClassifierPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)
from keras_hub.src.utils.preset_utils import TOKENIZER_ASSET_DIR


class TestPreprocessor(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertTextClassifierPreprocessor.presets.keys())
        gpt2_presets = set(GPT2Preprocessor.presets.keys())
        all_presets = set(Preprocessor.presets.keys())
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
        (AlbertTextClassifierPreprocessor, "albert_base_en_uncased"),
        (RobertaTextClassifierPreprocessor, "roberta_base_en"),
        (BertTextClassifierPreprocessor, "bert_tiny_en_uncased"),
    )
    @pytest.mark.large
    def test_save_to_preset(self, cls, preset_name):
        save_dir = self.get_temp_dir()
        preprocessor = cls.from_preset(preset_name, sequence_length=100)
        tokenizer = preprocessor.tokenizer
        preprocessor.save_to_preset(save_dir)
        # Save a backbone so the preset is valid.
        backbone = cls.backbone_cls.from_preset(preset_name, load_weights=False)
        backbone.save_to_preset(save_dir)

        if isinstance(tokenizer, BytePairTokenizer):
            vocab_filename = "vocabulary.json"
            expected_assets = ["vocabulary.json", "merges.txt"]
        elif isinstance(tokenizer, SentencePieceTokenizer):
            vocab_filename = "vocabulary.spm"
            expected_assets = ["vocabulary.spm"]
        else:
            vocab_filename = "vocabulary.txt"
            expected_assets = ["vocabulary.txt"]

        # Check existence of vocab file.
        path = pathlib.Path(save_dir)
        vocab_path = path / TOKENIZER_ASSET_DIR / vocab_filename
        self.assertTrue(os.path.exists(vocab_path))

        # Check assets.
        self.assertEqual(set(tokenizer.file_assets), set(expected_assets))

        # Check restore.
        restored = cls.from_preset(save_dir)
        self.assertEqual(preprocessor.get_config(), restored.get_config())
