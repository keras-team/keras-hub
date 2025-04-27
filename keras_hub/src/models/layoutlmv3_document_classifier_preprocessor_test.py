"""Tests for LayoutLMv3 document classifier preprocessor."""

import numpy as np
import pytest

from keras import backend
from keras.testing import test_utils
from keras_hub.src.models.layoutlmv3.layoutlmv3_document_classifier_preprocessor import LayoutLMv3DocumentClassifierPreprocessor
from keras_hub.src.models.layoutlmv3.layoutlmv3_tokenizer import LayoutLMv3Tokenizer

@pytest.mark.keras_serializable
class TestLayoutLMv3DocumentClassifierPreprocessor(test_utils.TestCase):
    """Test the LayoutLMv3 document classifier preprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.tokenizer = LayoutLMv3Tokenizer(
            vocabulary=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world"],
            sequence_length=128,
        )
        self.preprocessor = LayoutLMv3DocumentClassifierPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=128,
        )

    def test_forward_pass(self):
        """Test the forward pass of the preprocessor."""
        inputs = {
            "text": ["Hello world!", "Another document"],
            "bbox": [
                [[0, 0, 100, 20], [0, 30, 100, 50]],
                [[0, 0, 100, 20], [0, 30, 100, 50]],
            ],
            "image": backend.random.uniform((2, 112, 112, 3), 0, 1, dtype="float32"),
        }
        outputs = self.preprocessor(inputs)
        self.assertIn("input_ids", outputs)
        self.assertIn("bbox", outputs)
        self.assertIn("attention_mask", outputs)
        self.assertIn("image", outputs)

    def test_save_and_load(self):
        """Test saving and loading the preprocessor."""
        model = self.preprocessor
        path = self.get_temp_dir()
        model.save(path)
        loaded_model = LayoutLMv3DocumentClassifierPreprocessor.load(path)
        self.assertEqual(model.sequence_length, loaded_model.sequence_length)

    def test_from_preset(self):
        """Test creating preprocessor from preset."""
        preprocessor = LayoutLMv3DocumentClassifierPreprocessor.from_preset(
            "layoutlmv3_base",
            sequence_length=128,
        )
        self.assertIsInstance(preprocessor, LayoutLMv3DocumentClassifierPreprocessor)
        self.assertEqual(preprocessor.sequence_length, 128)

if __name__ == "__main__":
    pytest.main([__file__]) 