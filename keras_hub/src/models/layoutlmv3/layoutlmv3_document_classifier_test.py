"""Tests for LayoutLMv3 document classifier."""

import numpy as np
import pytest

from keras import backend
from keras.testing import test_utils
from keras_hub.src.models.layoutlmv3.layoutlmv3_document_classifier import LayoutLMv3DocumentClassifier
from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import LayoutLMv3Backbone

@pytest.mark.keras_serializable
class TestLayoutLMv3DocumentClassifier(test_utils.TestCase):
    """Test the LayoutLMv3 document classifier."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.backbone = LayoutLMv3Backbone(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=(112, 112),
        )
        self.classifier = LayoutLMv3DocumentClassifier(
            backbone=self.backbone,
            num_classes=2,
            dropout=0.1,
        )

    def test_forward_pass(self):
        """Test the forward pass of the classifier."""
        batch_size = 2
        seq_length = 128
        inputs = {
            "input_ids": backend.random.uniform(
                (batch_size, seq_length), 0, 30522, dtype="int32"
            ),
            "bbox": backend.random.uniform(
                (batch_size, seq_length, 4), 0, 1000, dtype="int32"
            ),
            "attention_mask": backend.ones((batch_size, seq_length), dtype="int32"),
            "image": backend.random.uniform(
                (batch_size, 112, 112, 3), 0, 1, dtype="float32"
            ),
        }
        outputs = self.classifier(inputs)
        self.assertEqual(outputs.shape, (batch_size, 2))

    def test_save_and_load(self):
        """Test saving and loading the classifier."""
        model = self.classifier
        path = self.get_temp_dir()
        model.save(path)
        loaded_model = LayoutLMv3DocumentClassifier.load(path)
        self.assertEqual(model.num_classes, loaded_model.num_classes)
        self.assertEqual(model.dropout, loaded_model.dropout)

    def test_from_preset(self):
        """Test creating classifier from preset."""
        classifier = LayoutLMv3DocumentClassifier.from_preset(
            "layoutlmv3_base",
            num_classes=2,
            dropout=0.1,
        )
        self.assertIsInstance(classifier, LayoutLMv3DocumentClassifier)
        self.assertEqual(classifier.num_classes, 2)
        self.assertEqual(classifier.dropout, 0.1)

if __name__ == "__main__":
    pytest.main([__file__]) 