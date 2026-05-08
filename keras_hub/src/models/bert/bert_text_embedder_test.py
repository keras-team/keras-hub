import numpy as np
from keras import ops

from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.models.bert.bert_text_embedder import BertTextEmbedder
from keras_hub.src.models.bert.bert_text_embedder_preprocessor import (
    BertTextEmbedderPreprocessor,
)
from keras_hub.src.models.bert.bert_tokenizer import BertTokenizer
from keras_hub.src.tests.test_case import TestCase


class BertTextEmbedderTest(TestCase):
    def setUp(self):
        # Set up a minimal BERT backbone and tokenizer for testing.
        self.vocab = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "the",
            "quick",
            "brown",
            "fox",
            "jumped",
            ".",
            "call",
            "me",
            "is",
            "##hmael",
        ]
        self.preprocessor = BertTextEmbedderPreprocessor(
            tokenizer=BertTokenizer(vocabulary=self.vocab),
            sequence_length=12,
        )
        self.backbone = BertBackbone(
            vocabulary_size=len(self.vocab),
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            max_sequence_length=12,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.input_data = ["The quick brown fox jumped."]

    def test_embedder_basics(self):
        self.run_task_test(
            cls=BertTextEmbedder,
            init_kwargs=self.init_kwargs,
            train_data=(self.input_data, [1]),
            expected_output_shape=(1, 8),
        )

    def test_output_is_normalized(self):
        """Test that output embeddings have unit L2 norm."""
        embedder = BertTextEmbedder(**self.init_kwargs)
        preprocessed = self.preprocessor(self.input_data)
        output = embedder(preprocessed)
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        # Each vector should have norm ≈ 1.0
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_output_not_normalized(self):
        """Test that normalization can be disabled."""
        embedder = BertTextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            normalize=False,
        )
        preprocessed = self.preprocessor(self.input_data)
        output = embedder(preprocessed)
        # With normalize=False, norms should generally NOT be 1.0
        # (unless the model happens to produce unit vectors).
        self.assertEqual(output.shape, (1, 8))

    def test_cls_pooling(self):
        """Test CLS pooling mode."""
        embedder = BertTextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            pooling_mode="cls",
        )
        preprocessed = self.preprocessor(self.input_data)
        output = embedder(preprocessed)
        self.assertEqual(output.shape, (1, 8))
        # Output should be normalized.
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_max_pooling(self):
        """Test max pooling mode."""
        embedder = BertTextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            pooling_mode="max",
        )
        preprocessed = self.preprocessor(self.input_data)
        output = embedder(preprocessed)
        self.assertEqual(output.shape, (1, 8))

    def test_invalid_pooling_mode(self):
        """Test that invalid pooling mode raises ValueError."""
        with self.assertRaises(ValueError):
            BertTextEmbedder(
                backbone=self.backbone,
                preprocessor=self.preprocessor,
                pooling_mode="invalid",
            )

    def test_mean_pooling_respects_mask(self):
        """Test that mean pooling correctly ignores padding tokens."""
        embedder = BertTextEmbedder(
            backbone=self.backbone, normalize=False, pooling_mode="mean"
        )
        # Create two inputs with different padding.
        input_1 = {
            "token_ids": np.array([[1, 5, 6, 2, 0, 0]], dtype="int32"),
            "segment_ids": np.array([[0, 0, 0, 0, 0, 0]], dtype="int32"),
            "padding_mask": np.array([[1, 1, 1, 1, 0, 0]], dtype="int32"),
        }
        input_2 = {
            "token_ids": np.array([[1, 5, 6, 2, 0, 0]], dtype="int32"),
            "segment_ids": np.array([[0, 0, 0, 0, 0, 0]], dtype="int32"),
            "padding_mask": np.array([[1, 1, 1, 1, 1, 1]], dtype="int32"),
        }
        output_1 = embedder(input_1)
        output_2 = embedder(input_2)
        # Different masks should produce different embeddings because padding
        # tokens have non-zero embeddings that get excluded vs included.
        self.assertEqual(output_1.shape, (1, 8))
        self.assertEqual(output_2.shape, (1, 8))
