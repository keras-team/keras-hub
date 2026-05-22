import numpy as np
import pytest
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
        # Setup model.
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = BertTextEmbedderPreprocessor(
            BertTokenizer(vocabulary=self.vocab),
            sequence_length=5,
        )
        self.backbone = BertBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=self.preprocessor.sequence_length,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (
            ["the quick brown fox.", "the slow brown fox."],  # Features.
            [1, 0],  # Labels.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_embedder_basics(self):
        self.run_task_test(
            cls=BertTextEmbedder,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BertTextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=BertTextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=BertTextEmbedder,
            preset="all_minilm_l6_v2_en",
            input_data=self.input_data,
            expected_output_shape=(2, 384),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BertTextEmbedder.presets:
            self.run_preset_test(
                cls=BertTextEmbedder,
                preset=preset,
                input_data=self.input_data,
            )

    def test_output_is_normalized(self):
        """Test that output embeddings have unit L2 norm."""
        embedder = BertTextEmbedder(**self.init_kwargs)
        output = embedder(self.input_data)
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_output_not_normalized(self):
        """Test that normalization can be disabled."""
        embedder = BertTextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            normalize=False,
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 2))

    def test_cls_pooling(self):
        """Test CLS pooling mode."""
        embedder = BertTextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            pooling_mode="cls",
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 2))
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_max_pooling(self):
        """Test max pooling mode."""
        embedder = BertTextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            pooling_mode="max",
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 2))

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
        input_1 = {
            "token_ids": np.array([[1, 5, 6, 2, 0]], dtype="int32"),
            "segment_ids": np.array([[0, 0, 0, 0, 0]], dtype="int32"),
            "padding_mask": np.array([[1, 1, 1, 1, 0]], dtype="int32"),
        }
        input_2 = {
            "token_ids": np.array([[1, 5, 6, 2, 0]], dtype="int32"),
            "segment_ids": np.array([[0, 0, 0, 0, 0]], dtype="int32"),
            "padding_mask": np.array([[1, 1, 1, 1, 1]], dtype="int32"),
        }
        output_1 = embedder(input_1)
        output_2 = embedder(input_2)
        self.assertEqual(output_1.shape, (1, 2))
        self.assertEqual(output_2.shape, (1, 2))
