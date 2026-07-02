import os

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_text_embedder import Gemma3TextEmbedder
from keras_hub.src.models.gemma3.gemma3_text_embedder_preprocessor import (
    Gemma3TextEmbedderPreprocessor,
)
from keras_hub.src.models.gemma3.gemma3_tokenizer import Gemma3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Gemma3TextEmbedderTest(TestCase):
    def setUp(self):
        self.tokenizer = Gemma3Tokenizer(
            # Generated using create_gemma3_test_proto.py
            proto=os.path.join(
                self.get_test_data_dir(), "gemma3_test_vocab.spm"
            ),
            has_vision_tokens=False,
        )
        self.preprocessor = Gemma3TextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=8,
        )
        # Build a small text-only backbone matching the tokenizer vocabulary.
        self.backbone = Gemma3Backbone(
            vocabulary_size=self.tokenizer.vocabulary_size(),
            image_size=16,  # dummy — text-only model, not used
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.train_data = (
            ["the quick brown fox", "the earth is round"],  # Features.
            np.zeros((2, 8), dtype="float32"),  # Labels matching embedding dim.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_embedder_basics(self):
        self.run_task_test(
            cls=Gemma3TextEmbedder,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8),
            compile_kwargs={"loss": "mean_squared_error"},
        )

    def test_output_is_normalized(self):
        """Test that output embeddings have unit L2 norm by default."""
        embedder = Gemma3TextEmbedder(**self.init_kwargs)
        output = embedder(self.input_data)
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_output_not_normalized(self):
        """Test that normalization can be disabled."""
        embedder = Gemma3TextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            normalize=False,
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 8))

    def test_mean_pooling(self):
        """Test mean pooling mode produces correct shape and unit norm."""
        embedder = Gemma3TextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            pooling_mode="mean",
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 8))
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_last_pooling_default(self):
        """Test that 'last' is the default pooling mode."""
        embedder = Gemma3TextEmbedder(**self.init_kwargs)
        self.assertEqual(embedder.pooling_mode, "last")

    def test_invalid_pooling_mode(self):
        """Test that an invalid pooling mode raises ValueError."""
        with self.assertRaises(ValueError):
            Gemma3TextEmbedder(
                backbone=self.backbone,
                preprocessor=self.preprocessor,
                pooling_mode="cls",
            )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=Gemma3TextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_last_token_pooling_respects_mask(self):
        """Test last-token pooling selects the correct non-padding token."""
        sequence_output = ops.convert_to_tensor(
            [[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]]]
        )
        # Mask: first 2 tokens real, last is padding — last real token is idx 1.
        mask_partial = np.array([[1, 1, 0]], dtype="int32")
        # Mask: all 3 tokens real — last real token is idx 2.
        mask_full = np.array([[1, 1, 1]], dtype="int32")

        pooled_partial = Gemma3TextEmbedder._last_token_pooling(
            sequence_output, mask_partial
        )
        pooled_full = Gemma3TextEmbedder._last_token_pooling(
            sequence_output, mask_full
        )

        self.assertAllClose(pooled_partial, [[3.0, 4.0]])
        self.assertAllClose(pooled_full, [[10.0, 20.0]])

    def test_mean_pooling_respects_mask(self):
        """Test that mean pooling correctly ignores padding tokens."""
        sequence_output = ops.convert_to_tensor(
            [[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]]]
        )
        mask_partial = np.array([[1, 1, 0]], dtype="int32")
        mask_full = np.array([[1, 1, 1]], dtype="int32")

        pooled_partial = Gemma3TextEmbedder._mean_pooling(
            sequence_output, mask_partial
        )
        pooled_full = Gemma3TextEmbedder._mean_pooling(
            sequence_output, mask_full
        )

        self.assertAllClose(pooled_partial, [[2.0, 3.0]])
        self.assertAllClose(pooled_full, [[14.0 / 3, 26.0 / 3]], atol=1e-5)
        self.assertNotAllClose(pooled_partial, pooled_full)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Gemma3TextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_harrier_preset(self):
        self.run_preset_test(
            cls=Gemma3TextEmbedder,
            preset="hf://microsoft/harrier-oss-v1-270m",
            input_data={
                "token_ids": ops.ones((2, 16), dtype="int32"),
                "padding_mask": ops.ones((2, 16), dtype="int32"),
            },
            expected_output_shape=(2, 640),
        )
