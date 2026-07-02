import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.models.qwen3.qwen3_text_embedder import Qwen3TextEmbedder
from keras_hub.src.models.qwen3.qwen3_text_embedder_preprocessor import (
    Qwen3TextEmbedderPreprocessor,
)
from keras_hub.src.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3TextEmbedderTest(TestCase):
    def setUp(self):
        # Build a minimal tokenizer.
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab += ["<|im_end|>", "<|endoftext|>", "!"]
        vocab = sorted(set(vocab))
        vocab = dict([(token, i) for i, token in enumerate(vocab)])
        self.tokenizer = Qwen3Tokenizer(
            vocabulary=vocab,
            merges=self.merges,
        )
        self.preprocessor = Qwen3TextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=8,
        )
        # Build a small backbone matching the tokenizer vocabulary size.
        self.backbone = Qwen3Backbone(
            vocabulary_size=self.tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            hidden_dim=16,
            intermediate_dim=32,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.train_data = (
            ["airplane at airport", "airplane airport at"],  # Features.
            np.zeros(
                (2, 16), dtype="float32"
            ),  # Labels matching embedding dim.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_embedder_basics(self):
        self.run_task_test(
            cls=Qwen3TextEmbedder,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 16),
            compile_kwargs={"loss": "mean_squared_error"},
        )

    def test_output_is_normalized(self):
        """Test that output embeddings have unit L2 norm by default."""
        embedder = Qwen3TextEmbedder(**self.init_kwargs)
        output = embedder(self.input_data)
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_output_not_normalized(self):
        """Test that normalization can be disabled."""
        embedder = Qwen3TextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            normalize=False,
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 16))

    def test_mean_pooling(self):
        """Test mean pooling mode produces correct shape and unit norm."""
        embedder = Qwen3TextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            pooling_mode="mean",
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 16))
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_last_pooling_default(self):
        """Test that 'last' is the default pooling mode."""
        embedder = Qwen3TextEmbedder(**self.init_kwargs)
        self.assertEqual(embedder.pooling_mode, "last")

    def test_invalid_pooling_mode(self):
        """Test that an invalid pooling mode raises ValueError."""
        with self.assertRaises(ValueError):
            Qwen3TextEmbedder(
                backbone=self.backbone,
                preprocessor=self.preprocessor,
                pooling_mode="cls",
            )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=Qwen3TextEmbedder,
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

        pooled_partial = Qwen3TextEmbedder._last_token_pooling(
            sequence_output, mask_partial
        )
        pooled_full = Qwen3TextEmbedder._last_token_pooling(
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

        pooled_partial = Qwen3TextEmbedder._mean_pooling(
            sequence_output, mask_partial
        )
        pooled_full = Qwen3TextEmbedder._mean_pooling(
            sequence_output, mask_full
        )

        self.assertAllClose(pooled_partial, [[2.0, 3.0]])
        self.assertAllClose(pooled_full, [[14.0 / 3, 26.0 / 3]], atol=1e-5)
        self.assertNotAllClose(pooled_partial, pooled_full)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3TextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_harrier_preset(self):
        self.run_preset_test(
            cls=Qwen3TextEmbedder,
            preset="hf://microsoft/harrier-oss-v1-0.6b",
            input_data={
                "token_ids": ops.ones((2, 16), dtype="int32"),
                "padding_mask": ops.ones((2, 16), dtype="int32"),
            },
            expected_output_shape=(2, 1024),
        )
