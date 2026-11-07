import keras
import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)
from keras_hub.src.models.mobile_bert.mobile_bert_text_embedder import (
    MobileBertTextEmbedder,
)
from keras_hub.src.models.mobile_bert.mobile_bert_text_embedder_preprocessor import (  # noqa: E501
    MobileBertTextEmbedderPreprocessor,
)
from keras_hub.src.models.mobile_bert.mobile_bert_tokenizer import (
    MobileBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MobileBertTextEmbedderTest(TestCase):
    def setUp(self):
        keras.utils.set_random_seed(0)
        # Setup model.
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.tokenizer = MobileBertTokenizer(vocabulary=self.vocab)
        self.preprocessor = MobileBertTextEmbedderPreprocessor(
            self.tokenizer,
            sequence_length=5,
        )
        self.backbone = MobileBertBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            embedding_dim=2,
            intra_bottleneck_dim=2,
            intermediate_dim=4,
            max_sequence_length=self.preprocessor.sequence_length,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (
            ["the quick brown fox.", "the slow brown fox."],  # Features.
            [[1, 1, 1, 1], [0, 0, 0, 0]],  # Labels.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_embedder_basics(self):
        self.run_task_test(
            cls=MobileBertTextEmbedder,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 4),
            compile_kwargs={"loss": "mean_squared_error"},
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileBertTextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=MobileBertTextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MobileBertTextEmbedder,
            preset="mobilebert_uncased",
            input_data=self.input_data,
            expected_output_shape=(2, 512),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MobileBertTextEmbedder.presets:
            self.run_preset_test(
                cls=MobileBertTextEmbedder,
                preset=preset,
                input_data=self.input_data,
            )

    def _mean_pooled_backbone_output(self):
        hidden = ops.convert_to_numpy(
            self.backbone(self.input_data)["sequence_output"]
        )
        mask = ops.convert_to_numpy(self.input_data["padding_mask"])
        mask = mask.astype("float32")[..., None]
        return (hidden * mask).sum(axis=1) / mask.sum(axis=1)

    def test_output_is_normalized(self):
        """Test that output embeddings have unit L2 norm."""
        embedder = MobileBertTextEmbedder(**self.init_kwargs)
        output = embedder(self.input_data)
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_output_not_normalized(self):
        """`normalize=False` returns the raw mean-pooled backbone output."""
        embedder = MobileBertTextEmbedder(**self.init_kwargs, normalize=False)
        output = ops.convert_to_numpy(embedder(self.input_data))
        self.assertAllClose(
            output, self._mean_pooled_backbone_output(), atol=1e-5
        )
        # Not unit length, so no normalization was applied.
        norms = np.linalg.norm(output, axis=-1)
        self.assertTrue(np.all(np.abs(norms - 1.0) > 1e-3))
        # `normalize=True` returns exactly the L2-normalized raw vector.
        normalized = MobileBertTextEmbedder(**self.init_kwargs)(self.input_data)
        self.assertAllClose(normalized, output / norms[:, None], atol=1e-5)

    def test_cls_pooling(self):
        """CLS pooling returns the first token's hidden state."""
        embedder = MobileBertTextEmbedder(
            **self.init_kwargs, pooling_mode="cls", normalize=False
        )
        output = embedder(self.input_data)
        hidden = self.backbone(self.input_data)["sequence_output"]
        self.assertAllClose(output, hidden[:, 0, :], atol=1e-5)

    def test_max_pooling(self):
        """Max pooling matches a masked numpy max over the backbone output."""
        embedder = MobileBertTextEmbedder(
            **self.init_kwargs, pooling_mode="max", normalize=False
        )
        output = embedder(self.input_data)
        hidden = ops.convert_to_numpy(
            self.backbone(self.input_data)["sequence_output"]
        )
        mask = ops.convert_to_numpy(self.input_data["padding_mask"])
        mask = mask.astype(bool)[..., None]
        expected = np.where(mask, hidden, -np.inf).max(axis=1)
        self.assertAllClose(output, expected, atol=1e-5)

    def test_invalid_pooling_mode(self):
        """Test that invalid pooling mode raises ValueError."""
        with self.assertRaises(ValueError):
            MobileBertTextEmbedder(
                backbone=self.backbone,
                preprocessor=self.preprocessor,
                pooling_mode="invalid",
            )

    def test_mean_pooling_respects_mask(self):
        """Test that mean pooling correctly ignores padding tokens."""
        sequence_output = ops.convert_to_tensor(
            [[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]]]
        )
        mask_partial = np.array([[1, 1, 0]], dtype="int32")
        mask_full = np.array([[1, 1, 1]], dtype="int32")

        pooled_partial = MobileBertTextEmbedder._mean_pooling(
            sequence_output, mask_partial
        )
        pooled_full = MobileBertTextEmbedder._mean_pooling(
            sequence_output, mask_full
        )

        self.assertAllClose(pooled_partial, [[2.0, 3.0]])
        self.assertAllClose(pooled_full, [[14.0 / 3, 26.0 / 3]], atol=1e-5)
        self.assertNotAllClose(pooled_partial, pooled_full)

    def test_mean_pooling_all_padded_float16(self):
        result = MobileBertTextEmbedder._mean_pooling(
            ops.cast(ops.ones((1, 3, 4)), "float16"), np.zeros((1, 3), "int32")
        )
        self.assertAllClose(result, np.zeros((1, 4)))

    def test_mean_pooling_accumulates_in_float32(self):
        # 512 tokens of value 200 sum to 102400, above the float16 maximum
        # (65504), so a float16 accumulator overflows to `inf`.
        pooled = MobileBertTextEmbedder._mean_pooling(
            ops.full((1, 512, 2), 200.0, dtype="float16"),
            np.ones((1, 512), dtype="int32"),
        )
        self.assertEqual(
            keras.backend.standardize_dtype(pooled.dtype), "float16"
        )
        self.assertAllClose(pooled, [[200.0, 200.0]])

    def test_max_pooling_respects_mask(self):
        sequence_output = ops.convert_to_tensor(
            [[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]]
        )
        self.assertAllClose(
            MobileBertTextEmbedder._max_pooling(
                sequence_output, np.array([[1, 1, 0]], dtype="int32")
            ),
            [[3.0, 4.0]],
        )
        self.assertAllClose(
            MobileBertTextEmbedder._max_pooling(
                sequence_output, np.array([[1, 1, 1]], dtype="int32")
            ),
            [[100.0, 200.0]],
        )

    def test_max_pooling_all_padded(self):
        mask = np.zeros((1, 3), dtype="int32")
        for dtype in ("float32", "float16"):
            pooled = MobileBertTextEmbedder._max_pooling(
                ops.full((1, 3, 4), 5.0, dtype=dtype), mask
            )
            self.assertEqual(
                keras.backend.standardize_dtype(pooled.dtype), dtype
            )
            self.assertAllClose(pooled, np.zeros((1, 4)))
        embedder = MobileBertTextEmbedder(
            **self.init_kwargs, pooling_mode="max"
        )
        output = ops.convert_to_numpy(
            embedder(
                {
                    "token_ids": np.zeros((1, 5), dtype="int32"),
                    "segment_ids": np.zeros((1, 5), dtype="int32"),
                    "padding_mask": np.zeros((1, 5), dtype="int32"),
                }
            )
        )
        self.assertTrue(np.isfinite(output).all())
        self.assertAllClose(output, np.zeros((1, 4)))

    def test_l2_normalize_accumulates_in_float32(self):
        # The sum of squares, 16 * 300**2 = 1.44e6, overflows float16, which
        # would turn the embedding into all zeros.
        normalized = MobileBertTextEmbedder._l2_normalize(
            ops.full((1, 16), 300.0, dtype="float16")
        )
        self.assertEqual(
            keras.backend.standardize_dtype(normalized.dtype), "float16"
        )
        normalized = ops.convert_to_numpy(normalized).astype("float32")
        self.assertAllClose(normalized, np.full((1, 16), 0.25), atol=1e-3)
        self.assertAllClose(
            np.linalg.norm(normalized, axis=-1), [1.0], atol=1e-3
        )

    def test_mixed_precision_output_dtype(self):
        """The symbolic output dtype must match the eager output dtype."""
        backbone = MobileBertBackbone.from_config(
            {**self.backbone.get_config(), "dtype": "mixed_float16"}
        )
        for pooling_mode in ("mean", "cls", "max"):
            for normalize in (True, False):
                with self.subTest(
                    pooling_mode=pooling_mode, normalize=normalize
                ):
                    embedder = MobileBertTextEmbedder(
                        backbone=backbone,
                        preprocessor=self.preprocessor,
                        pooling_mode=pooling_mode,
                        normalize=normalize,
                    )
                    output = embedder(self.input_data)
                    self.assertEqual(embedder.output.dtype, "float16")
                    self.assertEqual(
                        keras.backend.standardize_dtype(output.dtype),
                        "float16",
                    )

    @parameterized.named_parameters(
        ("mean", "mean"),
        ("cls", "cls"),
        ("max", "max"),
    )
    def test_padding_invariance(self, pooling_mode):
        """Embeddings must not depend on the amount of right-padding."""
        keras.utils.set_random_seed(0)
        backbone = MobileBertBackbone.from_config(
            {**self.backbone.get_config(), "max_sequence_length": 16}
        )
        short_preprocessor = MobileBertTextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=8,
        )
        long_preprocessor = MobileBertTextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=16,
        )
        embedder = MobileBertTextEmbedder(
            backbone=backbone,
            preprocessor=None,
            pooling_mode=pooling_mode,
        )
        texts = ["the quick brown fox.", "the fox."]
        short_output = embedder(short_preprocessor(texts))
        long_output = embedder(long_preprocessor(texts))
        self.assertFalse(np.isnan(ops.convert_to_numpy(long_output)).any())
        self.assertAllClose(short_output, long_output, atol=1e-5, rtol=1e-5)

    def test_encode_and_similarity(self):
        keras.utils.set_random_seed(0)
        embedder = MobileBertTextEmbedder(**self.init_kwargs)
        query = embedder.encode_text("the quick brown fox.")
        batch_queries = embedder.encode_text(["the fox.", "quick"])
        single_doc = embedder.encode_documents("the fox.")
        documents = embedder.encode_documents(
            ["the quick brown fox.", "the fox.", "quick"]
        )
        self.assertEqual(query.shape, (1, 4))
        self.assertEqual(batch_queries.shape, (2, 4))
        self.assertEqual(single_doc.shape, (1, 4))
        self.assertEqual(documents.shape, (3, 4))
        sims = ops.convert_to_numpy(embedder.similarity(query, documents))
        self.assertEqual(sims.shape, (1, 3))
        query = ops.convert_to_numpy(query)
        documents = ops.convert_to_numpy(documents)
        self.assertAllClose(sims, query @ documents.T, atol=1e-5)
        # Normalized embeddings, so an identical text has similarity 1.
        self.assertAllClose(sims[0, 0], 1.0, atol=1e-5)
        self.assertTrue(np.all(sims <= 1.0 + 1e-5))
        self.assertLess(sims[0, 2], 1.0 - 1e-4)
