import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.modern_bert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modern_bert.modern_bert_text_embedder import (
    ModernBertTextEmbedder,
)
from keras_hub.src.models.modern_bert.modern_bert_text_embedder_preprocessor import (  # noqa: E501
    ModernBertTextEmbedderPreprocessor,
)
from keras_hub.src.models.modern_bert.modern_bert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertTextEmbedderTest(TestCase):
    def setUp(self):
        # Byte-level BPE vocabulary, as in the text classifier tests.
        self.merges = [
            "Ġ a",
            "Ġ t",
            "Ġ i",
            "Ġ b",
            "a i",
            "p l",
            "n e",
            "Ġa t",
            "p o",
            "r t",
            "Ġt h",
            "ai r",
            "pl a",
            "po rt",
            "Ġai r",
            "Ġa i",
            "pla ne",
        ]
        vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab = sorted(set(vocab))
        vocab += ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]
        self.vocab = {token: i for i, token in enumerate(vocab)}

        self.tokenizer = ModernBertTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )
        self.preprocessor = ModernBertTextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=10,
        )
        # Layer 0 uses global attention and layer 1 local attention. The
        # local window (radius 2) is shorter than the padding, so padded
        # positions far from any real token have all of their keys masked.
        self.backbone = ModernBertBackbone(
            vocabulary_size=self.tokenizer.vocabulary_size(),
            hidden_dim=16,
            intermediate_dim=32,
            num_layers=2,
            num_heads=2,
            local_attention_window=4,
            global_attn_every_n_layers=2,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (
            [" airplane at airport", " airplane airport"],  # Features.
            np.ones((2, self.backbone.hidden_dim), dtype="float32"),  # Labels.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_embedder_basics(self):
        self.run_task_test(
            cls=ModernBertTextEmbedder,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 16),
            compile_kwargs={"loss": "mean_squared_error"},
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ModernBertTextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=ModernBertTextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in ModernBertTextEmbedder.presets:
            self.run_preset_test(
                cls=ModernBertTextEmbedder,
                preset=preset,
                input_data=self.input_data,
            )

    def test_output_is_normalized(self):
        embedder = ModernBertTextEmbedder(**self.init_kwargs)
        output = embedder(self.input_data)
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_output_not_normalized(self):
        embedder = ModernBertTextEmbedder(**self.init_kwargs, normalize=False)
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 16))
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertNotAllClose(norms, np.ones(norms.shape))

    def test_mean_pooling_matches_backbone(self):
        embedder = ModernBertTextEmbedder(**self.init_kwargs, normalize=False)
        output = embedder(self.input_data)
        hidden = ops.convert_to_numpy(self.backbone(self.input_data))
        mask = ops.convert_to_numpy(self.input_data["padding_mask"])
        mask = mask.astype("float32")[..., None]
        expected = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        self.assertAllClose(output, expected, atol=1e-5, rtol=1e-5)

    def test_cls_pooling_matches_backbone(self):
        embedder = ModernBertTextEmbedder(
            **self.init_kwargs, pooling_mode="cls", normalize=False
        )
        output = embedder(self.input_data)
        hidden = ops.convert_to_numpy(self.backbone(self.input_data))
        self.assertAllClose(output, hidden[:, 0, :], atol=1e-5, rtol=1e-5)

    def test_max_pooling_matches_backbone(self):
        embedder = ModernBertTextEmbedder(
            **self.init_kwargs, pooling_mode="max", normalize=False
        )
        output = embedder(self.input_data)
        hidden = ops.convert_to_numpy(self.backbone(self.input_data))
        mask = ops.convert_to_numpy(self.input_data["padding_mask"])
        mask = mask.astype(bool)[..., None]
        expected = np.where(mask, hidden, -np.inf).max(axis=1)
        self.assertAllClose(output, expected, atol=1e-5, rtol=1e-5)

    def test_invalid_pooling_mode(self):
        with self.assertRaisesRegex(ValueError, "Invalid pooling_mode"):
            ModernBertTextEmbedder(**self.init_kwargs, pooling_mode="invalid")

    def test_mean_pooling_respects_mask(self):
        sequence_output = ops.convert_to_tensor(
            [[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]]]
        )
        mask_partial = np.array([[1, 1, 0]], dtype="int32")
        mask_full = np.array([[1, 1, 1]], dtype="int32")

        pooled_partial = ModernBertTextEmbedder._mean_pooling(
            sequence_output, mask_partial
        )
        pooled_full = ModernBertTextEmbedder._mean_pooling(
            sequence_output, mask_full
        )

        self.assertAllClose(pooled_partial, [[2.0, 3.0]])
        self.assertAllClose(pooled_full, [[14.0 / 3, 26.0 / 3]], atol=1e-5)

    @parameterized.named_parameters(
        ("mean", "mean"),
        ("cls", "cls"),
        ("max", "max"),
    )
    def test_padding_invariance(self, pooling_mode):
        """Embeddings must not depend on the amount of right-padding.

        Padded positions far from any real token have every key masked in
        the local attention layer. Those rows must not leak NaNs or values
        into real tokens, or into the pooled embedding.
        """
        embedder = ModernBertTextEmbedder(
            **self.init_kwargs, pooling_mode=pooling_mode
        )
        long_preprocessor = ModernBertTextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=32,
        )
        texts = self.train_data[0]
        short_output = embedder(self.preprocessor(texts))
        long_output = embedder(long_preprocessor(texts))
        self.assertFalse(np.isnan(ops.convert_to_numpy(long_output)).any())
        self.assertAllClose(short_output, long_output, atol=1e-5, rtol=1e-5)

    def test_encode_and_similarity(self):
        embedder = ModernBertTextEmbedder(**self.init_kwargs)
        query = embedder.encode_text(" airplane at airport")
        documents = embedder.encode_documents(
            [" airplane at airport", " airplane airport", " airport"]
        )
        self.assertEqual(query.shape, (1, 16))
        self.assertEqual(documents.shape, (3, 16))
        sims = ops.convert_to_numpy(embedder.similarity(query, documents))
        self.assertEqual(sims.shape, (1, 3))
        # Normalized embeddings, so an identical text has similarity 1.
        self.assertAllClose(sims[0, 0], 1.0, atol=1e-5)
        self.assertTrue(np.all(sims <= 1.0 + 1e-5))

    def test_preprocessed_inputs_have_no_segment_ids(self):
        self.assertIn("token_ids", self.input_data)
        self.assertIn("padding_mask", self.input_data)
        self.assertNotIn("segment_ids", self.input_data)
