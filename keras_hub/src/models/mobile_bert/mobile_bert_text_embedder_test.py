import numpy as np
import pytest
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
        # Setup model.
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = MobileBertTextEmbedderPreprocessor(
            MobileBertTokenizer(vocabulary=self.vocab),
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
            preset="all_minilm_l6_v2_en",
            input_data=self.input_data,
            expected_output_shape=(2, 384),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MobileBertTextEmbedder.presets:
            self.run_preset_test(
                cls=MobileBertTextEmbedder,
                preset=preset,
                input_data=self.input_data,
            )

    def test_output_is_normalized(self):
        """Test that output embeddings have unit L2 norm."""
        embedder = MobileBertTextEmbedder(**self.init_kwargs)
        output = embedder(self.input_data)
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_output_not_normalized(self):
        """Test that normalization can be disabled."""
        embedder = MobileBertTextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            normalize=False,
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 4))

    def test_cls_pooling(self):
        """Test CLS pooling mode."""
        embedder = MobileBertTextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            pooling_mode="cls",
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 4))
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_max_pooling(self):
        """Test max pooling mode."""
        embedder = MobileBertTextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            pooling_mode="max",
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 4))

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
