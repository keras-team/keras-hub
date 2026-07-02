import os

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.xlm_roberta.xlm_roberta_backbone import (
    XLMRobertaBackbone,
)
from keras_hub.src.models.xlm_roberta.xlm_roberta_text_embedder import (
    XLMRobertaTextEmbedder,
)
from keras_hub.src.models.xlm_roberta.xlm_roberta_text_embedder_preprocessor import (  # noqa: E501
    XLMRobertaTextEmbedderPreprocessor,
)
from keras_hub.src.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class XLMRobertaTextEmbedderTest(TestCase):
    def setUp(self):
        self.tokenizer = XLMRobertaTokenizer(
            # Generated using create_xlm_roberta_test_proto.py
            proto=os.path.join(
                self.get_test_data_dir(), "xlm_roberta_test_vocab.spm"
            )
        )
        self.preprocessor = XLMRobertaTextEmbedderPreprocessor(
            self.tokenizer,
            sequence_length=8,
        )
        self.backbone = XLMRobertaBackbone(
            vocabulary_size=self.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            max_sequence_length=self.preprocessor.sequence_length,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (
            ["the quick brown fox.", "the slow brown fox."],  # Features.
            np.ones((2, self.backbone.hidden_dim), dtype="float32"),  # Labels.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_embedder_basics(self):
        self.run_task_test(
            cls=XLMRobertaTextEmbedder,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 16),
            compile_kwargs={"loss": "mean_squared_error"},
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=XLMRobertaTextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=XLMRobertaTextEmbedder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=XLMRobertaTextEmbedder,
            preset="multilingual_e5_base",
            input_data=self.input_data,
            expected_output_shape=(2, 768),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in XLMRobertaTextEmbedder.presets:
            self.run_preset_test(
                cls=XLMRobertaTextEmbedder,
                preset=preset,
                input_data=self.input_data,
            )

    def test_output_is_normalized(self):
        embedder = XLMRobertaTextEmbedder(**self.init_kwargs)
        output = embedder(self.input_data)
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_output_not_normalized(self):
        embedder = XLMRobertaTextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            normalize=False,
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 16))

    def test_cls_pooling(self):
        embedder = XLMRobertaTextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            pooling_mode="cls",
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 16))
        norms = ops.sqrt(ops.sum(ops.square(output), axis=-1))
        self.assertAllClose(norms, np.ones(norms.shape), atol=1e-5)

    def test_max_pooling(self):
        embedder = XLMRobertaTextEmbedder(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
            pooling_mode="max",
        )
        output = embedder(self.input_data)
        self.assertEqual(output.shape, (2, 16))

    def test_invalid_pooling_mode(self):
        with self.assertRaises(ValueError):
            XLMRobertaTextEmbedder(
                backbone=self.backbone,
                preprocessor=self.preprocessor,
                pooling_mode="invalid",
            )

    def test_mean_pooling_respects_mask(self):
        sequence_output = ops.convert_to_tensor(
            [[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]]]
        )
        mask_partial = np.array([[1, 1, 0]], dtype="int32")
        mask_full = np.array([[1, 1, 1]], dtype="int32")

        pooled_partial = XLMRobertaTextEmbedder._mean_pooling(
            sequence_output, mask_partial
        )
        pooled_full = XLMRobertaTextEmbedder._mean_pooling(
            sequence_output, mask_full
        )

        self.assertAllClose(pooled_partial, [[2.0, 3.0]])
        self.assertAllClose(pooled_full, [[14.0 / 3, 26.0 / 3]], atol=1e-5)
        self.assertNotAllClose(pooled_partial, pooled_full)

    def test_preprocessed_inputs_have_no_segment_ids(self):
        self.assertIn("token_ids", self.input_data)
        self.assertIn("padding_mask", self.input_data)
        self.assertNotIn("segment_ids", self.input_data)
