"""Tests for MetaCLIP 2 backbone."""

import numpy as np
import pytest

from keras_hub.src.models.metaclip_2.metaclip_2_backbone import (
    MetaCLIP2Backbone,
)
from keras_hub.src.models.metaclip_2.metaclip_2_text_encoder import (
    MetaCLIP2TextEncoder,
)
from keras_hub.src.models.metaclip_2.metaclip_2_vision_encoder import (
    MetaCLIP2VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class MetaCLIP2BackboneTest(TestCase):
    def setUp(self):
        self.vision_encoder = MetaCLIP2VisionEncoder(
            patch_size=16,
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            intermediate_dim=64,
            image_shape=(32, 32, 3),
        )
        self.text_encoder = MetaCLIP2TextEncoder(
            vocabulary_size=100,
            embedding_dim=32,
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            intermediate_dim=64,
            max_sequence_length=16,
        )
        self.init_kwargs = {
            "vision_encoder": self.vision_encoder,
            "text_encoder": self.text_encoder,
            "projection_dim": 32,
        }
        self.input_data = {
            "images": np.ones((2, 32, 32, 3), dtype="float32"),
            "token_ids": np.ones((2, 16), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MetaCLIP2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "vision_logits": (2, 2),
                "text_logits": (2, 2),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MetaCLIP2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_vision_encoder(self):
        output = self.vision_encoder({"images": self.input_data["images"]})
        self.assertEqual(
            output["sequence_output"].shape, (2, 5, 32)
        )  # 5 = (32/16)^2 + 1 cls token
        self.assertEqual(output["pooled_output"].shape, (2, 32))

    def test_text_encoder(self):
        output = self.text_encoder({"token_ids": self.input_data["token_ids"]})
        self.assertEqual(output["sequence_output"].shape, (2, 16, 32))
        self.assertEqual(output["pooled_output"].shape, (2, 32))

    def test_get_vision_embeddings(self):
        model = MetaCLIP2Backbone(**self.init_kwargs)
        embeddings = model.get_vision_embeddings(self.input_data["images"])
        self.assertEqual(embeddings.shape, (2, 32))

    def test_get_text_embeddings(self):
        model = MetaCLIP2Backbone(**self.init_kwargs)
        embeddings = model.get_text_embeddings(self.input_data["token_ids"])
        self.assertEqual(embeddings.shape, (2, 32))
