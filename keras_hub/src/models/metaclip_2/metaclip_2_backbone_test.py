"""Tests for MetaCLIP 2 backbone."""

import pytest
from keras import ops

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
        vision_encoder = MetaCLIP2VisionEncoder(
            16, 64, 2, 2, 128, name="vision_encoder"
        )
        text_encoder = MetaCLIP2TextEncoder(
            64, 64, 64, 2, 2, 128, name="text_encoder"
        )
        self.init_kwargs = {
            "vision_encoder": vision_encoder,
            "text_encoder": text_encoder,
            "projection_dim": 64,
        }
        self.input_data = {
            "images": ops.ones((2, 224, 224, 3)),
            "token_ids": ops.ones((2, 77), dtype="int32"),
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

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=MetaCLIP2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-4, "mean": 1e-5}},
        )
