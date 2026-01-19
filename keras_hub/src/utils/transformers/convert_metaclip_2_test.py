import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.metaclip_2.metaclip_2_backbone import (
    MetaCLIP2Backbone,
)
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_huge_preset(self):
        model = MetaCLIP2Backbone.from_preset(
            "hf://facebook/metaclip-2-worldwide-huge-quickgelu",
        )
        # Test with dummy image and token inputs
        images = np.ones((1, 224, 224, 3), dtype="float32")
        token_ids = np.ones((1, 77), dtype="int32")
        outputs = model.predict({"images": images, "token_ids": token_ids})
        # Check output shapes
        self.assertEqual(outputs["vision_logits"].shape, (1, 1))
        self.assertEqual(outputs["text_logits"].shape, (1, 1))

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://facebook/metaclip-2-worldwide-huge-quickgelu",
            load_weights=False,
        )
        self.assertIsInstance(model, MetaCLIP2Backbone)
