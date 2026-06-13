import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.tests.test_case import TestCase


class TestQwen3Converter(TestCase):
    @pytest.mark.extra_large
    def test_backbone_from_hf_preset(self):
        model = Qwen3Backbone.from_preset(
            "hf://microsoft/harrier-oss-v1-0.6b",
            load_weights=False,
        )
        # harrier: hidden_dim=1024, num_layers=28
        self.assertEqual(model.hidden_dim, 1024)
        self.assertEqual(model.num_layers, 28)

    @pytest.mark.extra_large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://microsoft/harrier-oss-v1-0.6b",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3Backbone)
