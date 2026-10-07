import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_backbone import (
    Qwen3_5MoeBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class TestQwen3_5MoeConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = Qwen3_5MoeBackbone.from_preset(
            "hf://yujiepan/qwen3.5-moe-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3_5MoeBackbone)

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://yujiepan/qwen3.5-moe-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3_5MoeBackbone)
