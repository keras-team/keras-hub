import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.swin_transformer.swin_transformer_backbone import (
    SwinTransformerBackbone,
)
from keras_hub.src.models.swin_transformer.swin_transformer_image_classifier import (  # noqa: E501
    SwinTransformerImageClassifier,
)
from keras_hub.src.tests.test_case import TestCase


class TestSwinTransformerConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_preset(self):
        model = SwinTransformerImageClassifier.from_preset(
            "hf://microsoft/swin-tiny-patch4-window7-224"
        )
        output = model.predict(np.ones((1, 224, 224, 3), dtype="float32"))
        self.assertEqual(output.shape, (1, 1000))

    @pytest.mark.extra_large
    def test_class_detection(self):
        model = ImageClassifier.from_preset(
            "hf://microsoft/swin-tiny-patch4-window7-224",
            load_weights=False,
        )
        self.assertIsInstance(model, SwinTransformerImageClassifier)
        model = Backbone.from_preset(
            "hf://microsoft/swin-tiny-patch4-window7-224",
            load_weights=False,
        )
        self.assertIsInstance(model, SwinTransformerBackbone)
