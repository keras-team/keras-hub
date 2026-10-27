import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.deit.deit_backbone import DeiTBackbone
from keras_hub.src.models.deit.deit_image_classifier import DeiTImageClassifier
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.tests.test_case import TestCase


class TestDeiTConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_preset(self):
        model = DeiTImageClassifier.from_preset(
            "hf://facebook/deit-base-distilled-patch16-384"
        )
        output = model.predict(np.ones((1, 384, 384, 3), dtype="float32"))
        self.assertEqual(output.shape, (1, 1000))

    @pytest.mark.extra_large
    def test_class_detection(self):
        model = ImageClassifier.from_preset(
            "hf://facebook/deit-base-distilled-patch16-384",
            load_weights=False,
        )
        self.assertIsInstance(model, DeiTImageClassifier)
        model = Backbone.from_preset(
            "hf://facebook/deit-base-distilled-patch16-384",
            load_weights=False,
        )
        self.assertIsInstance(model, DeiTBackbone)
