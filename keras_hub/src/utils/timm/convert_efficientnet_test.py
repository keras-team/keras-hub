import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.tests.test_case import TestCase


class TimmResNetBackboneTest(TestCase):
    @pytest.mark.large
    def test_convert_efficientnet_backbone(self):
        model = Backbone.from_preset("hf://timm/efficientnet_b0.ra_in1k")
        outputs = model.predict(ops.ones((1, 224, 224, 3)))
        self.assertEqual(outputs.shape, (1, 7, 7, 1280))

    @pytest.mark.large
    def test_convert_efficientnet_classifier(self):
        model = ImageClassifier.from_preset("hf://timm/efficientnet_b0.ra_in1k")
        outputs = model.predict(ops.ones((1, 512, 512, 3)))
        self.assertEqual(outputs.shape, (1, 1000))
