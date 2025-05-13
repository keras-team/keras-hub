import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.tests.test_case import TestCase


class TimmCSPNetBackboneTest(TestCase):
    @pytest.mark.large
    def test_convert_cspnet_backbone(self):
        model = Backbone.from_preset("hf://timm/cspdarknet53.ra_in1k")
        outputs = model.predict(ops.ones((1, 256, 256, 3)))
        self.assertEqual(outputs.shape, (1, 8, 8, 1024))

    @pytest.mark.large
    def test_convert_cspnet_classifier(self):
        model = ImageClassifier.from_preset("hf://timm/cspdarknet53.ra_in1k")
        outputs = model.predict(ops.ones((1, 512, 512, 3)))
        self.assertEqual(outputs.shape, (1, 1000))
