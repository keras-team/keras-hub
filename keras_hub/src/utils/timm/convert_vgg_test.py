import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.tests.test_case import TestCase


class TimmVGGBackboneTest(TestCase):
    @pytest.mark.extra_large
    def test_convert_vgg_backbone(self):
        model = Backbone.from_preset("hf://timm/vgg11.tv_in1k")
        outputs = model.predict(ops.ones((1, 224, 224, 3)))
        self.assertEqual(outputs.shape, (1, 7, 7, 512))

    @pytest.mark.extra_large
    def test_convert_vgg_classifier(self):
        model = ImageClassifier.from_preset("hf://timm/vgg11.tv_in1k")
        outputs = model.predict(ops.ones((1, 224, 224, 3)))
        self.assertEqual(outputs.shape, (1, 1000))
