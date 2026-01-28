import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.tests.test_case import TestCase


class TimmMobileNetV5Test(TestCase):
    @pytest.mark.large
    def test_convert_mobilenetv5_backbone(self):
        model = Backbone.from_preset("hf://timm/mobilenetv5_300m.gemma3n")
        outputs = model.predict(ops.ones((1, 224, 224, 3)))
        self.assertEqual(outputs.shape, (1, 16, 16, 2048))

    @pytest.mark.large
    def test_convert_mobilenetv5_classifier(self):
        model = ImageClassifier.from_preset(
            "hf://timm/mobilenetv5_300m.gemma3n"
        )
        outputs = model.predict(ops.ones((1, 224, 224, 3)))
        self.assertEqual(outputs.shape, (1, 2048))
