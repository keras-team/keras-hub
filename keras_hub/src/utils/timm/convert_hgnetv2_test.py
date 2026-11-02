import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.tests.test_case import TestCase


class TimmHGNetV2BackboneTest(TestCase):
    @pytest.mark.extra_large
    def test_convert_hgnetv2_backbone(self):
        model = Backbone.from_preset("hf://timm/hgnetv2_b4.ssld_stage2_ft_in1k")
        outputs = model.predict(ops.ones((1, 224, 224, 3)))
        self.assertEqual(outputs["stage4"].shape[0], 1)

    @pytest.mark.extra_large
    def test_convert_hgnetv2_classifier(self):
        model = ImageClassifier.from_preset(
            "hf://timm/hgnetv2_b4.ssld_stage2_ft_in1k"
        )
        outputs = model.predict(ops.ones((1, 224, 224, 3)))
        self.assertEqual(outputs.shape, (1, 1000))
