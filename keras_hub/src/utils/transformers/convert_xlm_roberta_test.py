import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.models.xlm_roberta.xlm_roberta_backbone import (
    XLMRobertaBackbone,
)
from keras_hub.src.models.xlm_roberta.xlm_roberta_text_classifier import (
    XLMRobertaTextClassifier,
)
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = XLMRobertaTextClassifier.from_preset(
            "hf://FacebookAI/xlm-roberta-base", num_classes=2
        )
        prompt = "That movie was terrible."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            "hf://FacebookAI/xlm-roberta-base",
            num_classes=2,
            load_weights=False,
        )
        self.assertIsInstance(model, XLMRobertaTextClassifier)
        model = Backbone.from_preset(
            "hf://FacebookAI/xlm-roberta-base",
            load_weights=False,
        )
        self.assertIsInstance(model, XLMRobertaBackbone)
