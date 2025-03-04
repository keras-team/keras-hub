import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.roberta.roberta_backbone import RobertaBackbone
from keras_hub.src.models.roberta.roberta_text_classifier import RobertaTextClassifier
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = RobertaTextClassifier.from_preset("hf://FacebookAI/roberta-base", num_classes=2)
        prompt = "That movies was terrible."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            "hf://FacebookAI/roberta-base",
            num_classes=2,
            load_weights=False,
        )
        self.assertIsInstance(model, RobertaTextClassifier)
        model = Backbone.from_preset(
            "hf://FacebookAI/roberta-base",
            load_weights=False,
        )
        self.assertIsInstance(model, RobertaBackbone)

    # TODO: compare numerics with huggingface model
