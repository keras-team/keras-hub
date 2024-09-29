import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.distil_bert.distil_bert_backbone import (
    DistilBertBackbone,
)
from keras_hub.src.models.distil_bert.distil_bert_text_classifier import (
    DistilBertTextClassifier,
)
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = DistilBertTextClassifier.from_preset(
            "hf://distilbert/distilbert-base-uncased", num_classes=2
        )
        prompt = "That movies was terrible."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            "hf://distilbert/distilbert-base-uncased",
            num_classes=2,
            load_weights=False,
        )
        self.assertIsInstance(model, DistilBertTextClassifier)
        model = Backbone.from_preset(
            "hf://distilbert/distilbert-base-uncased",
            load_weights=False,
        )
        self.assertIsInstance(model, DistilBertBackbone)

    # TODO: compare numerics with huggingface model
