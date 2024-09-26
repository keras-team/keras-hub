import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.models.bert.bert_text_classifier import BertTextClassifier
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = BertTextClassifier.from_preset(
            "hf://google-bert/bert-base-uncased", num_classes=2
        )
        prompt = "That movies was terrible."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            "hf://google-bert/bert-base-uncased",
            num_classes=2,
            load_weights=False,
        )
        self.assertIsInstance(model, BertTextClassifier)
        model = Backbone.from_preset(
            "hf://google-bert/bert-base-uncased",
            load_weights=False,
        )
        self.assertIsInstance(model, BertBackbone)

    # TODO: compare numerics with huggingface model
