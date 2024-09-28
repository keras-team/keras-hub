import pytest

from keras_hub.src.models.albert.albert_backbone import AlbertBackbone
from keras_hub.src.models.albert.albert_text_classifier import (
    AlbertTextClassifier,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = AlbertTextClassifier.from_preset(
            "hf://albert/albert-base-v2", num_classes=2
        )
        prompt = "That movies was terrible."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            "hf://albert/albert-base-v2",
            num_classes=2,
            load_weights=False,
        )
        self.assertIsInstance(model, AlbertTextClassifier)
        model = Backbone.from_preset(
            "hf://albert/albert-base-v2",
            load_weights=False,
        )
        self.assertIsInstance(model, AlbertBackbone)

    # TODO: compare numerics with huggingface model
