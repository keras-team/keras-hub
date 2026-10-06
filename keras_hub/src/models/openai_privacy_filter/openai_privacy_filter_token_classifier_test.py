from keras import ops

from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_token_classifier import (  # noqa: E501
    OpenAIPrivacyFilterTokenClassifier,
)
from keras_hub.src.tests.test_case import TestCase


class OpenAIPrivacyFilterTokenClassifierTest(TestCase):
    def setUp(self):
        self.backbone = OpenAIPrivacyFilterBackbone(
            vocabulary_size=1000,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=64,
            intermediate_dim=64,
            head_dim=16,
            num_experts=4,
            top_k=2,
            sliding_window=8,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 33,
            "classifier_dropout": 0.0,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 8), dtype="int32"),
            "padding_mask": ops.ones((2, 8), dtype="int32"),
        }

    def test_classifier_basics(self):
        classifier = OpenAIPrivacyFilterTokenClassifier(
            **self.init_kwargs, compile=False
        )
        output = classifier(self.input_data)
        self.assertEqual(output.shape, (2, 8, 33))

    def test_compile_defaults(self):
        """Test that compile() sets default optimizer, loss, metrics."""
        classifier = OpenAIPrivacyFilterTokenClassifier(
            **self.init_kwargs, compile=True
        )
        self.assertIsNotNone(classifier.optimizer)
        self.assertIsNotNone(classifier.loss)

    def test_get_config(self):
        classifier = OpenAIPrivacyFilterTokenClassifier(
            **self.init_kwargs, compile=False
        )
        config = classifier.get_config()
        self.assertEqual(config["num_classes"], 33)
        self.assertEqual(config["classifier_dropout"], 0.0)
