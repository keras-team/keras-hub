import numpy as np
import pytest

from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_preprocessor import (  # noqa: E501
    OpenAIPrivacyFilterPreprocessor,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_token_classifier import (  # noqa: E501
    OpenAIPrivacyFilterTokenClassifier,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_tokenizer import (  # noqa: E501
    OpenAIPrivacyFilterTokenizer,
)
from keras_hub.src.tests.mocks.mock_bpe_vocabulary import get_bpe_fixture
from keras_hub.src.tests.test_case import TestCase


class OpenAIPrivacyFilterTokenClassifierTest(TestCase):
    def setUp(self):
        self.vocab, self.merges = get_bpe_fixture()
        self.preprocessor = OpenAIPrivacyFilterPreprocessor(
            OpenAIPrivacyFilterTokenizer(
                vocabulary=self.vocab, merges=self.merges
            ),
            sequence_length=8,
        )
        self.backbone = OpenAIPrivacyFilterBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
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
            "num_classes": 3,
            "preprocessor": self.preprocessor,
        }
        # Per-token labels, aligned with the packed `sequence_length`.
        self.train_data = (
            ["airplane at airport", " the airplane"],
            np.array([[0, 1, 2, 0, 1, 0, 0, 0], [1, 0, 2, 0, 0, 0, 0, 0]]),
        )

    def test_classifier_basics(self):
        self.run_task_test(
            cls=OpenAIPrivacyFilterTokenClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, 3),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=OpenAIPrivacyFilterTokenClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.preprocessor(self.train_data[0]),
        )

    def test_missing_num_classes(self):
        with self.assertRaises(TypeError):
            OpenAIPrivacyFilterTokenClassifier(backbone=self.backbone)

    def test_activation(self):
        # `from_logits` is read from the loss config rather than the attribute,
        # which is not public on every Keras version.
        model_none = OpenAIPrivacyFilterTokenClassifier(
            backbone=self.backbone, num_classes=3, activation=None
        )
        model_none.compile(optimizer="adam", loss="auto")
        self.assertTrue(model_none.loss.get_config()["from_logits"])

        model_softmax = OpenAIPrivacyFilterTokenClassifier(
            backbone=self.backbone, num_classes=3, activation="softmax"
        )
        model_softmax.compile(optimizer="adam", loss="auto")
        self.assertFalse(model_softmax.loss.get_config()["from_logits"])

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in OpenAIPrivacyFilterTokenClassifier.presets:
            self.run_preset_test(
                cls=OpenAIPrivacyFilterTokenClassifier,
                preset=preset,
                input_data=["The quick brown fox jumps."],
            )
