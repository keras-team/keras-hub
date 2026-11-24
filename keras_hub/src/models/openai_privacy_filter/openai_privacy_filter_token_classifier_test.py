import keras
import numpy as np
import pytest

from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_token_classifier import (  # noqa: E501
    OpenAIPrivacyFilterTokenClassifier,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_token_classifier_preprocessor import (  # noqa: E501
    OpenAIPrivacyFilterTokenClassifierPreprocessor,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_tokenizer import (  # noqa: E501
    OpenAIPrivacyFilterTokenizer,
)
from keras_hub.src.tests.mocks.mock_bpe_vocabulary import get_bpe_fixture
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu


class OpenAIPrivacyFilterTokenClassifierTest(TestCase):
    def setUp(self):
        self.vocab, self.merges = get_bpe_fixture()
        self.preprocessor = OpenAIPrivacyFilterTokenClassifierPreprocessor(
            OpenAIPrivacyFilterTokenizer(
                vocabulary=self.vocab, merges=self.merges
            ),
            sequence_length=8,
        )
        self.backbone_kwargs = {
            "vocabulary_size": self.preprocessor.tokenizer.vocabulary_size(),
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 64,
            "intermediate_dim": 64,
            "head_dim": 16,
            "num_experts": 4,
            "top_k": 2,
            "sliding_window": 8,
        }
        self.backbone = OpenAIPrivacyFilterBackbone(**self.backbone_kwargs)
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

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=OpenAIPrivacyFilterTokenClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.preprocessor(self.train_data[0]),
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-4, "mean": 1e-5}},
        )

    def test_missing_num_classes(self):
        with self.assertRaises(TypeError):
            OpenAIPrivacyFilterTokenClassifier(backbone=self.backbone)

    def test_activation(self):
        # `from_logits` is read from the loss config rather than the attribute,
        # which is not public on every Keras version. The default compile at
        # construction must already see `activation`.
        for activation, from_logits in [(None, True), ("softmax", False)]:
            model = OpenAIPrivacyFilterTokenClassifier(
                backbone=self.backbone, num_classes=3, activation=activation
            )
            loss_config = model.loss.get_config()
            self.assertEqual(loss_config["from_logits"], from_logits)
            model.compile(optimizer="adam", loss="auto")
            loss_config = model.loss.get_config()
            self.assertEqual(loss_config["from_logits"], from_logits)

    def test_positional_preprocessor(self):
        model = OpenAIPrivacyFilterTokenClassifier(
            self.backbone, 3, self.preprocessor
        )
        self.assertIs(model.preprocessor, self.preprocessor)
        config = model.get_config()
        self.assertEqual(config["dropout"], 0.0)
        self.assertNotIn("classifier_dropout", config)

    def test_get_config_round_trip_keeps_head_args(self):
        model = OpenAIPrivacyFilterTokenClassifier(
            self.backbone, 3, activation="softmax", dropout=0.1
        )
        # A fresh head matches HF's default `initializer_range` (0.02).
        initializer = model._classifier_dense.kernel_initializer
        self.assertIsInstance(initializer, keras.initializers.TruncatedNormal)
        self.assertEqual(initializer.stddev, 0.02)
        restored = OpenAIPrivacyFilterTokenClassifier.from_config(
            model.get_config()
        )
        self.assertIs(restored.activation, keras.activations.softmax)
        self.assertEqual(restored.dropout, 0.1)
        self.assertEqual(restored._classifier_dropout_layer.rate, 0.1)

    def test_dtype_policy_follows_backbone(self):
        backbone = OpenAIPrivacyFilterBackbone(
            **self.backbone_kwargs, dtype="bfloat16"
        )
        model = OpenAIPrivacyFilterTokenClassifier(backbone, 3)
        self.assertEqual(model.dtype_policy.name, "bfloat16")

    def test_padded_positions_do_not_count_toward_accuracy(self):
        # Predicts class 0 everywhere. The one real token is labelled 2 and
        # the seven padded positions carry label 0 with zero weight, so the
        # accuracy is 0.0, not 7/8.
        model = OpenAIPrivacyFilterTokenClassifier(self.backbone, 3)
        dense = model._classifier_dense
        dense.kernel.assign(np.zeros(dense.kernel.shape))
        dense.bias.assign(np.array([1.0, 0.0, 0.0]))
        x = {
            "token_ids": np.ones((1, 8), dtype="int32"),
            "padding_mask": np.ones((1, 8), dtype="int32"),
        }
        y = np.array([[2, 0, 0, 0, 0, 0, 0, 0]])
        sample_weight = np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype="float32")
        results = model.evaluate(
            x, y, sample_weight=sample_weight, return_dict=True, verbose=0
        )
        accuracy = [v for k, v in results.items() if "accuracy" in k]
        self.assertLen(accuracy, 1)
        self.assertEqual(accuracy[0], 0.0)

    @pytest.mark.large
    def test_from_preset(self):
        model = OpenAIPrivacyFilterTokenClassifier.from_preset(
            "hf://yujiepan/openai-privacy-filter-tiny-random"
        )
        logits = model.predict(["My name is John Smith."])
        self.assertEqual(logits.shape, (1, 512, 33))
        # A100 worst drift 1.25e-3 (jax and tensorflow), at least 3x headroom.
        atol = 4e-3 if running_on_gpu() else 1e-4
        self.assertAllClose(
            logits[0, 0, :5],
            [0.571173, 2.729258, 0.409495, -0.488001, 0.259987],
            atol=atol,
        )
