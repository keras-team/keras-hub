import tempfile

import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

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
    @pytest.mark.extra_large
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

    def _save_hf_sequence_classifier(self, preset_dir, num_labels):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        config = transformers.DistilBertConfig(
            vocab_size=100,
            dim=32,
            n_layers=2,
            n_heads=4,
            hidden_dim=37,
            max_position_embeddings=64,
            num_labels=num_labels,
            architectures=["DistilBertForSequenceClassification"],
        )
        torch.manual_seed(0)
        hf_model = transformers.DistilBertForSequenceClassification(
            config
        ).eval()
        # HF zero-initializes biases. Randomize the head so that a head that
        # was never ported (zero bias, random kernel) cannot pass by accident.
        with torch.no_grad():
            for layer in (hf_model.pre_classifier, hf_model.classifier):
                layer.weight.normal_(std=0.2)
                layer.bias.normal_(std=0.2)
        hf_model.save_pretrained(preset_dir)
        return hf_model

    @parameterized.named_parameters(
        # `config.json` omits `id2label` for the default two-label setup.
        ("default_two_labels", 2),
        ("three_labels", 3),
    )
    @pytest.mark.large
    def test_sequence_classification_head(self, num_labels):
        torch = pytest.importorskip("torch")

        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model = self._save_hf_sequence_classifier(preset_dir, num_labels)
            keras_model = TextClassifier.from_preset(
                preset_dir, preprocessor=None
            )

        self.assertIsInstance(keras_model, DistilBertTextClassifier)
        self.assertEqual(keras_model.num_classes, num_labels)
        token_ids = np.array(
            [[1, 5, 9, 13, 2, 0, 0, 0], [1, 7, 11, 17, 19, 23, 29, 2]],
            dtype="int32",
        )
        padding_mask = token_ids != 0
        keras_logits = ops.convert_to_numpy(
            keras_model({"token_ids": token_ids, "padding_mask": padding_mask})
        )
        with torch.no_grad():
            hf_logits = hf_model(
                input_ids=torch.tensor(token_ids, dtype=torch.long),
                attention_mask=torch.tensor(padding_mask, dtype=torch.long),
            ).logits.numpy()
        self.assertAllClose(keras_logits, hf_logits, atol=1e-3, rtol=1e-3)

    @pytest.mark.large
    def test_num_classes_reinitializes_sequence_classification_head(self):
        with tempfile.TemporaryDirectory() as preset_dir:
            self._save_hf_sequence_classifier(preset_dir, num_labels=3)
            keras_model = DistilBertTextClassifier.from_preset(
                preset_dir, num_classes=5, preprocessor=None
            )

        self.assertEqual(keras_model.num_classes, 5)
        self.assertAllEqual(
            ops.convert_to_numpy(keras_model.output_dense.bias), np.zeros(5)
        )
