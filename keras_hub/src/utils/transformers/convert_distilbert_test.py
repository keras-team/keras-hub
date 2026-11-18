import json
import os
import tempfile

import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.distil_bert.distil_bert_backbone import (
    DistilBertBackbone,
)
from keras_hub.src.models.distil_bert.distil_bert_masked_lm import (
    DistilBertMaskedLM,
)
from keras_hub.src.models.distil_bert.distil_bert_text_classifier import (
    DistilBertTextClassifier,
)
from keras_hub.src.models.masked_lm import MaskedLM
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

    def _hf_config(self, **kwargs):
        transformers = pytest.importorskip("transformers")

        return transformers.DistilBertConfig(
            vocab_size=100,
            dim=32,
            n_layers=2,
            n_heads=4,
            hidden_dim=37,
            max_position_embeddings=64,
            **kwargs,
        )

    def _save_hf_sequence_classifier(
        self, preset_dir, num_labels, **config_kwargs
    ):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        config = self._hf_config(
            num_labels=num_labels,
            architectures=["DistilBertForSequenceClassification"],
            **config_kwargs,
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

    def _rewrite_config(self, preset_dir, remove=(), **update):
        path = os.path.join(preset_dir, "config.json")
        with open(path) as f:
            config = json.load(f)
        for key in remove:
            config.pop(key, None)
        config.update(update)
        with open(path, "w") as f:
            json.dump(config, f)

    def _assert_logits_match(self, keras_model, hf_model):
        torch = pytest.importorskip("torch")

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

    @parameterized.named_parameters(
        # `config.json` omits `id2label` for the default two-label setup.
        ("default_two_labels", 2),
        ("three_labels", 3),
    )
    @pytest.mark.large
    def test_sequence_classification_head(self, num_labels):
        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model = self._save_hf_sequence_classifier(preset_dir, num_labels)
            keras_model = TextClassifier.from_preset(
                preset_dir, preprocessor=None
            )

        self.assertIsInstance(keras_model, DistilBertTextClassifier)
        self.assertEqual(keras_model.num_classes, num_labels)
        self._assert_logits_match(keras_model, hf_model)

    @parameterized.named_parameters(
        ("num_labels", "num_labels"),
        ("private_num_labels", "_num_labels"),
    )
    @pytest.mark.large
    def test_sequence_classification_head_without_id2label(self, key):
        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model = self._save_hf_sequence_classifier(
                preset_dir, num_labels=3
            )
            # A config can give the label count without `id2label`, as
            # `num_labels` or as the legacy `_num_labels` key.
            self._rewrite_config(
                preset_dir, remove=("id2label", "label2id"), **{key: 3}
            )
            keras_model = TextClassifier.from_preset(
                preset_dir, preprocessor=None
            )

        self.assertEqual(keras_model.num_classes, 3)
        self._assert_logits_match(keras_model, hf_model)

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

    @pytest.mark.large
    def test_sequence_classification_head_without_weights(self):
        with tempfile.TemporaryDirectory() as preset_dir:
            self._save_hf_sequence_classifier(preset_dir, num_labels=3)
            # `load_weights=False` must not read the weights file at all.
            os.remove(os.path.join(preset_dir, "model.safetensors"))
            keras_model = TextClassifier.from_preset(
                preset_dir, preprocessor=None, load_weights=False
            )

        self.assertIsInstance(keras_model, DistilBertTextClassifier)
        self.assertEqual(keras_model.num_classes, 3)

    @pytest.mark.large
    def test_sequence_classification_dropout(self):
        with tempfile.TemporaryDirectory() as preset_dir:
            self._save_hf_sequence_classifier(
                preset_dir, num_labels=3, seq_classif_dropout=0.35
            )
            keras_model = TextClassifier.from_preset(
                preset_dir, preprocessor=None
            )
            # A user-supplied `dropout` wins over the checkpoint's value.
            override_model = TextClassifier.from_preset(
                preset_dir, preprocessor=None, dropout=0.5
            )
            # Other DistilBERT tasks do not take `dropout`. They must still
            # load from the same checkpoint.
            masked_lm = MaskedLM.from_preset(preset_dir, preprocessor=None)

        self.assertEqual(keras_model.dropout, 0.35)
        self.assertEqual(keras_model.output_dropout.rate, 0.35)
        self.assertEqual(override_model.dropout, 0.5)
        self.assertIsInstance(masked_lm, DistilBertMaskedLM)
        self.assertNotIn("dropout", masked_lm.get_config())

    @pytest.mark.large
    def test_sequence_classification_head_rejects_hidden_dim(self):
        with tempfile.TemporaryDirectory() as preset_dir:
            self._save_hf_sequence_classifier(preset_dir, num_labels=3)
            # The checkpoint's head has `dim=32`. A different `hidden_dim`
            # cannot hold the fine-tuned head weights.
            with self.assertRaisesRegex(ValueError, "`hidden_dim=16`"):
                TextClassifier.from_preset(
                    preset_dir, preprocessor=None, hidden_dim=16
                )
            # A `hidden_dim` that matches the checkpoint is fine.
            keras_model = TextClassifier.from_preset(
                preset_dir, preprocessor=None, hidden_dim=32
            )

        self.assertEqual(keras_model.num_classes, 3)

    @pytest.mark.large
    def test_checkpoint_without_sequence_classification_head(self):
        pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        with tempfile.TemporaryDirectory() as preset_dir:
            config = self._hf_config(architectures=["DistilBertForMaskedLM"])
            transformers.DistilBertForMaskedLM(config).save_pretrained(
                preset_dir
            )
            # There is no head to restore, so `num_classes` is required.
            with self.assertRaisesRegex(ValueError, "Pass `num_classes`"):
                TextClassifier.from_preset(preset_dir, preprocessor=None)
            keras_model = TextClassifier.from_preset(
                preset_dir, preprocessor=None, num_classes=4
            )

        self.assertIsInstance(keras_model, DistilBertTextClassifier)
        self.assertEqual(keras_model.num_classes, 4)
