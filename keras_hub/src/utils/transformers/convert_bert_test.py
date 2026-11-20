import tempfile

import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.models.bert.bert_text_classifier import BertTextClassifier
from keras_hub.src.models.bert.bert_text_embedder import BertTextEmbedder
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
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

    @pytest.mark.large
    def test_multilingual_e5_tokenizer(self):
        model = BertTextEmbedder.from_preset(
            "hf://intfloat/multilingual-e5-small",
            load_weights=False,
        )
        self.assertIsInstance(model.preprocessor.tokenizer, XLMRobertaTokenizer)

    @pytest.mark.large
    def test_sequence_classification_head_requires_num_classes(self):
        pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        config = transformers.BertConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=37,
            max_position_embeddings=64,
            num_labels=3,
            architectures=["BertForSequenceClassification"],
        )
        with tempfile.TemporaryDirectory() as preset_dir:
            transformers.BertForSequenceClassification(config).save_pretrained(
                preset_dir
            )
            # The BERT converter cannot port a classification head yet, so
            # the fine-tuned head cannot be restored.
            with self.assertRaisesRegex(ValueError, "Pass `num_classes`"):
                TextClassifier.from_preset(preset_dir, preprocessor=None)
            keras_model = TextClassifier.from_preset(
                preset_dir, preprocessor=None, num_classes=3
            )

        self.assertIsInstance(keras_model, BertTextClassifier)
        self.assertEqual(keras_model.num_classes, 3)

    # TODO: compare numerics with huggingface model.
