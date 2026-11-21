import pytest

from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.models.bert.bert_text_classifier import BertTextClassifier
from keras_hub.src.models.bert.bert_text_classifier_preprocessor import (
    BertTextClassifierPreprocessor,
)
from keras_hub.src.models.bert.bert_tokenizer import BertTokenizer
from keras_hub.src.tests.test_case import TestCase


class BertTextClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = BertTextClassifierPreprocessor(
            BertTokenizer(vocabulary=self.vocab),
            sequence_length=5,
        )
        self.backbone = BertBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=self.preprocessor.sequence_length,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
            "num_classes": 2,
        }
        self.train_data = (
            ["the quick brown fox.", "the slow brown fox."],  # Features.
            [1, 0],  # Labels.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_classifier_basics(self):
        self.run_task_test(
            cls=BertTextClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BertTextClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=BertTextClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=BertTextClassifier,
            preset="bert_tiny_en_uncased_sst2",
            input_data=self.input_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BertTextClassifier.presets:
            self.run_preset_test(
                cls=BertTextClassifier,
                preset=preset,
                init_kwargs={"num_classes": 2},
                input_data=self.input_data,
                expected_output_shape=(2, 2),
            )
