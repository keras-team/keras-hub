import pytest

from keras_hub.src.models.esm.esm_backbone import ESMBackbone
from keras_hub.src.models.esm.esm_classifier import ESMProteinClassifier
from keras_hub.src.models.esm.esm_classifier_preprocessor import (
    ESMProteinClassifierPreprocessor,
)
from keras_hub.src.models.esm.esm_tokenizer import ESMTokenizer
from keras_hub.src.tests.test_case import TestCase


class ESMProteinClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.vocab = ["<pad>", "<unk>", "<cls>", "<eos>", "<mask>"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = ESMProteinClassifierPreprocessor(
            ESMTokenizer(vocabulary=self.vocab),
            sequence_length=5,
        )
        self.backbone = ESMBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
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
            cls=ESMProteinClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ESMProteinClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=ESMProteinClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
