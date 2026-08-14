import pytest

from keras_hub.src.models.modernbert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modernbert.modern_bert_text_classifier import (
    ModernBertTextClassifier,
)
from keras_hub.src.models.modernbert.modern_bert_text_classifier_preprocessor import (
    ModernBertTextClassifierPreprocessor,
)
from keras_hub.src.models.modernbert.modern_bert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertTextClassifierTest(TestCase):
    def setUp(self):
        # Simple byte-pair vocabulary for testing.
        self.merges = [
            "Ġ a",
            "Ġ t",
            "Ġ i",
            "Ġ b",
            "a i",
            "p l",
            "n e",
            "Ġa t",
            "p o",
            "r t",
            "Ġt h",
            "ai r",
            "pl a",
            "po rt",
            "Ġai r",
            "Ġa i",
            "pla ne",
        ]

        self.vocab = []

        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend(
                [
                    a,
                    b,
                    a + b,
                ]
            )

        self.vocab += [
            "<|endoftext|>",
            "<|padding|>",
            "[MASK]",
        ]

        self.vocab = sorted(set(self.vocab))
        self.vocab = {token: i for i, token in enumerate(self.vocab)}

        self.tokenizer = ModernBertTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )

        self.preprocessor = ModernBertTextClassifierPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=10,
        )

        self.vocabulary_size = self.preprocessor.tokenizer.vocabulary_size()

        self.backbone = ModernBertBackbone(
            vocabulary_size=self.vocabulary_size,
            hidden_dim=16,
            intermediate_dim=32,
            num_layers=2,
            num_heads=2,
            local_attention_window=8,
            global_attn_every_n_layers=2,
            dropout=0.0,
            rotary_max_wavelength=10000,
            layer_norm_epsilon=1e-5,
        )

        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
            "num_classes": 2,
        }

        self.train_data = (
            [
                " airplane at airport",
                " airplane airport",
            ],
            [1, 0],
        )

        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_classifier_basics(self):
        self.run_task_test(
            cls=ModernBertTextClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ModernBertTextClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=ModernBertTextClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
