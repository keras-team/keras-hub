import pytest

from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modernbert.modernbert_masked_lm import (
    ModernBertMaskedLM,
)
from keras_hub.src.models.modernbert.modernbert_preprocessor import (
    ModernBertMaskedLMPreprocessor,
)
from keras_hub.src.models.modernbert.modernbert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertMaskedLMTest(TestCase):
    """Tests for verifying the `ModernBertMaskedLM` task model."""

    def setUp(self):
        self.vocab = [
            "<|endoftext|>",
            "<|padding|>",
            "[MASK]",
            "[UNK]",
            "Ġ",
            "a",
            "t",
            "i",
            "b",
            "p",
            "l",
            "n",
            "e",
            "o",
            "r",
            "h",
            "ai",
            "pl",
            "po",
            "rt",
            "th",
            "air",
            "pla",
            "ne",
            "port",
            "plane",
            "Ġa",
            "Ġt",
            "Ġi",
            "Ġb",
            "Ġat",
            "Ġair",
        ]

        self.vocab = {t: i for i, t in enumerate(self.vocab)}

        self.merges = [
            "Ġ a",
            "Ġ t",
            "a i",
            "p l",
            "n e",
            "p o",
            "r t",
            "t h",
            "ai r",
            "pl a",
            "po rt",
            "pla ne",
        ]

        self.tokenizer = ModernBertTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )

        self.preprocessor = ModernBertMaskedLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=12,
            mask_selection_rate=0.2,
            mask_selection_length=2,
        )

        self.backbone = ModernBertBackbone(
            vocabulary_size=self.tokenizer.vocabulary_size,
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            local_attention_window=128,
        )

        self.model = ModernBertMaskedLM(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
        )

        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }

        self.input_data = ["airplane airport", "airplane"]

    @pytest.mark.extra_large
    def test_fit(self):
        """
        Validate model execution and compilation
        using standard training APIs.
        """
        input_data = ["airplane airport", "airplane"]
        self.model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy"
        )

        self.model.fit(input_data, epochs=1)

    @pytest.mark.large
    def test_saved_model(self):
        """
        Validate serialization lifecycle routines and
        graph re-instantiation.
        """
        self.run_model_saving_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
