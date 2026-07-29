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
    """Tests for verifying the ModernBERT MaskedLM task model."""

    def setUp(self):
        vocab = [
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

        vocab = {token: index for index, token in enumerate(vocab)}

        merges = [
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
            vocabulary=vocab,
            merges=merges,
        )

        self.preprocessor = ModernBertMaskedLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=12,
            mask_selection_rate=0.2,
            mask_selection_length=2,
        )
        self.backbone = ModernBertBackbone(
            vocabulary_size=self.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            local_attention_window=128,
        )

        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }

        self.input_data = [
            "airplane airport",
            "airplane",
        ]

        self.model = ModernBertMaskedLM(
            **self.init_kwargs,
        )

    def test_task(self):
        """Validate task model with KerasHub standard task runner."""

        self.run_task_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_fit(self):
        """Validate training execution."""

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
        )

        self.model.fit(
            self.input_data,
            epochs=1,
        )

    @pytest.mark.large
    def test_saved_model(self):
        """Validate serialization lifecycle."""

        self.run_model_saving_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
