import pytest

from keras_hub.src.models.modernbert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modernbert.modern_bert_masked_lm import (
    ModernBertMaskedLM,
)
from keras_hub.src.models.modernbert.modern_bert_masked_lm_preprocessor import (
    ModernBertMaskedLMPreprocessor,
)
from keras_hub.src.models.modernbert.modern_bert_tokenizer import (
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
            mask_selection_rate=0.0,
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

        self.train_data = (
            self.input_data,
            None,
            None,
        )

        self.model = ModernBertMaskedLM(
            **self.init_kwargs,
        )

    def test_tokenizer_serialization(self):
        """Test tokenizer serialization and deserialization."""
        config = self.tokenizer.get_config()
        restored = ModernBertTokenizer.from_config(config)
        self.assertEqual(
            restored.vocabulary_size(), self.tokenizer.vocabulary_size()
        )
        self.assertEqual(
            restored.get_vocabulary(), self.tokenizer.get_vocabulary()
        )
        self.assertEqual(restored.merges, self.tokenizer.merges)

    @pytest.mark.large
    def test_fit(self):
        """Validate training, output shape, and serialization."""
        self.run_task_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
        )

    @pytest.mark.large
    def test_saved_model(self):
        """Validate serialization lifecycle."""
        input_data = self.preprocessor(self.input_data)[0]
        self.run_model_saving_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
        )
