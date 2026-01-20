import os
import pytest

from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modernbert.modernbert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.models.modernbert.modernbert_masked_lm import (
    ModernBertMaskedLM,
)
from keras_hub.src.models.modernbert.modernbert_preprocessor import (
    ModernBertMaskedLMPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertMaskedLMTest(TestCase):
    def setUp(self):
        self.vocab = ["<|padding|>", "<|endoftext|>", "[MASK]", "the", "quick", "brown"]
        self.vocab += [f"token_{i}" for i in range(10)]
        self.vocabulary = {token: i for i, token in enumerate(self.vocab)}
        
        self.merges = ["t h", "th e", "q u", "qu i", "qui ck"]
        
        self.tokenizer = ModernBertTokenizer(
            vocabulary=self.vocabulary,
            merges=self.merges,
        )

        self.preprocessor = ModernBertMaskedLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=10,
            mask_selection_rate=0.2,
            mask_selection_length=2,
        )
        
        self.backbone = ModernBertBackbone(
            vocabulary_size=len(self.vocab),
            hidden_dim=16,
            intermediate_dim=32,
            local_attention_window=128,
            num_layers=2,
            num_heads=2,
        )
        
        self.model = ModernBertMaskedLM(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
        )

    @pytest.mark.extra_large
    def test_fit(self):
        # Verify the model can actually train (one step)
        input_data = ["the quick brown", "the quick"]
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        print(type(input_data))
        ds = self.model.preprocessor(input_data)
        self.model.fit(ds, epochs=1)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
