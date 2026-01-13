import os
import pytest
import numpy as np
from keras import ops
from keras_hub.src.tests.test_case import TestCase
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
            num_layers=2,
            num_heads=2,
        )
        
        self.model = ModernBertMaskedLM(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
        )

    def test_valid_call(self):
        # Test with raw string input
        input_data = ["the quick brown", "the quick"]
        # In pytest, we manually apply preprocessing for Functional models 
        # unless testing the higher-level Task.predict
        x, y, sw = self.preprocessor(input_data)
        output = self.model(x)
        
        # Assertions
        # Batch size 2, Mask length 2, Vocab size len(self.vocab)
        self.assertEqual(output.shape, (2, 2, len(self.vocab)))

    def test_predict(self):
        # Test the end-to-end predict() method
        input_data = ["the quick brown", "the quick"]
        output = self.model.predict(input_data)
        
        self.assertEqual(output.shape, (2, 2, len(self.vocab)))

    def test_serialization(self):
        # Ensure the model can be saved and reloaded
        new_model = ModernBertMaskedLM.from_config(self.model.get_config())
        self.assertEqual(new_model.get_config(), self.model.get_config())

    @pytest.mark.extra_large
    def test_fit(self):
        # Verify the model can actually train (one step)
        input_data = ["the quick brown", "the quick"]
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        
        # Preprocessor produces (x_dict, y_labels, sample_weights)
        ds = self.model.preprocessor(input_data)
        self.model.fit(ds, epochs=1)