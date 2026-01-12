import numpy as np
from keras_hub.src.models.modernbert.modernbert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.models.modernbert.modernbert_preprocessor import (
    ModernBertMaskedLMPreprocessor,
)

from keras_hub.src.tests.test_case import TestCase

class ModernBertMaskedLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["<|endoftext|>", "<|padding|>", "hello", "world", "<maskKeyboard>"]
        self.tokenizer = ModernBertTokenizer(
            vocabulary={v: i for i, v in enumerate(self.vocab)},
            merges=[],
        )
        self.preprocessor = ModernBertMaskedLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=8,
            mask_selection_rate=0.5, 
        )

    def test_tokenize_and_mask(self):
        input_data = ["hello world"]
        x, y, sw = self.preprocessor(input_data)

        # ===== shapes =====
        self.assertAllEqual(x["token_ids"].shape, (1, 8))
        self.assertAllEqual(x["padding_mask"].shape, (1, 8))
        
        self.assertAllEqual(x["padding_mask"][0, 2:], [False] * 6)

    def test_serialization(self):
        new_preprocessor = ModernBertMaskedLMPreprocessor.from_config(
            self.preprocessor.get_config()
        )
        self.assertEqual(new_preprocessor.sequence_length, 8)

    def test_mask_positions_output(self):
        input_data = ["hello world hello world"]
        x, y, sw = self.preprocessor(input_data)
        
        # y should contain the IDs of the tokens that were masked
        self.assertAllEqual(y.shape, (1, 96))