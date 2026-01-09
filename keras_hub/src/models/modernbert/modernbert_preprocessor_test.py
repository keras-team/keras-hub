import numpy as np
from keras import ops
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.models.modernbert.modernbert_tokenizer import ModernBertTokenizer
from keras_hub.src.models.modernbert.modernbert_preprocessor import ModernBertPreprocessor

class ModernBertPreprocessorTest(TestCase):
    """
    Test suite for ModernBertPreprocessor.
    """
    def setUp(self):
        self.vocab = {
            "<|endoftext|>": 0,
            "<|padding|>": 1,
            "a": 2, "b": 3, "c": 4, "ab": 5,
        }
        self.merges = ["a b"]
        self.tokenizer = ModernBertTokenizer(
            vocabulary=self.vocab, 
            merges=self.merges
        )
        self.preprocessor = ModernBertPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=4
        )

    def test_preprocess_dict(self):
        """Check that output is a dict with correct keys and shapes."""
        input_data = ["ab"]
        output = self.preprocessor(input_data)
        self.assertAllEqual(ops.shape(output["token_ids"]), [1, 4])
        self.assertAllEqual(ops.shape(output["padding_mask"]), [1, 4])

    def test_padding_logic(self):
        """Verify that sequence padding and ID mapping work correctly."""
        input_data = ["a"] 
        output = self.preprocessor(input_data)
        token_ids = ops.convert_to_numpy(output["token_ids"])
        # Expected: [2, 1, 1, 1] (ID 2 is 'a', ID 1 is pad)
        self.assertEqual(token_ids[0, 0], 2)
        self.assertEqual(token_ids[0, 1], 1)

    def test_serialization(self):
        """
        Ensure preprocessor can be reconstructed from config.
        """
        new_preprocessor = ModernBertPreprocessor.from_config(
            self.preprocessor.get_config()
        )
        self.assertEqual(new_preprocessor.sequence_length, 4)
        self.assertEqual(
            new_preprocessor.tokenizer.pad_token_id, 
            self.tokenizer.pad_token_id
        )