import os
import pytest
from keras_hub.src.models.stablelm.stablelm_causal_lm_preprocessor import StableLMCausalLMPreprocessor
from keras_hub.src.models.stablelm.stablelm_tokenizer import StableLMTokenizer
from keras_hub.src.tests.test_case import TestCase

class StableLMCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "<|endoftext|>", "!", "air", "plane", "at", "port"]
        self.merges = ["a i", "p l", "n e", "pl a", "po rt"]  
        self.tokenizer = StableLMTokenizer(vocabulary=self.vocab, merges=self.merges)
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = ["air plane at port"] 

    def test_preprocessor_basics(self):
        preprocessor = StableLMCausalLMPreprocessor(**self.init_kwargs)
        x, y, sw = preprocessor(self.input_data)
        # Expected tokenization: "<|endoftext|> air plane at port" -> [1, 3, 4, 5, 6]
        # Padded to sequence_length=8: [1, 3, 4, 5, 6, 0, 0, 0]
        self.assertAllEqual(x["token_ids"], [[1, 3, 4, 5, 6, 0, 0, 0]])
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 0, 0, 0]])
        # Labels are shifted: [3, 4, 5, 6, 0, 0, 0, 0]
        self.assertAllEqual(y, [[3, 4, 5, 6, 0, 0, 0, 0]])
        # Sample weights are 1 where labels are non-padding
        self.assertAllEqual(sw, [[1, 1, 1, 1, 0, 0, 0, 0]])

    def test_no_start_end_token(self):
        # Test without start and end tokens, with batch size of 4
        input_data = ["air plane at port"] * 4
        preprocessor = StableLMCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=8,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        # Tokenization: "air plane at port" -> [3, 4, 5, 6]
        # Padded: [3, 4, 5, 6, 0, 0, 0, 0]
        self.assertAllEqual(x["token_ids"], [[3, 4, 5, 6, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)
        # Labels: [4, 5, 6, 0, 0, 0, 0, 0]
        self.assertAllEqual(y, [[4, 5, 6, 0, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 0, 0, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        # Test preprocessing for generation
        preprocessor = StableLMCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess("air plane at port")
        # Expected: [1, 3, 4, 5, 6, 0, 0, 0]
        self.assertAllEqual(x["token_ids"], [1, 3, 4, 5, 6, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_postprocess(self):
        # Test postprocessing for generation
        preprocessor = StableLMCausalLMPreprocessor(**self.init_kwargs)
        input_data = {
            "token_ids": [1, 3, 4, 5, 6, 0, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        x = preprocessor.generate_postprocess(input_data)
        # Expect detokenized string, may include minor formatting differences due to BPE
        self.assertEqual(x, "air plane at port")

    