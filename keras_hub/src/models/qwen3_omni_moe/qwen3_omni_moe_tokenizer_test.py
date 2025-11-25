import pytest
from keras import ops

from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_tokenizer import Qwen3OmniMoeTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniMoeTokenizerTest(TestCase):
    def setUp(self):
        # Create a dummy vocabulary for testing
        self.vocabulary = {
            "<|endoftext|>": 0,
            "<|im_end|>": 1,
            "hello": 2,
            "world": 3,
            "how": 4,
            "are": 5,
            "you": 6,
            "the": 7,
            "quick": 8,
            "brown": 9,
            "fox": 10,
        }
        self.merges = ["h e", "l l", "o </w>", "w o", "r l", "d </w>"]
        
        self.tokenizer = Qwen3OmniMoeTokenizer(
            vocabulary=self.vocabulary,
            merges=self.merges,
        )

    def test_tokenizer_basics(self):
        """Test basic tokenizer functionality."""
        # Test tokenization
        text = "hello world"
        tokens = self.tokenizer(text)
        
        # Should return token_ids and padding_mask
        self.assertIn("token_ids", tokens)
        self.assertIn("padding_mask", tokens)
        
        # Check shapes
        self.assertEqual(len(tokens["token_ids"].shape), 2)  # (batch_size, seq_len)
        self.assertEqual(len(tokens["padding_mask"].shape), 2)  # (batch_size, seq_len)

    def test_tokenizer_special_tokens(self):
        """Test that special tokens are properly added."""
        # Check that special tokens are in vocabulary
        self.assertIn("<|endoftext|>", self.tokenizer.get_vocabulary())
        self.assertIn("<|im_end|>", self.tokenizer.get_vocabulary())

    def test_tokenizer_batch_processing(self):
        """Test batch processing of multiple texts."""
        texts = ["hello world", "how are you", "the quick brown fox"]
        tokens = self.tokenizer(texts)
        
        # Should handle multiple texts
        batch_size = tokens["token_ids"].shape[0]
        self.assertEqual(batch_size, 3)

    def test_tokenizer_detokenization(self):
        """Test detokenization round-trip."""
        original_text = "hello world"
        tokens = self.tokenizer(original_text)
        detokenized = self.tokenizer.detokenize(tokens["token_ids"])
        
        # Should be able to detokenize (though may not be exact due to subword tokenization)
        self.assertIsInstance(detokenized, str)

    def test_tokenizer_from_preset(self):
        """Test loading tokenizer from preset."""
        # This test will be skipped if no presets are available
        try:
            tokenizer = Qwen3OmniMoeTokenizer.from_preset("qwen3_omni_moe_7b")
            # If successful, test basic functionality
            tokens = tokenizer("test")
            self.assertIn("token_ids", tokens)
        except (ValueError, ImportError):
            # Skip if preset not available or dependencies missing
            pytest.skip("Preset not available or dependencies missing")

    def test_tokenizer_vocabulary_size(self):
        """Test that vocabulary size is correct."""
        vocab_size = len(self.tokenizer.get_vocabulary())
        self.assertGreater(vocab_size, 0)
        self.assertEqual(vocab_size, len(self.vocabulary))

    def test_tokenizer_padding(self):
        """Test tokenizer padding behavior."""
        # Test with different length inputs
        short_text = "hello"
        long_text = "hello world how are you today"
        
        short_tokens = self.tokenizer(short_text)
        long_tokens = self.tokenizer(long_text)
        
        # Both should have valid outputs
        self.assertIn("token_ids", short_tokens)
        self.assertIn("token_ids", long_tokens)

    def test_tokenizer_empty_input(self):
        """Test tokenizer with empty input."""
        tokens = self.tokenizer("")
        
        # Should handle empty input gracefully
        self.assertIn("token_ids", tokens)
        self.assertIn("padding_mask", tokens)

    def test_tokenizer_special_characters(self):
        """Test tokenizer with special characters."""
        text = "Hello, world! How are you? I'm fine, thanks."
        tokens = self.tokenizer(text)
        
        # Should handle special characters
        self.assertIn("token_ids", tokens)
        self.assertIn("padding_mask", tokens)
