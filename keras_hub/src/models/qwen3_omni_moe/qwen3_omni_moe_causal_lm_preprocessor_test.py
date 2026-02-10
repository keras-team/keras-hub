import pytest
from keras import ops

from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_causal_lm_preprocessor import Qwen3OmniMoeCausalLMPreprocessor
from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_tokenizer import Qwen3OmniMoeTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniMoeCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        # Create a dummy tokenizer for testing
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
            "jumps": 11,
            "over": 12,
            "lazy": 13,
            "dog": 14,
        }
        self.merges = ["h e", "l l", "o </w>", "w o", "r l", "d </w>"]
        
        self.tokenizer = Qwen3OmniMoeTokenizer(
            vocabulary=self.vocabulary,
            merges=self.merges,
        )
        
        # Create preprocessor
        self.preprocessor = Qwen3OmniMoeCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=128,
        )
        
        # Test data
        self.test_texts = [
            "Hello, world!",
            "How are you today?",
            "The quick brown fox jumps over the lazy dog."
        ]

    def test_preprocessor_basics(self):
        """Test basic preprocessor functionality."""
        # Test preprocessing single text
        text = "Hello, world!"
        preprocessed = self.preprocessor(text)
        
        # Should return token_ids and padding_mask
        self.assertIn("token_ids", preprocessed)
        self.assertIn("padding_mask", preprocessed)
        
        # Check shapes
        self.assertEqual(len(preprocessed["token_ids"].shape), 2)  # (batch_size, seq_len)
        self.assertEqual(len(preprocessed["padding_mask"].shape), 2)  # (batch_size, seq_len)

    def test_preprocessor_batch_processing(self):
        """Test batch processing of multiple texts."""
        preprocessed = self.preprocessor(self.test_texts)
        
        # Should handle multiple texts
        batch_size = preprocessed["token_ids"].shape[0]
        seq_len = preprocessed["token_ids"].shape[1]
        
        self.assertEqual(batch_size, len(self.test_texts))
        self.assertEqual(seq_len, 128)  # sequence_length

    def test_preprocessor_sequence_length(self):
        """Test that preprocessor respects sequence_length parameter."""
        # Test with different sequence lengths
        for seq_len in [32, 64, 128]:
            preprocessor = Qwen3OmniMoeCausalLMPreprocessor(
                tokenizer=self.tokenizer,
                sequence_length=seq_len,
            )
            
            preprocessed = preprocessor("Hello, world!")
            actual_seq_len = preprocessed["token_ids"].shape[1]
            self.assertEqual(actual_seq_len, seq_len)

    def test_preprocessor_padding(self):
        """Test padding behavior."""
        # Test with short text
        short_text = "Hello"
        preprocessed = self.preprocessor(short_text)
        
        # Should pad to sequence_length
        seq_len = preprocessed["token_ids"].shape[1]
        self.assertEqual(seq_len, 128)
        
        # Padding mask should indicate valid tokens
        padding_mask = preprocessed["padding_mask"]
        self.assertIn(0, padding_mask.numpy().flatten())  # Should have padding tokens

    def test_preprocessor_truncation(self):
        """Test truncation behavior."""
        # Create very long text
        long_text = " ".join(["word"] * 200)  # Very long text
        
        preprocessed = self.preprocessor(long_text)
        
        # Should truncate to sequence_length
        seq_len = preprocessed["token_ids"].shape[1]
        self.assertEqual(seq_len, 128)

    def test_preprocessor_special_tokens(self):
        """Test that special tokens are properly handled."""
        # Test with text that might trigger special token handling
        text = "Hello <|im_end|> world"
        preprocessed = self.preprocessor(text)
        
        # Should handle special tokens correctly
        self.assertIn("token_ids", preprocessed)
        self.assertIn("padding_mask", preprocessed)

    def test_preprocessor_empty_input(self):
        """Test preprocessor with empty input."""
        preprocessed = self.preprocessor("")
        
        # Should handle empty input gracefully
        self.assertIn("token_ids", preprocessed)
        self.assertIn("padding_mask", preprocessed)
        
        # Should still have correct shape
        seq_len = preprocessed["token_ids"].shape[1]
        self.assertEqual(seq_len, 128)

    def test_preprocessor_different_input_types(self):
        """Test preprocessor with different input types."""
        # Test with string
        preprocessed_str = self.preprocessor("Hello, world!")
        
        # Test with list of strings
        preprocessed_list = self.preprocessor(["Hello", "world"])
        
        # Both should work
        self.assertIn("token_ids", preprocessed_str)
        self.assertIn("token_ids", preprocessed_list)

    def test_preprocessor_from_preset(self):
        """Test loading preprocessor from preset."""
        try:
            preprocessor = Qwen3OmniMoeCausalLMPreprocessor.from_preset("qwen3_omni_moe_7b")
            
            # If successful, test basic functionality
            preprocessed = preprocessor("test")
            self.assertIn("token_ids", preprocessed)
        except (ValueError, ImportError):
            # Skip if preset not available
            pytest.skip("Preset not available")

    def test_preprocessor_consistency(self):
        """Test that preprocessor gives consistent results."""
        text = "Hello, world!"
        
        # Preprocess same text multiple times
        results = []
        for _ in range(3):
            preprocessed = self.preprocessor(text)
            results.append(preprocessed)
        
        # Results should be identical
        for i in range(1, len(results)):
            self.assertTrue(
                ops.allclose(results[0]["token_ids"], results[i]["token_ids"])
            )
            self.assertTrue(
                ops.allclose(results[0]["padding_mask"], results[i]["padding_mask"])
            )

    def test_preprocessor_tokenizer_integration(self):
        """Test integration with tokenizer."""
        # Test that preprocessor uses tokenizer correctly
        text = "Hello, world!"
        preprocessed = self.preprocessor(text)
        
        # Token IDs should be valid vocabulary indices
        token_ids = preprocessed["token_ids"]
        vocab_size = len(self.tokenizer.get_vocabulary())
        
        # All token IDs should be within vocabulary range
        self.assertTrue(ops.all(token_ids >= 0))
        self.assertTrue(ops.all(token_ids < vocab_size))

    def test_preprocessor_output_format(self):
        """Test that preprocessor output format matches model expectations."""
        preprocessed = self.preprocessor("Hello, world!")
        
        # Should return dictionary with required keys
        required_keys = ["token_ids", "padding_mask"]
        for key in required_keys:
            self.assertIn(key, preprocessed)
        
        # Should have correct data types
        self.assertEqual(preprocessed["token_ids"].dtype.name, "int32")
        self.assertEqual(preprocessed["padding_mask"].dtype.name, "int32")

    def test_preprocessor_with_long_sequences(self):
        """Test preprocessor with sequences at the boundary."""
        # Test with sequence exactly at sequence_length
        text = " ".join(["word"] * 100)  # Long text
        
        preprocessed = self.preprocessor(text)
        
        # Should handle long sequences correctly
        seq_len = preprocessed["token_ids"].shape[1]
        self.assertEqual(seq_len, 128)
        
        # Should have valid padding mask
        padding_mask = preprocessed["padding_mask"]
        self.assertIn(0, padding_mask.numpy().flatten())  # Should have padding
        self.assertIn(1, padding_mask.numpy().flatten())  # Should have valid tokens
