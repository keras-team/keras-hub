import os
import pytest
import tensorflow as tf
import numpy as np
from keras import backend
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras_hub.src.models.layoutlmv3.layoutlmv3_tokenizer import LayoutLMv3Tokenizer

@test_combinations.run_all_keras_modes
class LayoutLMv3TokenizerTest(test_combinations.TestCase):
    def setUp(self):
        super(LayoutLMv3TokenizerTest, self).setUp()
        
        # Create a dummy vocabulary
        self.vocab = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "##s",
            "##ing",
            "##ed",
        ]
        
        self.tokenizer = LayoutLMv3Tokenizer(
            vocabulary=self.vocab,
            lowercase=True,
            strip_accents=True,
        )
    
    def test_tokenizer_basics(self):
        """Test the basic functionality of the tokenizer."""
        # Test tokenizer creation
        self.assertIsInstance(self.tokenizer, LayoutLMv3Tokenizer)
        
        # Test special tokens
        self.assertEqual(self.tokenizer.cls_token, "[CLS]")
        self.assertEqual(self.tokenizer.sep_token, "[SEP]")
        self.assertEqual(self.tokenizer.pad_token, "[PAD]")
        self.assertEqual(self.tokenizer.mask_token, "[MASK]")
        self.assertEqual(self.tokenizer.unk_token, "[UNK]")
        
        # Test tokenization
        text = "The quick brown fox jumps over the lazy dog"
        outputs = self.tokenizer(text)
        
        self.assertIsInstance(outputs, dict)
        self.assertIn("token_ids", outputs)
        self.assertIn("padding_mask", outputs)
        self.assertIn("attention_mask", outputs)
        
        # Check output shapes
        token_ids = outputs["token_ids"]
        padding_mask = outputs["padding_mask"]
        attention_mask = outputs["attention_mask"]
        
        self.assertEqual(token_ids.shape[0], 1)  # batch size
        self.assertEqual(padding_mask.shape[0], 1)  # batch size
        self.assertEqual(attention_mask.shape[0], 1)  # batch size
        self.assertEqual(token_ids.shape[1], padding_mask.shape[1])  # sequence length
        self.assertEqual(token_ids.shape[1], attention_mask.shape[1])  # sequence length
    
    def test_tokenizer_special_tokens(self):
        """Test that special tokens are correctly added."""
        text = "The quick brown fox"
        outputs = self.tokenizer(text)
        token_ids = outputs["token_ids"][0]  # Get first sequence
        
        # Check that [CLS] is at the beginning
        self.assertEqual(token_ids[0], self.tokenizer.cls_token_id)
        
        # Check that [SEP] is at the end
        self.assertEqual(token_ids[-1], self.tokenizer.sep_token_id)
        
        # Check that padding mask is correct
        padding_mask = outputs["padding_mask"][0]
        self.assertEqual(padding_mask[0], 1)  # [CLS] token
        self.assertEqual(padding_mask[-1], 1)  # [SEP] token
        self.assertTrue(tf.reduce_all(padding_mask[1:-1] == 1))  # All other tokens
    
    def test_tokenizer_batch(self):
        """Test tokenization with batch inputs."""
        texts = [
            "The quick brown fox",
            "The lazy dog jumps",
        ]
        outputs = self.tokenizer(texts)
        
        # Check batch dimension
        self.assertEqual(outputs["token_ids"].shape[0], 2)
        self.assertEqual(outputs["padding_mask"].shape[0], 2)
        self.assertEqual(outputs["attention_mask"].shape[0], 2)
        
        # Check that each sequence has [CLS] and [SEP]
        for i in range(2):
            token_ids = outputs["token_ids"][i]
            self.assertEqual(token_ids[0], self.tokenizer.cls_token_id)
            self.assertEqual(token_ids[-1], self.tokenizer.sep_token_id)
    
    def test_tokenizer_detokenize(self):
        """Test detokenization."""
        text = "The quick brown fox"
        outputs = self.tokenizer(text)
        token_ids = outputs["token_ids"]
        
        # Detokenize
        detokenized = self.tokenizer.detokenize(token_ids)
        
        # Check that special tokens are removed
        self.assertNotIn("[CLS]", detokenized[0])
        self.assertNotIn("[SEP]", detokenized[0])
        
        # Check that the text is preserved (up to tokenization)
        self.assertIn("quick", detokenized[0].lower())
        self.assertIn("brown", detokenized[0].lower())
        self.assertIn("fox", detokenized[0].lower())
    
    def test_tokenizer_save_and_load(self):
        """Test saving and loading the tokenizer."""
        # Save the tokenizer
        save_path = os.path.join(self.get_temp_dir(), "layoutlmv3_tokenizer")
        self.tokenizer.save(save_path)
        
        # Load the tokenizer
        loaded_tokenizer = tf.keras.models.load_model(save_path)
        
        # Test loaded tokenizer
        text = "The quick brown fox"
        original_outputs = self.tokenizer(text)
        loaded_outputs = loaded_tokenizer(text)
        
        # Compare outputs
        tf.debugging.assert_equal(
            original_outputs["token_ids"], loaded_outputs["token_ids"]
        )
        tf.debugging.assert_equal(
            original_outputs["padding_mask"], loaded_outputs["padding_mask"]
        )
        tf.debugging.assert_equal(
            original_outputs["attention_mask"], loaded_outputs["attention_mask"]
        )
    
    def test_tokenizer_unknown_tokens(self):
        """Test handling of unknown tokens."""
        text = "The xyz abc"  # Contains unknown words
        outputs = self.tokenizer(text)
        token_ids = outputs["token_ids"][0]
        
        # Check that unknown tokens are replaced with [UNK]
        for token_id in token_ids[1:-1]:  # Skip [CLS] and [SEP]
            if token_id not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]:
                self.assertEqual(token_id, self.tokenizer.unk_token_id) 