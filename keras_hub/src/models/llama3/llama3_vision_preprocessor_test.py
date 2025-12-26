import numpy as np

from keras_hub.src.models.llama3.llama3_tokenizer import Llama3Tokenizer
from keras_hub.src.models.llama3.llama3_vision_image_converter import (
    Llama3VisionImageConverter,
)
from keras_hub.src.models.llama3.llama3_vision_preprocessor import (
    Llama3VisionPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class Llama3VisionPreprocessorTest(TestCase):
    def setUp(self):
        # Mock Tokenizer (needs a vocab with all required special tokens)
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += ["<|begin_of_text|>", "<|end_of_text|>"]
        self.vocab += ["<|start_header_id|>", "<|end_header_id|>"]
        self.vocab += ["<|eot_id|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]

        self.tokenizer = Llama3Tokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )

        # Image Converter
        self.image_converter = Llama3VisionImageConverter(
            image_size=(16, 16), scale=1.0 / 255.0
        )

        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 10,
        }

    def test_text_only_preprocessing(self):
        """Test preprocessing with text only."""
        preprocessor = Llama3VisionPreprocessor(**self.init_kwargs)
        input_data = ["airplane"]

        output = preprocessor(input_data)

        # Verify Text Output
        self.assertIn("token_ids", output)
        self.assertIn("padding_mask", output)
        self.assertEqual(output["token_ids"].shape, (1, 10))
        self.assertEqual(output["padding_mask"].shape, (1, 10))

    def test_text_and_image_preprocessing(self):
        """Test preprocessing with both text and images."""
        preprocessor = Llama3VisionPreprocessor(**self.init_kwargs)
        input_data = {
            "text": ["airplane"],
            "images": np.random.randint(0, 255, (1, 32, 32, 3)).astype(
                "float32"
            ),
        }

        output = preprocessor(input_data)

        # Verify Text Output
        self.assertIn("token_ids", output)
        self.assertIn("padding_mask", output)
        self.assertEqual(output["token_ids"].shape, (1, 10))

        # Verify Image Output
        self.assertIn("images", output)
        self.assertEqual(output["images"].shape, (1, 16, 16, 3))

    def test_serialization(self):
        """Test get_config/from_config."""
        preprocessor = Llama3VisionPreprocessor(**self.init_kwargs)
        config = preprocessor.get_config()

        # Check config contains expected keys
        self.assertEqual(config["sequence_length"], 10)
        self.assertEqual(config["add_start_token"], True)
        self.assertEqual(config["add_end_token"], True)

    # ============================================================
    # New tests for missing coverage
    # ============================================================

    def test_image_only_preprocessing(self):
        """Test preprocessing with only images (no text)."""
        preprocessor = Llama3VisionPreprocessor(**self.init_kwargs)
        
        # Input with only images
        input_data = {
            "images": np.random.randint(0, 255, (1, 32, 32, 3)).astype(
                "float32"
            ),
        }
        
        output = preprocessor(input_data)
        
        # Should only have images, no text outputs
        self.assertIn("images", output)
        self.assertNotIn("token_ids", output)
        self.assertNotIn("padding_mask", output)
        self.assertEqual(output["images"].shape, (1, 16, 16, 3))

    def test_preprocessing_with_labels(self):
        """Test preprocessing with labels (y parameter)."""
        preprocessor = Llama3VisionPreprocessor(**self.init_kwargs)
        
        input_data = ["airplane"]
        labels = ["airport"]
        
        # Call with both x and y
        output = preprocessor(input_data, y=labels)
        
        # Should return packed x, y, sample_weight tuple
        # The keras.utils.pack_x_y_sample_weight structure
        self.assertIsNotNone(output)

    def test_sequence_length_setter(self):
        """Test sequence_length property setter."""
        preprocessor = Llama3VisionPreprocessor(**self.init_kwargs)
        
        # Build the preprocessor first (creates packer)
        preprocessor.build(None)
        
        # Initial value
        self.assertEqual(preprocessor.sequence_length, 10)
        
        # Change sequence length
        preprocessor.sequence_length = 20
        
        # Verify it changed
        self.assertEqual(preprocessor.sequence_length, 20)
        # Verify packer was updated
        self.assertEqual(preprocessor.packer.sequence_length, 20)
