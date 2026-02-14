import keras
import numpy as np

# FIX: Added '_causal_lm' to the import path to match the filename you created
from keras_hub.src.models.qwen2_vl.qwen2_vl_causal_lm_preprocessor import (
    Qwen2VLCausalLMPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


# 1. Mock the Image Converter (Simulates resizing images)
class MockImageConverter(keras.layers.Layer):
    def call(self, x):
        # Return a fake 5D tensor: (Batch, Time, Height, Width, Channels)
        return np.random.random((1, 1, 14, 14, 3)).astype("float32")


# 2. Mock the Tokenizer (Simulates turning text into numbers)
class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0

    def __call__(self, text):
        # Just return random integers simulating token IDs
        # Shape: (Batch, Seq_Len) -> (1, 5)
        return np.array([[1, 2, 3, 4, 5]], dtype="int32")


class Qwen2VLIntegrationTest(TestCase):
    def test_end_to_end_flow(self):
        # Setup Preprocessor with Mocks
        preprocessor = Qwen2VLCausalLMPreprocessor(
            tokenizer=MockTokenizer(),
            image_converter=MockImageConverter(),
            sequence_length=16,
        )

        # Inputs
        # Note: In a real scenario, this would be a real image path or array
        input_data = {
            "text": "Hello world",
            "images": np.random.random((224, 224, 3)),
        }

        # Run Preprocessor
        # The preprocessor handles the dictionary unpacking
        processed = preprocessor.generate_preprocess(input_data)

        # Verify Structure
        self.assertTrue("token_ids" in processed)
        self.assertTrue("images" in processed)

        # Check shapes
        # Token IDs should come from our MockTokenizer (1, 5)
        self.assertEqual(processed["token_ids"].shape, (1, 5))

        # Images should come from our MockImageConverter (1, 1, 14, 14, 3)
        self.assertEqual(processed["images"].shape, (1, 1, 14, 14, 3))

        print("\n✅ End-to-End Preprocessing flow successful!")
