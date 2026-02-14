import numpy as np

from keras_hub.src.models.qwen2_vl.qwen2_vl_causal_lm_preprocessor import (
    Qwen2VLCausalLMPreprocessor,
)

# FIX: Import the Real Image Converter
from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (
    Qwen2VLImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0

    def __call__(self, text):
        return np.array([[1, 2, 3, 4, 5]], dtype="int32")


class Qwen2VLIntegrationTest(TestCase):
    def test_smart_resizing_flow(self):
        # 1. Setup Real Converter
        # We set min_pixels small so we can test resizing easily
        image_converter = Qwen2VLImageConverter(
            min_pixels=100 * 100, max_pixels=1000 * 1000
        )

        # 2. Setup Preprocessor
        preprocessor = Qwen2VLCausalLMPreprocessor(
            tokenizer=MockTokenizer(),
            image_converter=image_converter,
            sequence_length=16,
        )

        # 3. Create a weirdly shaped image (e.g., 50x300 - very wide)
        # The smart resizer should try to keep this aspect ratio
        input_h, input_w = 50, 300
        raw_image = np.random.randint(0, 255, (input_h, input_w, 3)).astype(
            "float32"
        )

        input_data = {"text": "Hello world", "images": raw_image}

        # 4. Run Preprocessor
        processed = preprocessor.generate_preprocess(input_data)

        # 5. Verify Structure
        images = processed["images"]
        print(f"\nOriginal Shape: {(input_h, input_w)}")
        print(f"Resized Shape:  {images.shape}")

        # Check 1: It should be 4D (Time, H, W, C)
        self.assertEqual(len(images.shape), 4)

        # Check 2: Time dimension should be 1
        self.assertEqual(images.shape[0], 1)

        # Check 3: Dimensions should be multiples of 28 (The 'snap' logic)
        h, w = images.shape[1], images.shape[2]
        self.assertTrue(h % 28 == 0, f"Height {h} is not multiple of 28")
        self.assertTrue(w % 28 == 0, f"Width {w} is not multiple of 28")

        print("✅ Smart Resizing Logic successful!")
