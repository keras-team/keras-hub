import numpy as np
import pytest

from keras_hub.src.models.qwen3_omni.qwen3_omni_image_converter import (
    Qwen3OmniImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniImageConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "height": 224,
            "width": 224,
        }

    def test_converter_basics(self):
        converter = Qwen3OmniImageConverter(**self.init_kwargs)
        # Create dummy image
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        output = converter(image)
        # Single image returns unbatched (height, width, channels)
        self.assertEqual(len(output.shape), 3)
        self.assertEqual(output.shape[0], 224)
        self.assertEqual(output.shape[1], 224)
        self.assertEqual(output.shape[2], 3)

    def test_batch_processing(self):
        converter = Qwen3OmniImageConverter(**self.init_kwargs)
        # Create batch of dummy images with uniform shape
        batch_size = 2
        images = np.ones((batch_size, 512, 512, 3), dtype=np.uint8) * 128
        output = converter(images)
        # Batch returns (batch, height, width, channels)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 224)
        self.assertEqual(output.shape[2], 224)
        self.assertEqual(output.shape[3], 3)

    def test_single_image(self):
        converter = Qwen3OmniImageConverter(**self.init_kwargs)
        # Create single image
        image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        output = converter(image)
        # Single image returns unbatched (height, width, channels)
        self.assertEqual(output.shape[0], 224)
        self.assertEqual(output.shape[1], 224)
        self.assertEqual(output.shape[2], 3)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen3OmniImageConverter.presets:
            self.run_preset_test(
                cls=Qwen3OmniImageConverter,
                preset=preset,
                input_data=np.ones((224, 224, 3), dtype=np.uint8),
            )
