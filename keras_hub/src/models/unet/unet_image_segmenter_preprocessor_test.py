import numpy as np
import pytest

from keras_hub.src.models.unet.unet_backbone import UNetBackbone
from keras_hub.src.models.unet.unet_image_segmenter_preprocessor import (
    UNetImageSegmenterPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class UNetImageSegmenterPreprocessorTest(TestCase):
    def test_preprocessor_basics(self):
        """Test basic preprocessor functionality."""
        preprocessor = UNetImageSegmenterPreprocessor()

        # Test that backbone_cls is set correctly
        self.assertEqual(preprocessor.backbone_cls, UNetBackbone)

        # Test pass-through behavior
        images = np.random.uniform(0, 1, size=(2, 128, 128, 3)).astype(
            "float32"
        )
        labels = np.random.randint(0, 2, size=(2, 128, 128, 1)).astype("int32")

        x, y = preprocessor(images, labels)

        # Verify that data is passed through unchanged
        self.assertAllClose(x, images)
        self.assertAllClose(y, labels)

    def test_preprocessor_with_sample_weight(self):
        """Test preprocessor with sample weights."""
        preprocessor = UNetImageSegmenterPreprocessor()

        images = np.random.uniform(0, 1, size=(2, 128, 128, 3)).astype(
            "float32"
        )
        labels = np.random.randint(0, 2, size=(2, 128, 128, 1)).astype("int32")
        sample_weight = np.random.uniform(0, 1, size=(2,)).astype("float32")

        x, y, sw = preprocessor(images, labels, sample_weight)

        # Verify that all data is passed through unchanged
        self.assertAllClose(x, images)
        self.assertAllClose(y, labels)
        self.assertAllClose(sw, sample_weight)

    def test_preprocessor_different_shapes(self):
        """Test preprocessor with different input shapes."""
        preprocessor = UNetImageSegmenterPreprocessor()

        # Test with different image sizes
        for size in [64, 128, 256]:
            images = np.random.uniform(0, 1, size=(1, size, size, 3)).astype(
                "float32"
            )
            labels = np.random.randint(0, 2, size=(1, size, size, 1)).astype(
                "int32"
            )

            x, y = preprocessor(images, labels)

            self.assertAllClose(x, images)
            self.assertAllClose(y, labels)

    @pytest.mark.large
    def test_saved_model(self):
        """Test saving and loading the preprocessor."""
        self.run_preprocessing_layer_test(
            cls=UNetImageSegmenterPreprocessor,
            init_kwargs={},
            input_data={
                "x": np.random.uniform(0, 1, size=(2, 128, 128, 3)).astype(
                    "float32"
                ),
                "y": np.random.randint(0, 2, size=(2, 128, 128, 1)).astype(
                    "int32"
                ),
            },
        )
