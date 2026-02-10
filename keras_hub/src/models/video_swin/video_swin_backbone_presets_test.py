"""Tests for loading pretrained Video Swin model presets."""

import numpy as np
import pytest

from keras_hub.src.models.video_swin.video_swin_backbone import (
    VideoSwinBackbone,
)
from keras_hub.src.tests.test_case import TestCase


@pytest.mark.large
class VideoSwinPresetSmokeTest(TestCase):
    """Smoke test for Video Swin presets with minimal inference.

    Run with:
    `pytest
    keras_hub/models/backbones/video_swin/video_swin_backbone_presets_test.py
        --run_large`
    """

    def setUp(self):
        self.input_data = np.ones(shape=(1, 32, 224, 224, 3))

    def test_backbone_presets(self):
        # Minimal test for each backbone preset (with weights and no weights)
        for preset in VideoSwinBackbone.presets:
            with self.subTest(preset=preset):
                self.run_backbone_test(
                    cls=VideoSwinBackbone,
                    preset=preset,
                    input_data=self.input_data,
                    required_output_keys=["logits"],
                )


@pytest.mark.extra_large
class VideoSwinPresetFullTest(TestCase):
    """Full test suite for Video Swin presets.

    Run with:
    `pytest
    keras_hub/models/backbones/video_swin/video_swin_backbone_presets_test.py
     --run_extra_large`
    """

    def test_all_video_swin_presets(self):
        input_data = np.ones(shape=(1, 32, 224, 224, 3))
        for preset in VideoSwinBackbone.presets:
            with self.subTest(preset=preset):
                self.run_saved_model_test(
                    cls=VideoSwinBackbone,
                    preset=preset,
                    input_data=input_data,
                )
