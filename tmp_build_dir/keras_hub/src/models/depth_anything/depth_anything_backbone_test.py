import numpy as np
import pytest

from keras_hub.src.models.depth_anything.depth_anything_backbone import (
    DepthAnythingBackbone,
)
from keras_hub.src.models.dinov2.dinov2_backbone import DINOV2Backbone
from keras_hub.src.tests.test_case import TestCase


class DepthAnythingBackboneTest(TestCase):
    def setUp(self):
        image_encoder = DINOV2Backbone(
            14,
            4,
            16,
            2,
            16 * 4,
            1.0,
            0,
            image_shape=(126, 126, 3),
            apply_layernorm=True,
            name="image_encoder",
        )
        self.init_kwargs = {
            "image_encoder": image_encoder,
            "reassemble_factors": [4, 2, 1, 0.5],
            "neck_hidden_dims": [16, 32, 64, 128],
            "fusion_hidden_dim": 128,
            "head_hidden_dim": 16,
            "head_in_index": -1,
            "feature_keys": ["stage1", "stage2", "stage3", "stage4"],
        }
        self.input_data = np.ones((2, 126, 126, 3), dtype="float32")

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=DepthAnythingBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 126, 126, 1),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DepthAnythingBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_preset(self):
        image_batch = (
            self.load_test_image(target_size=(518, 518))[None, ...] / 255.0
        )
        self.run_preset_test(
            cls=DepthAnythingBackbone,
            preset="depth_anything_v2_small",
            input_data=image_batch,
            expected_output_shape=(1, 518, 518, 1),
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        image_batch = (
            self.load_test_image(target_size=(518, 518))[None, ...] / 255.0
        )
        for preset in DepthAnythingBackbone.presets:
            self.run_preset_test(
                cls=DepthAnythingBackbone,
                preset=preset,
                input_data=image_batch,
                expected_output_shape=(1, 518, 518, 1),
            )
