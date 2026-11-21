import numpy as np
import pytest

from keras_hub.src.models.depth_anything.depth_anything_backbone import (
    DepthAnythingBackbone,
)
from keras_hub.src.models.depth_anything.depth_anything_depth_estimator import (
    DepthAnythingDepthEstimator,
)
from keras_hub.src.models.depth_anything.depth_anything_depth_estimator_preprocessor import (  # noqa: E501
    DepthAnythingDepthEstimatorPreprocessor,
)
from keras_hub.src.models.depth_anything.depth_anything_image_converter import (
    DepthAnythingImageConverter,
)
from keras_hub.src.models.dinov2.dinov2_backbone import DINOV2Backbone
from keras_hub.src.tests.test_case import TestCase


class DepthAnythingDepthEstimatorTest(TestCase):
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
        )
        self.images = np.ones((2, 126, 126, 3), dtype="float32")
        self.depths = np.zeros((2, 126, 126, 1), dtype="float32")
        self.image_converter = DepthAnythingImageConverter(
            image_size=(126, 126)
        )
        self.preprocessor = DepthAnythingDepthEstimatorPreprocessor(
            self.image_converter
        )
        self.backbone = DepthAnythingBackbone(
            image_encoder=image_encoder,
            reassemble_factors=[4, 2, 1, 0.5],
            neck_hidden_dims=[16, 32, 64, 128],
            fusion_hidden_dim=128,
            head_hidden_dim=16,
            head_in_index=-1,
            feature_keys=["stage1", "stage2", "stage3", "stage4"],
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "depth_estimation_type": "metric",
            "max_depth": 10.0,
            "preprocessor": self.preprocessor,
        }
        self.train_data = (self.images, self.depths)

    def test_depth_estimator_basics(self):
        self.run_task_test(
            cls=DepthAnythingDepthEstimator,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape={"depths": (2, 126, 126, 1)},
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        image_batch = (
            self.load_test_image(target_size=(518, 518))[None, ...] / 255.0
        )
        self.run_preset_test(
            cls=DepthAnythingDepthEstimator,
            preset="depth_anything_v2_small",
            input_data=image_batch,
            init_kwargs={
                "depth_estimation_type": "relative",
                "max_depth": None,
            },
            expected_output_shape={"depths": (1, 518, 518, 1)},
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DepthAnythingDepthEstimator,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=DepthAnythingDepthEstimator,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
            comparison_mode="statistical",
            output_thresholds={"depths": {"max": 2e-4, "mean": 1e-5}},
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        images = np.ones((2, 518, 518, 3), dtype="float32")
        for preset in DepthAnythingDepthEstimator.presets:
            self.run_preset_test(
                cls=DepthAnythingDepthEstimator,
                preset=preset,
                init_kwargs={
                    "depth_estimation_type": "relative",
                    "max_depth": None,
                },
                input_data=images,
                expected_output_shape={"depths": (2, 518, 518, 1)},
            )
