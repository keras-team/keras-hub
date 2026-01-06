import numpy as np
import pytest

from keras_hub.src.models.mit.mit_backbone import MiTBackbone
from keras_hub.src.models.mit.mit_image_classifier import MiTImageClassifier
from keras_hub.src.tests.test_case import TestCase


class MiTImageClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.images = np.ones((2, 32, 32, 3), dtype="float32")
        self.labels = [0, 3]
        self.backbone = MiTBackbone(
            layerwise_depths=[2, 2, 2, 2],
            image_shape=(32, 32, 3),
            hidden_dims=[4, 8],
            num_layers=2,
            layerwise_num_heads=[1, 2],
            layerwise_sr_ratios=[8, 4],
            max_drop_path_rate=0.1,
            layerwise_patch_sizes=[7, 3],
            layerwise_strides=[4, 2],
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 2,
            "activation": "softmax",
        }
        self.train_data = (
            self.images,
            self.labels,
        )

    def test_classifier_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_task_test(
            cls=MiTImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(4, 4),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MiTImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=MiTImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
