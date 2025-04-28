import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.mit.mit_backbone import MiTBackbone
from keras_hub.src.models.segformer.segformer_backbone import SegFormerBackbone
from keras_hub.src.models.segformer.segformer_image_segmenter import (
    SegFormerImageSegmenter,
)
from keras_hub.src.tests.test_case import TestCase


class SegFormerTest(TestCase):
    def setUp(self):
        image_encoder = MiTBackbone(
            depths=[2, 2],
            image_shape=(224, 224, 3),
            hidden_dims=[32, 64],
            num_layers=2,
            blockwise_num_heads=[1, 2],
            blockwise_sr_ratios=[8, 4],
            max_drop_path_rate=0.1,
            patch_sizes=[7, 3],
            strides=[4, 2],
        )
        projection_filters = 256
        self.backbone = SegFormerBackbone(
            image_encoder=image_encoder, projection_filters=projection_filters
        )

        self.input_size = 224
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

        self.init_kwargs = {"backbone": self.backbone, "num_classes": 4}

    def test_segformer_segmenter_construction(self):
        SegFormerImageSegmenter(backbone=self.backbone, num_classes=4)

    @pytest.mark.large
    def test_segformer_call(self):
        segformer = SegFormerImageSegmenter(
            backbone=self.backbone, num_classes=4
        )

        images = np.random.uniform(size=(2, 224, 224, 4))
        segformer_output = segformer(images)
        segformer_predict = segformer.predict(images)

        assert segformer_output.shape == images.shape
        assert segformer_predict.shape == images.shape

    def test_task(self):
        self.run_task_test(
            cls=SegFormerImageSegmenter,
            init_kwargs={**self.init_kwargs},
            train_data=self.input_data,
            expected_output_shape=(2, 224, 224),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SegFormerImageSegmenter,
            init_kwargs={**self.init_kwargs},
            input_data=self.input_data,
        )
