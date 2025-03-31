import numpy as np
import pytest

from keras_hub.api.models import MiTBackbone
from keras_hub.api.models import SegFormerBackbone
from keras_hub.api.models import SegFormerImageSegmenter
from keras_hub.api.models import SegFormerImageSegmenterPreprocessor
from keras_hub.src.models.mit.mit_backbone import MiTBackbone
from keras_hub.src.models.segformer.segformer_backbone import SegFormerBackbone
from keras_hub.src.models.segformer.segformer_image_segmenter import (
    SegFormerImageSegmenter,
)
from keras_hub.src.tests.test_case import TestCase


class SegFormerTest(TestCase):
    def setUp(self):
        image_encoder = MiTBackbone(
            layerwise_depths=[2, 2],
            image_shape=(224, 224, 3),
            hidden_dims=[32, 64],
            num_layers=2,
            layerwise_num_heads=[1, 2],
            layerwise_sr_ratios=[8, 4],
            max_drop_path_rate=0.1,
            layerwise_patch_sizes=[7, 3],
            layerwise_strides=[4, 2],
        )
        projection_filters = 256
        self.preprocessor = SegFormerImageSegmenterPreprocessor()
        self.backbone = SegFormerBackbone(
            image_encoder=image_encoder, projection_filters=projection_filters
        )

        self.input_size = 224
        self.input_data = np.ones((2, self.input_size, self.input_size, 3))
        self.label_data = np.ones((2, self.input_size, self.input_size, 4))

        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 4,
            "preprocessor": self.preprocessor,
        }

    def test_segformer_segmenter_construction(self):
        SegFormerImageSegmenter(backbone=self.backbone, num_classes=4)

    @pytest.mark.large
    def test_segformer_call(self):
        segformer = SegFormerImageSegmenter(
            backbone=self.backbone, num_classes=4
        )

        images = np.random.uniform(size=(2, 224, 224, 3))
        segformer_output = segformer(images)
        segformer_predict = segformer.predict(images)

        self.assertAllEqual(segformer_output.shape, (2, 224, 224, 4))
        self.assertAllEqual(segformer_predict.shape, (2, 224, 224, 4))

    def test_task(self):
        self.run_task_test(
            cls=SegFormerImageSegmenter,
            init_kwargs={**self.init_kwargs},
            train_data=(self.input_data, self.label_data),
            expected_output_shape=(2, 224, 224, 4),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SegFormerImageSegmenter,
            init_kwargs={**self.init_kwargs},
            input_data=self.input_data,
        )
