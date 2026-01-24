import numpy as np
import pytest

from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
from keras_hub.src.models.hgnetv2.hgnetv2_image_classifier import (
    HGNetV2ImageClassifier,
)
from keras_hub.src.models.hgnetv2.hgnetv2_image_classifier_preprocessor import (
    HGNetV2ImageClassifierPreprocessor,
)
from keras_hub.src.models.hgnetv2.hgnetv2_image_converter import (
    HGNetV2ImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class HGNetV2ImageClassifierTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_shape = (64, 64, 3)
        self.num_classes = 3
        self.images = np.ones(
            (self.batch_size, *self.image_shape), dtype="float32"
        )
        self.labels = np.random.randint(0, self.num_classes, self.batch_size)
        self.train_data = (self.images, self.labels)
        # Setup model.
        self.backbone = HGNetV2Backbone(
            depths=[1, 1],
            embedding_size=32,
            hidden_sizes=[64, 128],
            stem_channels=[self.image_shape[-1], 16, 32],
            hidden_act="relu",
            use_learnable_affine_block=False,
            stackwise_stage_filters=[
                [32, 16, 64, 1, 1, 3],
                [64, 32, 128, 1, 1, 3],
            ],
            apply_downsample=[False, True],
            use_lightweight_conv_block=[False, False],
            image_shape=self.image_shape,
        )
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        scale = [1 / (255.0 * s) for s in std]
        offset = [-m / s for m, s in zip(mean, std)]
        self.image_converter = HGNetV2ImageConverter(
            image_size=self.image_shape[:2],
            scale=scale,
            offset=offset,
            interpolation="bilinear",
            antialias=False,
        )
        self.preprocessor = HGNetV2ImageClassifierPreprocessor(
            image_converter=self.image_converter
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
            "num_classes": self.num_classes,
        }
        self.preset_image_shape = (224, 224, 3)
        self.images_for_presets = np.ones(
            (self.batch_size, *self.preset_image_shape), dtype="float32"
        )

    def test_classifier_basics(self):
        self.run_task_test(
            cls=HGNetV2ImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(self.batch_size, self.num_classes),
        )

    @pytest.mark.large
    def test_all_presets(self):
        for preset in HGNetV2ImageClassifier.presets:
            self.run_preset_test(
                cls=HGNetV2ImageClassifier,
                preset=preset,
                input_data=self.images_for_presets,
                expected_output_shape=(self.batch_size, 1000),
            )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=HGNetV2ImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=HGNetV2ImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
