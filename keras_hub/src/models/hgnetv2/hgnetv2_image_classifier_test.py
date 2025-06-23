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
        self.height = 64
        self.width = 64
        self.num_channels = 3
        self.image_input_shape = (self.height, self.width, self.num_channels)
        self.num_classes = 3
        self.images = np.ones(
            (self.batch_size, *self.image_input_shape), dtype="float32"
        )
        self.labels = np.random.randint(0, self.num_classes, self.batch_size)
        num_stages = 2
        # Setup model.
        stem_channels = [self.num_channels, 16, 32]
        stage_in_channels = [stem_channels[-1], 64][:num_stages]
        stage_mid_channels = [16, 32][:num_stages]
        stage_out_channels = [64, 128][:num_stages]
        stage_num_blocks = [1] * num_stages
        stage_numb_of_layers = [1] * num_stages
        stage_downsample = [False, True][:num_stages]
        stage_light_block = [False, False][:num_stages]
        stage_kernel_size = [3] * num_stages

        self.backbone = HGNetV2Backbone(
            initializer_range=0.02,
            depths=stage_num_blocks,
            embedding_size=stem_channels[-1],
            hidden_sizes=stage_out_channels,
            stem_channels=stem_channels,
            hidden_act="relu",
            use_learnable_affine_block=False,
            num_channels=self.num_channels,
            stage_in_channels=stage_in_channels,
            stage_mid_channels=stage_mid_channels,
            stage_out_channels=stage_out_channels,
            stage_num_blocks=stage_num_blocks,
            stage_numb_of_layers=stage_numb_of_layers,
            stage_downsample=stage_downsample,
            stage_light_block=stage_light_block,
            stage_kernel_size=stage_kernel_size,
            image_shape=self.image_input_shape,
        )
        self.image_converter = HGNetV2ImageConverter(
            image_size=(self.height, self.width),
            crop_pct=0.875,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            interpolation="bilinear",
        )
        self.preprocessor = HGNetV2ImageClassifierPreprocessor(
            image_converter=self.image_converter
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
            "num_classes": self.num_classes,
            "head_filters": stage_out_channels[-1],
        }
        self.train_data = (
            self.images,
            self.labels,
        )
        self.expected_backbone_output_shapes = {
            "stage0": (self.batch_size, 16, 16, 64),
            "stage1": (self.batch_size, 8, 8, 128),
        }
        self.preset_image_size = 224
        self.images_for_presets = np.ones(
            (
                self.batch_size,
                self.preset_image_size,
                self.preset_image_size,
                self.num_channels,
            ),
            dtype="float32",
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
