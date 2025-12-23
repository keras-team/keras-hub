import keras
import numpy as np
import pytest
from absl.testing import parameterized
from packaging import version

from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
from keras_hub.src.models.d_fine.d_fine_image_converter import (
    DFineImageConverter,
)
from keras_hub.src.models.d_fine.d_fine_object_detector import (
    DFineObjectDetector,
)
from keras_hub.src.models.d_fine.d_fine_object_detector_preprocessor import (
    DFineObjectDetectorPreprocessor,
)
from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
from keras_hub.src.tests.test_case import TestCase


@pytest.mark.skipif(
    version.parse(keras.__version__) < version.parse("3.8.0"),
    reason="Bbox utils are not supported before Keras < 3.8.0",
)
class DFineObjectDetectorTest(TestCase):
    def setUp(self):
        self.labels = [
            {
                "boxes": np.array([[0.5, 0.5, 0.2, 0.2]]),
                "labels": np.array([1]),
            },
            {
                "boxes": np.array([[0.6, 0.6, 0.3, 0.3]]),
                "labels": np.array([2]),
            },
        ]
        self.stackwise_stage_filters = [
            [8, 8, 16, 1, 1, 3],
            [16, 8, 32, 1, 1, 3],
        ]
        self.apply_downsample = [False, True]
        self.use_lightweight_conv_block = [False, False]
        self.input_size = 32
        self.bounding_box_format = "yxyx"

        image_converter = DFineImageConverter(
            bounding_box_format=self.bounding_box_format,
            image_size=(self.input_size, self.input_size),
        )
        preprocessor = DFineObjectDetectorPreprocessor(
            image_converter=image_converter,
        )
        self.preprocessor = preprocessor
        self.images = np.random.uniform(
            low=0, high=255, size=(1, self.input_size, self.input_size, 3)
        ).astype("float32")
        self.bounding_boxes = {
            "boxes": np.array([[[10.0, 10.0, 20.0, 20.0]]]),
            "labels": np.array([[0]]),
        }
        self.train_data = (
            self.images,
            self.bounding_boxes,
        )
        hgnetv2_backbone = HGNetV2Backbone(
            stem_channels=[3, 8, 8],
            stackwise_stage_filters=self.stackwise_stage_filters,
            apply_downsample=self.apply_downsample,
            use_lightweight_conv_block=self.use_lightweight_conv_block,
            depths=[1, 1],
            hidden_sizes=[16, 32],
            embedding_size=8,
            use_learnable_affine_block=True,
            hidden_act="relu",
            image_shape=(None, None, 3),
            out_features=["stage1", "stage2"],
            data_format="channels_last",
        )
        self.base_backbone_kwargs = {
            "backbone": hgnetv2_backbone,
            "decoder_in_channels": [16, 16],
            "encoder_hidden_dim": 16,
            "num_denoising": 10,
            "num_labels": 4,
            "hidden_dim": 16,
            "learn_initial_query": False,
            "num_queries": 10,
            "anchor_image_size": (self.input_size, self.input_size),
            "feat_strides": [4, 8],
            "num_feature_levels": 2,
            "encoder_in_channels": [16, 32],
            "encode_proj_layers": [1],
            "num_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "num_encoder_layers": 1,
            "hidden_expansion": 0.5,
            "depth_multiplier": 0.5,
            "eval_idx": -1,
            "num_decoder_layers": 1,
            "decoder_attention_heads": 2,
            "decoder_ffn_dim": 32,
            "decoder_method": "default",
            "decoder_n_points": [2, 2],
            "lqe_hidden_dim": 16,
            "num_lqe_layers": 1,
            "out_features": ["stage1", "stage2"],
            "image_shape": (None, None, 3),
            "data_format": "channels_last",
            "seed": 0,
        }

    @parameterized.named_parameters(
        ("default", False),
        ("denoising", True),
    )
    def test_detection_basics(self, use_noise_and_labels):
        backbone_kwargs = self.base_backbone_kwargs.copy()
        if use_noise_and_labels:
            backbone_kwargs["box_noise_scale"] = 1.0
            backbone_kwargs["label_noise_ratio"] = 0.5
            backbone_kwargs["labels"] = self.labels
        backbone = DFineBackbone(**backbone_kwargs)
        init_kwargs = {
            "backbone": backbone,
            "num_classes": 4,
            "bounding_box_format": self.bounding_box_format,
            "preprocessor": self.preprocessor,
        }
        self.run_task_test(
            cls=DFineObjectDetector,
            init_kwargs=init_kwargs,
            train_data=self.train_data,
            expected_output_shape={
                "boxes": (1, 10, 4),
                "labels": (1, 10),
                "confidence": (1, 10),
                "num_detections": (1,),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        backbone = DFineBackbone(**self.base_backbone_kwargs)
        init_kwargs = {
            "backbone": backbone,
            "num_classes": 4,
            "bounding_box_format": self.bounding_box_format,
            "preprocessor": self.preprocessor,
        }
        self.run_model_saving_test(
            cls=DFineObjectDetector,
            init_kwargs=init_kwargs,
            input_data=self.images,
        )

    def test_litert_export(self):
        backbone = DFineBackbone(**self.base_backbone_kwargs)
        init_kwargs = {
            "backbone": backbone,
            "num_classes": 4,
            "bounding_box_format": self.bounding_box_format,
            "preprocessor": self.preprocessor,
        }

        # D-Fine ObjectDetector only takes images as input
        input_data = self.images

        self.run_litert_export_test(
            cls=DFineObjectDetector,
            init_kwargs=init_kwargs,
            input_data=input_data,
            comparison_mode="statistical",
            output_thresholds={
                "intermediate_predicted_corners": {"max": 5.0, "mean": 0.05},
                "intermediate_logits": {"max": 5.0, "mean": 0.1},
                "enc_topk_logits": {"max": 5.0, "mean": 0.03},
                "logits": {"max": 2.0, "mean": 0.03},
                "*": {"max": 1.0, "mean": 0.03},
            },
        )
