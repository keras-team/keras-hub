import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.layers.modeling.non_max_supression import NonMaxSuppression
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


class DFineObjectDetectorTest(TestCase):
    def setUp(self):
        self.labels = [
            {
                "boxes": np.array([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]]),
                "labels": np.array([1, 10]),
            },
            {
                "boxes": np.array([[0.6, 0.6, 0.3, 0.3]]),
                "labels": np.array([20]),
            },
        ]
        self.stackwise_stage_filters = [
            [16, 16, 64, 1, 3, 3],
            [64, 32, 256, 1, 3, 3],
            [256, 64, 512, 2, 3, 5],
            [512, 128, 1024, 1, 3, 5],
        ]
        self.apply_downsample = [False, True, True, True]
        self.use_lightweight_conv_block = [False, False, True, True]
        self.input_size = 256
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
            "boxes": np.array(
                [[[10.0, 20.0, 20.0, 30.0], [20.0, 30.0, 30.0, 40.0]]]
            ),
            "labels": np.array([[0, 2]]),
        }
        self.train_data = (
            self.images,
            self.bounding_boxes,
        )
        hgnetv2_backbone = HGNetV2Backbone(
            stem_channels=[3, 16, 16],
            stackwise_stage_filters=self.stackwise_stage_filters,
            apply_downsample=self.apply_downsample,
            use_lightweight_conv_block=self.use_lightweight_conv_block,
            depths=[1, 1, 2, 1],
            hidden_sizes=[64, 256, 512, 1024],
            embedding_size=16,
            use_learnable_affine_block=True,
            hidden_act="relu",
            image_shape=(None, None, 3),
            out_features=["stage3", "stage4"],
            data_format="channels_last",
        )
        self.base_backbone_kwargs = {
            "backbone": hgnetv2_backbone,
            "decoder_in_channels": [128, 128],
            "encoder_hidden_dim": 128,
            "num_denoising": 100,
            "num_labels": 80,
            "hidden_dim": 128,
            "learn_initial_query": False,
            "num_queries": 300,
            "anchor_image_size": (256, 256),
            "feat_strides": [16, 32],
            "num_feature_levels": 2,
            "encoder_in_channels": [512, 1024],
            "encode_proj_layers": [1],
            "num_attention_heads": 8,
            "encoder_ffn_dim": 512,
            "num_encoder_layers": 1,
            "hidden_expansion": 0.34,
            "depth_multiplier": 0.5,
            "eval_idx": -1,
            "num_decoder_layers": 3,
            "decoder_attention_heads": 8,
            "decoder_ffn_dim": 512,
            "decoder_method": "default",
            "decoder_n_points": [6, 6],
            "lqe_hidden_dim": 64,
            "num_lqe_layers": 2,
            "out_features": ["stage3", "stage4"],
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
        prediction_decoder = NonMaxSuppression(
            from_logits=True,
            bounding_box_format=self.bounding_box_format,
            max_detections=self.base_backbone_kwargs["num_queries"],
        )
        init_kwargs = {
            "backbone": backbone,
            "num_classes": 80,
            "bounding_box_format": self.bounding_box_format,
            "preprocessor": self.preprocessor,
            "prediction_decoder": prediction_decoder,
        }
        self.run_task_test(
            cls=DFineObjectDetector,
            init_kwargs=init_kwargs,
            train_data=self.train_data,
            expected_output_shape={
                "boxes": (1, 300, 4),
                "labels": (1, 300),
                "confidence": (1, 300),
                "num_detections": (1,),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        backbone = DFineBackbone(**self.base_backbone_kwargs)
        init_kwargs = {
            "backbone": backbone,
            "num_classes": 80,
            "bounding_box_format": self.bounding_box_format,
            "preprocessor": self.preprocessor,
        }
        self.run_model_saving_test(
            cls=DFineObjectDetector,
            init_kwargs=init_kwargs,
            input_data=self.images,
        )
