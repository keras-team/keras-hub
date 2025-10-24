import numpy as np
import pytest

from keras_hub.src.models.detr.detr_backbone import DETRBackbone
from keras_hub.src.models.detr.detr_image_converter import DETRImageConverter
from keras_hub.src.models.detr.detr_object_detector import DETRObjectDetector
from keras_hub.src.models.detr.detr_object_detector_preprocessor import (
    DETRObjectDetectorPreprocessor,
)
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class DETRObjectDetectorTest(TestCase):
    def setUp(self):
        resnet_kwargs = {
            "input_conv_filters": [64],
            "input_conv_kernel_sizes": [7],
            "stackwise_num_filters": [64, 128, 256],
            "stackwise_num_blocks": [2, 2, 2],
            "stackwise_num_strides": [1, 2, 2],
            "image_shape": (None, None, 3),
            "block_type": "bottleneck_block",
            "use_pre_activation": False,
        }
        image_encoder = ResNetBackbone(**resnet_kwargs)

        detr_backbone_kwargs = {
            "image_encoder": image_encoder,
            "hidden_dim": 256,
            "num_encoder_layers": 2,
            "num_heads": 8,
            "intermediate_size": 512,
            "dropout": 0.0,
            "activation": "relu",
            "image_shape": (None, None, 3),
        }

        backbone = DETRBackbone(**detr_backbone_kwargs)

        image_converter = DETRImageConverter(
            image_size=(512, 512),
            scale=1 / 255.0,
        )

        preprocessor = DETRObjectDetectorPreprocessor(
            image_converter=image_converter
        )

        self.init_kwargs = {
            "backbone": backbone,
            "num_queries": 100,
            "num_classes": 10,
            "num_decoder_layers": 2,
            "num_heads": 8,
            "intermediate_size": 512,
            "dropout": 0.1,
            "activation": "relu",
            "preprocessor": preprocessor,
        }

        self.input_size = 512
        self.images = np.random.uniform(
            low=0, high=255, size=(1, self.input_size, self.input_size, 3)
        ).astype("float32")
        self.labels = {
            "boxes": np.array(
                [[[20.0, 10.0, 12.0, 11.0], [30.0, 20.0, 40.0, 12.0]]]
            ),
            "labels": np.array([[0, 2]]),
        }
        self.train_data = (self.images, self.labels)

    def test_detection_basics(self):
        model = DETRObjectDetector(**self.init_kwargs)
        outputs = model(self.images)

        self.assertEqual(outputs["cls_logits"].shape, (1, 100, 11))
        self.assertEqual(outputs["bbox_regression"].shape, (1, 100, 4))

        self.run_serialization_test(model)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DETRObjectDetector,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DETRObjectDetector.presets:
            self.run_preset_test(
                cls=DETRObjectDetector,
                preset=preset,
                input_data=self.images,
            )
