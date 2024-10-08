import pytest
from keras import ops
from keras import random

from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.models.retinanet.anchor_generator import AnchorGenerator
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone
from keras_hub.src.models.retinanet.retinanet_label_encoder import (
    RetinaNetLabelEncoder,
)
from keras_hub.src.models.retinanet.retinanet_object_detector import (
    RetinaNetObjectDetector,
)
from keras_hub.src.models.retinanet.retinanet_object_detector_preprocessor import (
    RetinaNetObjectDetectorPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class RetinaNetObjectDetectorTest(TestCase):
    def setUp(self):
        resnet_kwargs = {
            "input_conv_filters": [64],
            "input_conv_kernel_sizes": [7],
            "stackwise_num_filters": [64, 64, 64],
            "stackwise_num_blocks": [2, 2, 2],
            "stackwise_num_strides": [1, 2, 2],
            "image_shape": (None, None, 3),
            "block_type": "bottleneck_block",
            "use_pre_activation": False,
        }
        image_encoder = ResNetBackbone(**resnet_kwargs)

        retinanet_backbone_kwargs = {
            "image_encoder": image_encoder,
            "min_level": 3,
            "max_level": 4,
        }

        feature_extractor = RetinaNetBackbone(**retinanet_backbone_kwargs)
        anchor_generator = AnchorGenerator(
            bounding_box_format="yxyx",
            min_level=3,
            max_level=4,
            num_scales=3,
            aspect_ratios=[0.5, 1.0, 2.0],
            anchor_size=8,
        )
        label_encoder = RetinaNetLabelEncoder(
            bounding_box_format="yxyx", anchor_generator=anchor_generator
        )

        image_converter = ImageConverter(
            image_size=(256, 256),
        )

        preprocessor = RetinaNetObjectDetectorPreprocessor(
            image_converter=image_converter, target_bounding_box_format="xyxy"
        )

        self.init_kwargs = {
            "backbone": feature_extractor,
            "anchor_generator": anchor_generator,
            "label_encoder": label_encoder,
            "num_classes": 10,
            "bounding_box_format": "yxyx",
            "preprocessor": preprocessor,
        }

        self.input_size = 512
        self.images = random.uniform((1, self.input_size, self.input_size, 3))
        self.labels = {
            "boxes": ops.convert_to_numpy(
                [[[20, 10, 120, 110], [30, 20, 130, 120]]]
            ),
            "classes": ops.convert_to_numpy([[0, 2]]),
        }

        self.train_data = (self.images, self.labels)

    @pytest.mark.large
    def test_detection_basics(self):
        self.run_task_test(
            cls=RetinaNetObjectDetector,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape={
                "boxes": (1, 100, 4),
                "classes": (1, 100),
                "confidence": (1, 100),
                "num_detections": (1,),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=RetinaNetObjectDetector,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
