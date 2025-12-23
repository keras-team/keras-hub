import keras
import numpy as np
import pytest
from packaging import version

from keras_hub.src.layers.modeling.anchor_generator import AnchorGenerator
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone
from keras_hub.src.models.retinanet.retinanet_image_converter import (
    RetinaNetImageConverter,
)
from keras_hub.src.models.retinanet.retinanet_label_encoder import (
    RetinaNetLabelEncoder,
)
from keras_hub.src.models.retinanet.retinanet_object_detector import (
    RetinaNetObjectDetector,
)
from keras_hub.src.models.retinanet.retinanet_object_detector_preprocessor import (  # noqa: E501
    RetinaNetObjectDetectorPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


@pytest.mark.skipif(
    version.parse(keras.__version__) < version.parse("3.8.0"),
    reason="Bbox utils are not supported before keras < 3.8.0",
)
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
            "use_p5": False,
        }

        feature_extractor = RetinaNetBackbone(**retinanet_backbone_kwargs)
        anchor_generator = AnchorGenerator(
            bounding_box_format="yxyx",
            min_level=3,
            max_level=4,
            num_scales=3,
            aspect_ratios=[0.5, 1.0, 2.0],
            anchor_size=4,
        )
        label_encoder = RetinaNetLabelEncoder(
            bounding_box_format="yxyx", anchor_generator=anchor_generator
        )

        image_converter = RetinaNetImageConverter(
            bounding_box_format="yxyx", scale=1 / 255.0, image_size=(512, 512)
        )

        preprocessor = RetinaNetObjectDetectorPreprocessor(
            image_converter=image_converter
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
        self.run_task_test(
            cls=RetinaNetObjectDetector,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape={
                "boxes": (1, 100, 4),
                "labels": (1, 100),
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

    def test_litert_export(self):
        input_data = self.images

        self.run_litert_export_test(
            cls=RetinaNetObjectDetector,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
            comparison_mode="statistical",
            output_thresholds={
                "enc_topk_logits": {"max": 5.0, "mean": 0.05},
                "logits": {"max": 2.0, "mean": 0.05},
                "*": {"max": 1.5, "mean": 0.05},
            },
        )
