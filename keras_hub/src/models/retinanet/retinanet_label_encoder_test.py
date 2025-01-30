import keras
import numpy as np
import pytest
from keras import ops
from packaging import version

from keras_hub.src.layers.modeling.anchor_generator import AnchorGenerator
from keras_hub.src.models.retinanet.retinanet_label_encoder import (
    RetinaNetLabelEncoder,
)
from keras_hub.src.tests.test_case import TestCase


@pytest.mark.skipif(
    version.parse(keras.__version__) < version.parse("3.8.0"),
    reason="Bbox utils are not supported before keras < 3.8.0",
)
class RetinaNetLabelEncoderTest(TestCase):
    def setUp(self):
        anchor_generator = AnchorGenerator(
            bounding_box_format="xyxy",
            min_level=3,
            max_level=7,
            num_scales=3,
            aspect_ratios=[0.5, 1.0, 2.0],
            anchor_size=8,
        )
        self.init_kwargs = {
            "anchor_generator": anchor_generator,
            "bounding_box_format": "xyxy",
        }

    def test_layer_behaviors(self):
        images_shape = (8, 128, 128, 3)
        boxes_shape = (8, 10, 4)
        classes_shape = (8, 10)
        self.run_layer_test(
            cls=RetinaNetLabelEncoder,
            init_kwargs=self.init_kwargs,
            input_data={
                "images": np.random.uniform(size=images_shape),
                "gt_boxes": np.random.uniform(
                    size=boxes_shape, low=0.0, high=1.0
                ),
                "gt_classes": np.random.uniform(
                    size=classes_shape, low=0, high=5
                ),
            },
            expected_output_shape=((8, 3069, 4), (8, 3069)),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            run_training_check=False,
            run_precision_checks=False,
        )

    def test_label_encoder_output_shapes(self):
        images_shape = (8, 128, 128, 3)
        boxes_shape = (8, 10, 4)
        classes_shape = (8, 10)

        images = np.random.uniform(size=images_shape)
        boxes = np.random.uniform(size=boxes_shape, low=0.0, high=1.0)
        classes = np.random.uniform(size=classes_shape, low=0, high=5)

        encoder = RetinaNetLabelEncoder(
            **self.init_kwargs,
        )

        box_targets, class_targets = encoder(images, boxes, classes)

        self.assertEqual(box_targets.shape, (8, 3069, 4))
        self.assertEqual(class_targets.shape, (8, 3069))

    def test_all_negative_1(self):
        images_shape = (8, 128, 128, 3)
        boxes_shape = (8, 10, 4)
        classes_shape = (8, 10)

        images = np.random.uniform(size=images_shape)
        boxes = -np.ones(shape=boxes_shape, dtype="float32")
        classes = -np.ones(shape=classes_shape, dtype="float32")

        encoder = RetinaNetLabelEncoder(
            **self.init_kwargs,
        )

        box_targets, class_targets = encoder(images, boxes, classes)

        self.assertFalse(ops.any(ops.isnan(box_targets)))
        self.assertFalse(ops.any(ops.isnan(class_targets)))
