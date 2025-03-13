import keras
import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops
from packaging import version

from keras_hub.src.layers.modeling.anchor_generator import AnchorGenerator
from keras_hub.src.tests.test_case import TestCase


@pytest.mark.skipif(
    version.parse(keras.__version__) < version.parse("3.8.0"),
    reason="Bbox utils are not supported before keras < 3.8.0",
)
class AnchorGeneratorTest(TestCase):
    def test_layer_behaviors(self):
        images_shape = (8, 128, 128, 3)
        self.run_layer_test(
            cls=AnchorGenerator,
            init_kwargs={
                "bounding_box_format": "xyxy",
                "min_level": 3,
                "max_level": 7,
                "num_scales": 3,
                "aspect_ratios": [0.5, 1.0, 2.0],
                "anchor_size": 4,
            },
            input_data=np.random.uniform(size=images_shape),
            expected_output_shape={
                "P3": (2304, 4),
                "P4": (576, 4),
                "P5": (144, 4),
                "P6": (36, 4),
                "P7": (9, 4),
            },
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            run_training_check=False,
            run_precision_checks=False,
        )

    @parameterized.parameters(
        # Single scale anchor
        ("yxyx", 5, 5, 1, [1.0], 2.0, [64, 64])
        + (
            {
                "P5": [
                    [-32.0, -32.0, 32.0, 32.0],
                    [-32.0, 0, 32.0, 64.0],
                    [0.0, -32.0, 64.0, 32.0],
                    [0.0, 0.0, 64.0, 64.0],
                ]
            },
        ),
    )
    def test_anchor_generator(
        self,
        bounding_box_format,
        min_level,
        max_level,
        num_scales,
        aspect_ratios,
        anchor_size,
        image_shape,
        expected_boxes,
    ):
        anchor_generator = AnchorGenerator(
            bounding_box_format,
            min_level,
            max_level,
            num_scales,
            aspect_ratios,
            anchor_size,
        )
        images = ops.ones(shape=(1, image_shape[0], image_shape[1], 3))
        multilevel_boxes = anchor_generator(images)
        for key in expected_boxes:
            expected_boxes[key] = ops.convert_to_tensor(expected_boxes[key])
        self.assertAllClose(expected_boxes, multilevel_boxes)
