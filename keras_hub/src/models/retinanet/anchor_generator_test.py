from absl.testing import parameterized
from keras import ops

from keras_hub.src.bounding_box.converters import convert_format
from keras_hub.src.models.retinanet.anchor_generator import AnchorGenerator
from keras_hub.src.tests.test_case import TestCase


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
                "anchor_size": 8,
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
                    [-16.0, -16.0, 48.0, 48.0],
                    [-16.0, 16.0, 48.0, 80.0],
                    [16.0, -16.0, 80.0, 48.0],
                    [16.0, 16.0, 80.0, 80.0],
                ]
            },
        ),
        # Multi scale anchor
        ("xywh", 5, 6, 1, [1.0], 2.0, [64, 64])
        + (
            {
                "P5": [
                    [-16.0, -16.0, 48.0, 48.0],
                    [-16.0, 16.0, 48.0, 80.0],
                    [16.0, -16.0, 80.0, 48.0],
                    [16.0, 16.0, 80.0, 80.0],
                ],
                "P6": [[-32, -32, 96, 96]],
            },
        ),
        # Multi aspect ratio anchor
        ("xyxy", 6, 6, 1, [1.0, 4.0, 0.25], 2.0, [64, 64])
        + (
            {
                "P6": [
                    [-32.0, -32.0, 96.0, 96.0],
                    [0.0, -96.0, 64.0, 160.0],
                    [-96.0, 0.0, 160.0, 64.0],
                ]
            },
        ),
        # Intermidate scales
        ("yxyx", 5, 5, 2, [1.0], 1.0, [32, 32])
        + (
            {
                "P5": [
                    [0.0, 0.0, 32.0, 32.0],
                    [
                        16 - 16 * 2**0.5,
                        16 - 16 * 2**0.5,
                        16 + 16 * 2**0.5,
                        16 + 16 * 2**0.5,
                    ],
                ]
            },
        ),
        # Non-square
        ("xywh", 5, 5, 1, [1.0], 1.0, [64, 32])
        + ({"P5": [[0, 0, 32, 32], [32, 0, 64, 32]]},),
        # Indivisible by 2^level
        ("xyxy", 5, 5, 1, [1.0], 1.0, [40, 32])
        + ({"P5": [[-6, 0, 26, 32], [14, 0, 46, 32]]},),
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
            expected_boxes[key] = convert_format(
                expected_boxes[key],
                source="yxyx",
                target=bounding_box_format,
            )
        self.assertAllClose(expected_boxes, multilevel_boxes)
