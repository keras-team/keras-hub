from absl.testing import parameterized
from keras import ops
from keras import random

from keras_hub.src.models.retinanet.feature_pyramid import FeaturePyramid
from keras_hub.src.tests.test_case import TestCase


class FeaturePyramidTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=FeaturePyramid,
            init_kwargs={
                "min_level": 3,
                "max_level": 7,
                "activation": "relu",
                "batch_norm_momentum": 0.99,
                "batch_norm_epsilon": 0.0001,
                "kernel_initializer": "HeNormal",
                "bias_initializer": "Zeros",
            },
            input_data={
                "P3": random.uniform(shape=(2, 64, 64, 4)),
                "P4": random.uniform(shape=(2, 32, 32, 8)),
                "P5": random.uniform(shape=(2, 16, 16, 16)),
            },
            expected_output_shape={
                "P3": (2, 64, 64, 256),
                "P4": (2, 32, 32, 256),
                "P5": (2, 16, 16, 256),
                "P6": (2, 8, 8, 256),
                "P7": (2, 4, 4, 256),
            },
            expected_num_trainable_weights=16,
            expected_num_non_trainable_weights=0,
        )

    @parameterized.named_parameters(
        (
            "equal_resolutions",
            3,
            7,
            {"P3": (2, 16, 16, 3), "P4": (2, 8, 8, 3), "P5": (2, 4, 4, 3)},
        ),
        (
            "different_resolutions",
            2,
            6,
            {
                "P2": (2, 64, 128, 4),
                "P3": (2, 32, 64, 8),
                "P4": (2, 16, 32, 16),
                "P5": (2, 8, 16, 32),
            },
        ),
    )
    def test_layer_output_shapes(self, min_level, max_level, input_shapes):
        layer = FeaturePyramid(min_level=min_level, max_level=max_level)

        inputs = {
            level: ops.ones(input_shapes[level]) for level in input_shapes
        }
        if layer.data_format == "channels_first":
            inputs = {
                level: ops.transpose(inputs[level], (0, 3, 1, 2))
                for level in inputs
            }

        output = layer(inputs)

        for level in inputs:
            self.assertEqual(
                output[level].shape,
                (
                    (input_shapes[level][0],)
                    + (layer.num_filters,)
                    + input_shapes[level][1:3]
                    if layer.data_format == "channels_first"
                    else input_shapes[level][:-1] + (layer.num_filters,)
                ),
            )
