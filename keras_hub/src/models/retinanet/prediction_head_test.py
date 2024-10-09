from absl.testing import parameterized
from keras import random

from keras_hub.src.models.retinanet.prediction_head import PredictionHead
from keras_hub.src.tests.test_case import TestCase


class FeaturePyramidTest(TestCase):
    @parameterized.named_parameters(
        ("without_group_normalization", False, 10),
        ("with_group_normalization", True, 14),
    )
    def test_layer_behaviors(
        self, use_group_norm, expected_num_trainable_weights
    ):
        self.run_layer_test(
            cls=PredictionHead,
            init_kwargs={
                "output_filters": 9 * 4,  # anchors_per_location * box length(4)
                "num_filters": 256,
                "num_conv_layers": 4,
                "use_group_norm": use_group_norm,
            },
            input_data=random.uniform(shape=(2, 64, 64, 256)),
            expected_output_shape=(2, 64, 64, 36),
            expected_num_trainable_weights=expected_num_trainable_weights,
        )
