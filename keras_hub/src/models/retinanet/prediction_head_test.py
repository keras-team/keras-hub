from keras import random

from keras_hub.src.models.retinanet.prediction_head import PredictionHead
from keras_hub.src.tests.test_case import TestCase


class FeaturePyramidTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=PredictionHead,
            init_kwargs={
                "output_filters": 9 * 4,  # anchors_per_location * box length(4)
            },
            input_data=random.uniform(shape=(2, 64, 64, 256)),
            expected_output_shape=(2, 64, 64, 36),
            expected_num_trainable_weights=10,
        )
