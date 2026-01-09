import numpy as np

from keras_hub.src.models.llama3.llama3_vision_projector import (
    Llama3VisionProjector,
)
from keras_hub.src.tests.test_case import TestCase


class Llama3VisionProjectorTest(TestCase):
    def test_projector_basics(self):
        self.run_layer_test(
            cls=Llama3VisionProjector,
            init_kwargs={
                "hidden_dim": 128,  # Vision Encoder output
                "output_dim": 256,  # Text Model input
                "intermediate_dim": 512,  # Internal MLP size
            },
            input_data=np.random.uniform(size=(2, 10, 128)).astype("float32"),
            expected_output_shape=(2, 10, 256),  # Should match output_dim
            # dense_1 (kernel, bias) + dense_2 (kernel, bias)
            expected_num_trainable_weights=4,
            run_precision_checks=False,
        )

    def test_defaults(self):
        # Test that intermediate_dim defaults to output_dim if not set
        projector = Llama3VisionProjector(hidden_dim=32, output_dim=64)
        images = np.random.uniform(size=(2, 5, 32)).astype("float32")
        outputs = projector(images)
        self.assertEqual(outputs.shape, (2, 5, 64))
