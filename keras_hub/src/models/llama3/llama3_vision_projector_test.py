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
                "input_dim": 7680,  # vision_output_dim
                "output_dim": 4096,  # text hidden_dim
            },
            input_data=np.random.uniform(size=(2, 10, 7680)).astype("float32"),
            expected_output_shape=(2, 10, 4096),
            # projection layer (kernel, bias)
            expected_num_trainable_weights=2,
            run_precision_checks=False,
        )

    def test_projector_with_defaults(self):
        # Test with minimal arguments.
        projector = Llama3VisionProjector(input_dim=1280, output_dim=512)
        images = np.random.uniform(size=(2, 5, 1280)).astype("float32")
        outputs = projector(images)
        self.assertEqual(outputs.shape, (2, 5, 512))
