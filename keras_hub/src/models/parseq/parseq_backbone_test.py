from keras import ops

from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.tests.test_case import TestCase


class PARSeqTest(TestCase):
    def setUp(self):
        self.images = ops.ones((2, 32, 128, 3))
        self.init_kwargs = {
            "alphabet_size": 5,
            "max_text_length": 5,
        }

    def test_backbone_basics(self):
        # output shape should be
        #  `(batch_size, max_text_length+1, alphabet_size-2)`
        expected_output_shape = (2, 6, 3)
        self.run_backbone_test(
            cls=PARSeqBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
            expected_output_shape=expected_output_shape,
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )
