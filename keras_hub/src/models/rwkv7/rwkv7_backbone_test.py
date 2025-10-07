from keras import ops

from keras_hub.src.models.rwkv7.rwkv7_backbone import RWKV7Backbone
from keras_hub.src.tests.test_case import TestCase


class RWKV7BackboneTest(TestCase):
    def setUp(self):
        """
        Set up the test case with default arguments and input data.
        """
        self.init_kwargs = {
            "vocabulary_size": 10,
            "hidden_size": 16,
            "num_layers": 2,
            "head_size": 4,
            "intermediate_dim": 32,
            "gate_lora": 32,
            "mv_lora": 16,
            "aaa_lora": 16,
            "decay_lora": 16,
        }
        self.input_data = ops.ones((2, 5), dtype="int32")
        self.backbone = RWKV7Backbone(**self.init_kwargs)

    def test_backbone_basics(self):
        """
        Test basic functionality of the RWKV7 backbone.
        """
        y = self.backbone(self.input_data)
        self.assertEqual(y.shape, (2, 5, 10))

    def test_num_parameters(self):
        """
        Test that the model has the expected number of parameters.
        """
        self.assertEqual(self.backbone.count_params(), 10208)
