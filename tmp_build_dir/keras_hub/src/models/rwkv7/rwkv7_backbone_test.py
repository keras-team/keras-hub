import pytest
from keras import ops

from keras_hub.src.models.rwkv7.rwkv7_backbone import RWKV7Backbone
from keras_hub.src.tests.test_case import TestCase


class RWKV7BackboneTest(TestCase):
    def setUp(self):
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
        test_input = ops.ones((2, 16), dtype="int32")
        self.input_data = {"token_ids": test_input, "padding_mask": test_input}

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=RWKV7Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 16, 10),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=RWKV7Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
