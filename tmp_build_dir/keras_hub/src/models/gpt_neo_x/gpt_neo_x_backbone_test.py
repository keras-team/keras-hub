import pytest
from keras import ops

from keras_hub.src.models.gpt_neo_x.gpt_neo_x_backbone import GPTNeoXBackbone
from keras_hub.src.tests.test_case import TestCase


class GPTNeoXBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_sequence_length": 5,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=GPTNeoXBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=GPTNeoXBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
