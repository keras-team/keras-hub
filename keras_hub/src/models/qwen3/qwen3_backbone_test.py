import pytest
from keras import ops

from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.tests.test_case import TestCase


class Qwen3Test(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "hidden_dim": 8,
            "intermediate_dim": 8,
            "use_sliding_window_attention": False,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = Qwen3Backbone(**self.init_kwargs)
        self.assertEqual(model.count_params(), 896)
