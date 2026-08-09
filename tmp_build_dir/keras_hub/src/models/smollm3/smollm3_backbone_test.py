import pytest
from keras import ops

from keras_hub.src.models.smollm3.smollm3_backbone import SmolLM3Backbone
from keras_hub.src.tests.test_case import TestCase


class SmolLM3BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 100,
            "hidden_dim": 64,
            "intermediate_dim": 128,
            "num_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rope_layer_enabled_list": [True, True],
            "layer_types": ["attention", "attention"],
            "mlp_bias": False,
            "layer_norm_epsilon": 1e-5,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=SmolLM3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 64),
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SmolLM3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = SmolLM3Backbone(**self.init_kwargs)
        # Reference value calculated from the model architecture
        self.assertEqual(model.count_params(), 80464)
