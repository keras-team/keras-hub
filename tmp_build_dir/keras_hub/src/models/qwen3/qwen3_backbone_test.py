import pytest
from keras import ops

from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.models.qwen3.qwen3_decoder import Qwen3TransformerDecoder
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

    def test_attention_norm_uses_backbone_epsilon(self):
        model = Qwen3Backbone(
            **self.init_kwargs,
            layer_norm_epsilon=2e-6,
        )
        attention = model.transformer_layers[0]._self_attention_layer

        self.assertEqual(attention._query_dense_layer_norm.epsilon, 2e-6)
        self.assertEqual(attention._key_dense_layer_norm.epsilon, 2e-6)

    def test_decoder_config_round_trip(self):
        decoder = Qwen3TransformerDecoder(
            intermediate_dim=16,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=6,
            sliding_window_size=128,
        )

        restored = Qwen3TransformerDecoder.from_config(decoder.get_config())

        self.assertEqual(restored.head_dim, 6)
        self.assertEqual(restored.sliding_window_size, 128)
