import keras
import pytest
from keras import ops

from keras_hub.src.models.t5.t5_backbone import T5Backbone
from keras_hub.src.models.t5.t5_transformer_layer import T5TransformerLayer
from keras_hub.src.tests.test_case import TestCase


class T5BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
        }
        self.input_data = {
            "encoder_token_ids": ops.ones((2, 3), dtype="int32"),
            "encoder_padding_mask": ops.zeros((2, 3), dtype="int32"),
            "decoder_token_ids": ops.ones((2, 3), dtype="int32"),
            "decoder_padding_mask": ops.zeros((2, 3), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=T5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "encoder_sequence_output": (2, 3, 2),
                "decoder_sequence_output": (2, 3, 2),
            },
        )

    def test_transformer_layer_output_spec_matches_call(self):
        layer = T5TransformerLayer(
            is_decoder=False,
            hidden_dim=2,
            intermediate_dim=4,
            key_value_dim=1,
            dropout=0.0,
            activation="relu",
            layer_norm_epsilon=1e-6,
            num_heads=2,
            use_gated_activation=True,
            use_relative_attention_bias=True,
        )
        hidden_states = ops.ones((2, 3, 2))
        call_output = layer(hidden_states, position_bias=None)
        spec_output = layer.compute_output_spec(
            keras.KerasTensor(shape=(2, 3, 2)), position_bias=None
        )
        self.assertIsInstance(call_output, tuple)
        self.assertIsInstance(spec_output, tuple)
        self.assertLen(call_output, len(spec_output))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=T5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=T5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=T5Backbone,
            preset="t5_small_multi",
            input_data=self.input_data,
            expected_output_shape={
                "encoder_sequence_output": (2, 3, 512),
                "decoder_sequence_output": (2, 3, 512),
            },
            expected_partial_output={
                "encoder_sequence_output": ops.array(
                    [-0.0034, 0.0293, -0.0827, -0.1076]
                ),
                "decoder_sequence_output": ops.array(
                    [0.0097, 0.3576, -0.1508, 0.0150]
                ),
            },
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in T5Backbone.presets:
            self.run_preset_test(
                cls=T5Backbone,
                preset=preset,
                input_data=self.input_data,
            )
