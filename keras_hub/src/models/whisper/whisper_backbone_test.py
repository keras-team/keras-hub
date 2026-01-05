import pytest
from keras import ops

from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.tests.test_case import TestCase


class WhisperBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_encoder_sequence_length": 6,
            "max_decoder_sequence_length": 6,
        }
        self.input_data = {
            "encoder_features": ops.ones((2, 5, 80), dtype="float32"),
            "decoder_token_ids": ops.ones((2, 5), dtype="int32"),
            "decoder_padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=WhisperBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "encoder_sequence_output": (2, 3, 2),
                "decoder_sequence_output": (2, 5, 2),
            },
        )

    def test_key_projection_bias_absence(self):
        backbone = WhisperBackbone(**self.init_kwargs)
        # Check only for the first encoder layer and first decoder layer.
        self.assertIsNone(
            backbone.get_layer(
                "transformer_encoder_layer_0"
            )._self_attention_layer._key_dense.bias
        )
        self.assertIsNone(
            backbone.get_layer(
                "transformer_decoder_layer_0"
            )._self_attention_layer._key_dense.bias
        )
        self.assertIsNone(
            backbone.get_layer(
                "transformer_decoder_layer_0"
            )._cross_attention_layer._key_dense.bias
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=WhisperBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=WhisperBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=WhisperBackbone,
            preset="whisper_tiny_en",
            input_data={
                "encoder_features": ops.ones((1, 3000, 80)),
                "decoder_token_ids": ops.array(
                    [[50257, 50362, 464, 2068, 7586, 21831, 13, 50256, 50256]]
                ),
                "decoder_padding_mask": ops.array(
                    [[1, 1, 1, 1, 1, 1, 1, 1, 0]]
                ),
            },
            expected_output_shape={
                "encoder_sequence_output": (1, 1500, 384),
                "decoder_sequence_output": (1, 9, 384),
            },
            # The forward pass from a preset should be stable!
            expected_partial_output={
                "encoder_sequence_output": ops.array(
                    [-0.21382, -0.48528, 0.42348, -1.33874, -0.14191]
                ),
                "decoder_sequence_output": ops.array(
                    [13.238, 1.051, 8.348, -20.012, -5.022]
                ),
            },
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in WhisperBackbone.presets:
            self.run_preset_test(
                cls=WhisperBackbone,
                preset=preset,
                input_data=self.input_data,
            )
