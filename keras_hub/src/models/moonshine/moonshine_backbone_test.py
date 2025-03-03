import keras
import pytest

from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.tests.test_case import TestCase


class MoonshineBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10000,
            "encoder_num_layers": 2,
            "decoder_num_layers": 2,
            "hidden_dim": 64,
            "intermediate_dim": 512,
            "encoder_num_heads": 8,
            "decoder_num_heads": 8,
            "feedforward_expansion_factor": 4,
            "use_swiglu_activation": False,
            "max_position_embeddings": 2048,
            "pad_head_dim_to_multiple_of": None,
            "partial_rotary_factor": 0.62,
            "dropout": 0.0,
        }
        encoder_input_values = keras.random.uniform((2, 16, 64))
        encoder_attention_mask = keras.ops.ones((2, 16), dtype="int32")
        decoder_token_ids = keras.random.randint(
            shape=(2, 10), minval=0, maxval=10000
        )
        decoder_padding_mask = keras.ops.ones((2, 10), dtype="int32")
        self.input_data = [
            encoder_input_values,
            encoder_attention_mask,
            decoder_token_ids,
            decoder_padding_mask,
        ]
        super(MoonshineBackboneTest, self).setUp()

    # ----------------------------
    # Backbone Functionality Tests
    # ----------------------------

    def test_forward_pass(self):
        backbone = MoonshineBackbone(**self.init_kwargs)
        outputs = backbone(self.input_data)
        self.assertEqual(outputs["encoder_sequence_output"].shape, (2, 16, 64))
        self.assertEqual(
            outputs["decoder_sequence_output"].shape, (2, 10, 10000)
        )

    def test_serialization(self):
        backbone = MoonshineBackbone(**self.init_kwargs)
        config = backbone.get_config()
        new_backbone = MoonshineBackbone.from_config(config)
        outputs = new_backbone(self.input_data)
        self.assertEqual(outputs["encoder_sequence_output"].shape, (2, 16, 64))
        self.assertEqual(
            outputs["decoder_sequence_output"].shape, (2, 10, 10000)
        )

    def test_swiglu_feedforward(self):
        init_kwargs = self.init_kwargs.copy()
        init_kwargs["use_swiglu_activation"] = True
        backbone = MoonshineBackbone(**init_kwargs)
        outputs = backbone(self.input_data)
        self.assertEqual(outputs["encoder_sequence_output"].shape, (2, 16, 64))
        self.assertEqual(
            outputs["decoder_sequence_output"].shape, (2, 10, 10000)
        )

    def test_different_sequence_lengths(self):
        backbone = MoonshineBackbone(**self.init_kwargs)

        # Short sequences.
        short_encoder_input_values = keras.random.uniform((2, 8, 64))
        short_encoder_attention_mask = keras.ops.ones((2, 8), dtype="int32")
        short_decoder_token_ids = keras.random.randint(
            shape=(2, 5), minval=0, maxval=10000
        )
        short_decoder_padding_mask = keras.ops.ones((2, 5), dtype="int32")
        short_input_data = [
            short_encoder_input_values,
            short_encoder_attention_mask,
            short_decoder_token_ids,
            short_decoder_padding_mask,
        ]
        short_outputs = backbone(short_input_data)
        self.assertEqual(
            short_outputs["encoder_sequence_output"].shape, (2, 8, 64)
        )
        self.assertEqual(
            short_outputs["decoder_sequence_output"].shape, (2, 5, 10000)
        )

        # Long sequences.
        long_encoder_input_values = keras.random.uniform((2, 32, 64))
        long_encoder_attention_mask = keras.ops.ones((2, 32), dtype="int32")
        long_decoder_token_ids = keras.random.randint(
            shape=(2, 15), minval=0, maxval=10000
        )
        long_decoder_padding_mask = keras.ops.ones((2, 15), dtype="int32")
        long_input_data = [
            long_encoder_input_values,
            long_encoder_attention_mask,
            long_decoder_token_ids,
            long_decoder_padding_mask,
        ]
        long_outputs = backbone(long_input_data)
        self.assertEqual(
            long_outputs["encoder_sequence_output"].shape, (2, 32, 64)
        )
        self.assertEqual(
            long_outputs["decoder_sequence_output"].shape, (2, 15, 10000)
        )

    def test_predict_model(self):
        backbone = MoonshineBackbone(**self.init_kwargs)
        outputs = backbone.predict(self.input_data)
        self.assertEqual(outputs["encoder_sequence_output"].shape, (2, 16, 64))
        self.assertEqual(
            outputs["decoder_sequence_output"].shape, (2, 10, 10000)
        )

    def test_varying_batch_sizes(self):
        backbone = MoonshineBackbone(**self.init_kwargs)
        for batch_size in [1, 3, 5]:
            encoder_input_values = keras.random.uniform((batch_size, 16, 64))
            encoder_attention_mask = keras.ops.ones(
                (batch_size, 16), dtype="int32"
            )
            decoder_token_ids = keras.random.randint(
                shape=(batch_size, 10), minval=0, maxval=10000
            )
            decoder_padding_mask = keras.ops.ones(
                (batch_size, 10), dtype="int32"
            )
            input_data = [
                encoder_input_values,
                encoder_attention_mask,
                decoder_token_ids,
                decoder_padding_mask,
            ]
            outputs = backbone(input_data)
            self.assertEqual(
                outputs["encoder_sequence_output"].shape, (batch_size, 16, 64)
            )
            self.assertEqual(
                outputs["decoder_sequence_output"].shape,
                (batch_size, 10, 10000),
            )

    # ------------------
    # Standardized tests
    # ------------------

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MoonshineBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MoonshineBackbone.presets.keys():
            input_data = [
                keras.ops.ones(
                    (1, 100, 320 if preset == "moonshine_tiny_en" else 512)
                ),
                keras.ops.ones((1, 100), dtype="int32"),
                keras.ops.ones((1, 10), dtype="int32") * 0,
                keras.ops.ones((1, 10), dtype="int32"),
            ]
            self.run_preset_test(
                cls=MoonshineBackbone,
                preset=preset,
                input_data=input_data,
            )
