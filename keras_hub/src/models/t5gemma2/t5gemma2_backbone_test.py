import keras
import pytest

from keras_hub.src.models.t5gemma2.t5gemma2_backbone import T5Gemma2Backbone
from keras_hub.src.tests.test_case import TestCase


class T5Gemma2BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 100,
            "encoder_hidden_dim": 32,
            "encoder_intermediate_dim": 64,
            "encoder_num_layers": 2,
            "encoder_num_attention_heads": 4,
            "encoder_num_key_value_heads": 2,
            "encoder_head_dim": 8,
            "encoder_layer_types": [
                "sliding_attention",
                "full_attention",
            ],
            "decoder_hidden_dim": 32,
            "decoder_intermediate_dim": 64,
            "decoder_num_layers": 2,
            "decoder_num_attention_heads": 4,
            "decoder_num_key_value_heads": 2,
            "decoder_head_dim": 8,
            "decoder_layer_types": [
                "sliding_attention",
                "full_attention",
            ],
            "dropout_rate": 0.1,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "query_pre_attn_scalar": 1.0,
            "attention_bias": False,
            "hidden_activation": "gelu_approximate",
            "sliding_window": 16,
            "cross_attention_hidden_size": 32,
            "attn_logit_softcapping": 50.0,
            "rope_max_wavelength": 10000.0,
            "initializer_range": 0.04,
            "attention_dropout": 0.1,
            "use_query_key_norm": True,
        }
        self.input_data = {
            "encoder_token_ids": keras.ops.ones((2, 16), dtype="int32"),
            "encoder_padding_mask": keras.ops.ones((2, 16), dtype="int32"),
            "decoder_token_ids": keras.ops.ones((2, 16), dtype="int32"),
            "decoder_padding_mask": keras.ops.ones((2, 16), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=T5Gemma2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "encoder_sequence_output": (2, 16, 32),
                "decoder_sequence_output": (2, 16, 32),
            },
        )

    def test_asymmetrical_backbone(self):
        asym_kwargs = {
            "vocabulary_size": 100,
            "encoder_hidden_dim": 32,
            "encoder_intermediate_dim": 96,
            "encoder_num_layers": 3,
            "encoder_num_attention_heads": 4,
            "encoder_num_key_value_heads": 2,
            "encoder_head_dim": 8,
            "encoder_layer_types": ["full_attention"] * 3,
            "decoder_hidden_dim": 32,
            "decoder_intermediate_dim": 64,
            "decoder_num_layers": 2,
            "decoder_num_attention_heads": 4,
            "decoder_num_key_value_heads": 2,
            "decoder_head_dim": 8,
            "decoder_layer_types": [
                "sliding_attention",
                "full_attention",
            ],
            "sliding_window": 16,
            "dropout_rate": 0.1,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "cross_attention_hidden_size": 32,
            "use_query_key_norm": True,
        }
        self.run_backbone_test(
            cls=T5Gemma2Backbone,
            init_kwargs=asym_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "encoder_sequence_output": (2, 16, 32),
                "decoder_sequence_output": (2, 16, 32),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=T5Gemma2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in T5Gemma2Backbone.presets:
            self.run_preset_test(
                cls=T5Gemma2Backbone,
                preset=preset,
                input_data=self.input_data,
            )

    def test_distribution(self):
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")
        devices = keras.distribution.list_devices("CPU")
        if len(devices) == 1:
            self.skipTest("`ModelParallel` testing requires multiple devices.")
        # Pinned to exactly 2 devices (not len(devices)): the default test
        # config's *_num_key_value_heads=2 is intentionally left as-is (not
        # divisible by every host's device count) to regression-test that
        # key/value kernels are left replicated rather than sharded -- see
        # get_layout_map's comment.
        devices = devices[:2]
        device_mesh = keras.distribution.DeviceMesh(
            shape=(1, 2),
            axis_names=("batch", "model"),
            devices=devices,
        )

        layout_map = T5Gemma2Backbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(layout_map=layout_map)
        with distribution.scope():
            model = T5Gemma2Backbone(**self.init_kwargs)

        for w in model.weights:
            if "encoder_token_embedding/embeddings" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch")
                )
            if "decoder_token_embedding/embeddings" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch")
                )
            if "self_attention/query/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch", None)
                )
            if "self_attention/key/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", None, None)
                )
            if "self_attention/value/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", None, None)
                )
            if "merged_attention/query/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch", None)
                )
            if "merged_attention/key/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", None, None)
                )
            if "merged_attention/value/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", None, None)
                )
            if "attention/attention_output/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", None, "batch")
                )
            if "gate_proj/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "up_proj/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "down_proj/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch")
                )
