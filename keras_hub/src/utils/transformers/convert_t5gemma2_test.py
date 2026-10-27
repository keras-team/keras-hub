import keras
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.t5gemma2.t5gemma2_backbone import T5Gemma2Backbone
from keras_hub.src.models.t5gemma2.t5gemma2_seq_2_seq_lm import (
    T5Gemma2Seq2SeqLM,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_t5gemma2


class TestT5Gemma2Converter(TestCase):
    @pytest.mark.skipif(
        keras.backend.backend() == "tensorflow",
        reason="TensorFlow GPU CI OOM (ResourceExhaustedError)",
    )
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = T5Gemma2Seq2SeqLM.from_preset("hf://google/t5gemma-2-270m-270m")
        prompt = "What is the capital of France?"
        model.generate([prompt], max_length=15)

    @pytest.mark.extra_large
    def test_class_detection(self):
        preset_name = "hf://google/t5gemma-2-270m-270m"
        model = Seq2SeqLM.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, T5Gemma2Seq2SeqLM)
        model = Backbone.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, T5Gemma2Backbone)

    def test_convert_backbone_config_rope_theta(self):
        # transformers < 5 format (flat)
        enc_text_cfg = {
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "layer_types": ["full_attention", "sliding_attention"],
            "tie_word_embeddings": True,
            "rope_theta": 10000.0,
        }
        decoder_cfg = {
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "layer_types": ["full_attention", "sliding_attention"],
            "dropout_rate": 0.1,
            "rms_norm_eps": 1e-5,
            "query_pre_attn_scalar": 1.0,
            "attention_bias": False,
            "hidden_activation": "gelu",
            "initializer_range": 0.02,
            "attention_dropout": 0.1,
            "sliding_window": 4096,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "rope_theta": 10000.0,
        }
        transformers_config = {
            "encoder": {"text_config": enc_text_cfg},
            "decoder": decoder_cfg,
            "eoi_token_index": 1,
        }
        keras_config = convert_t5gemma2.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)
        self.assertEqual(keras_config["encoder_rope_max_wavelength"], 10000.0)

        # transformers >= 5 format (nested)
        enc_text_cfg["rope_parameters"] = {
            "sliding_attention": {"rope_theta": 20000.0},
            "full_attention": {"factor": 1.0},
        }
        decoder_cfg["rope_parameters"] = {
            "sliding_attention": {"rope_theta": 20000.0},
            "full_attention": {"factor": 1.0},
        }
        keras_config = convert_t5gemma2.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)
        self.assertEqual(keras_config["encoder_rope_max_wavelength"], 20000.0)
