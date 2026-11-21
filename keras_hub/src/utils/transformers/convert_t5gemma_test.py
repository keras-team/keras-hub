import keras
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.t5gemma.t5gemma_backbone import T5GemmaBackbone
from keras_hub.src.models.t5gemma.t5gemma_seq_2_seq_lm import T5GemmaSeq2SeqLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_t5gemma


# NOTE: This test is valid and should pass locally. It is skipped only on
# TensorFlow GPU CI because of ResourceExhaustedError (OOM). Revisit once
# TensorFlow GPU CI runs without hitting OOM.
@pytest.mark.skipif(
    keras.backend.backend() == "tensorflow",
    reason="TensorFlow GPU CI OOM (ResourceExhaustedError)",
)
class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = T5GemmaSeq2SeqLM.from_preset(
            "hf://harshaljanjani/tiny-t5gemma-test"
        )
        prompt = "What is the capital of France?"
        model.generate([prompt], max_length=15)

    @pytest.mark.extra_large
    def test_class_detection(self):
        preset_name = "hf://harshaljanjani/tiny-t5gemma-test"
        model = Seq2SeqLM.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, T5GemmaSeq2SeqLM)
        model = Backbone.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, T5GemmaBackbone)


class TestT5GemmaRopeTheta(TestCase):
    def test_convert_backbone_config_rope_theta(self):
        # transformers < 5 format
        encoder_cfg = {
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "layer_types": ["full_attention", "full_attention"],
        }
        decoder_cfg = {
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "layer_types": ["full_attention", "full_attention"],
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
            "encoder": encoder_cfg,
            "decoder": decoder_cfg,
        }
        keras_config = convert_t5gemma.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)

        # transformers >= 5 format
        decoder_cfg["rope_parameters"] = {"rope_theta": 20000.0}
        keras_config = convert_t5gemma.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)
