import keras
import pytest

from keras_hub.src.models.t5gemma.t5gemma_backbone import T5GemmaBackbone
from keras_hub.src.tests.test_case import TestCase


class T5GemmaBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 100,
            "hidden_dim": 32,
            "intermediate_dim": 64,
            "num_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "dropout_rate": 0.1,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "query_pre_attn_scalar": 1.0,
            "attention_bias": False,
            "hidden_activation": "gelu_approximate",
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 16,
            "cross_attention_hidden_size": 32,
            "attn_logit_softcapping": 50.0,
            "rope_max_wavelength": 10000.0,
            "initializer_range": 0.02,
            "attention_dropout": 0.0,
        }
        self.input_data = {
            "token_ids": keras.ops.ones((2, 16), dtype="int32"),
            "padding_mask": keras.ops.ones((2, 16), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=T5GemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 16, 32),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=T5GemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
