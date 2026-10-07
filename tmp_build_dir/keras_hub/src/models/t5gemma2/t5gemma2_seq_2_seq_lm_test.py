import keras
import pytest

from keras_hub.src.models.t5gemma2.t5gemma2_backbone import T5Gemma2Backbone
from keras_hub.src.models.t5gemma2.t5gemma2_seq_2_seq_lm import (
    T5Gemma2Seq2SeqLM,
)
from keras_hub.src.tests.test_case import TestCase


class T5Gemma2Seq2SeqLMTest(TestCase):
    def setUp(self):
        self.backbone_kwargs = {
            "vocabulary_size": 100,
            "encoder_hidden_dim": 32,
            "encoder_intermediate_dim": 64,
            "encoder_num_layers": 2,
            "encoder_num_attention_heads": 4,
            "encoder_num_key_value_heads": 2,
            "encoder_head_dim": 8,
            "encoder_layer_types": [
                "full_attention",
                "full_attention",
            ],
            "decoder_hidden_dim": 32,
            "decoder_intermediate_dim": 64,
            "decoder_num_layers": 2,
            "decoder_num_attention_heads": 4,
            "decoder_num_key_value_heads": 2,
            "decoder_head_dim": 8,
            "decoder_layer_types": [
                "full_attention",
                "full_attention",
            ],
            "dropout_rate": 0.0,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "query_pre_attn_scalar": 1.0,
            "attention_bias": False,
            "hidden_activation": "gelu_approximate",
            "cross_attention_hidden_size": 32,
            "rope_max_wavelength": 10000.0,
            "initializer_range": 0.04,
            "use_query_key_norm": True,
        }
        self.input_data = {
            "encoder_token_ids": keras.ops.ones((2, 16), dtype="int32"),
            "encoder_padding_mask": keras.ops.ones((2, 16), dtype="int32"),
            "decoder_token_ids": keras.ops.ones((2, 16), dtype="int32"),
            "decoder_padding_mask": keras.ops.ones((2, 16), dtype="int32"),
        }

    def test_seq2seq_lm_basics(self):
        backbone = T5Gemma2Backbone(**self.backbone_kwargs)
        lm = T5Gemma2Seq2SeqLM(backbone=backbone)
        output = lm(self.input_data)
        self.assertEqual(keras.ops.shape(output), (2, 16, 100))

    def test_call_encoder(self):
        backbone = T5Gemma2Backbone(**self.backbone_kwargs)
        lm = T5Gemma2Seq2SeqLM(backbone=backbone)
        encoder_output, padding_mask = lm.call_encoder(
            self.input_data["encoder_token_ids"],
            self.input_data["encoder_padding_mask"],
        )
        self.assertEqual(keras.ops.shape(encoder_output), (2, 16, 32))

    def test_build_cache(self):
        backbone = T5Gemma2Backbone(**self.backbone_kwargs)
        lm = T5Gemma2Seq2SeqLM(backbone=backbone)
        hidden_states, cache, extra = lm._build_cache(
            encoder_token_ids=self.input_data["encoder_token_ids"],
            encoder_padding_mask=self.input_data["encoder_padding_mask"],
            decoder_token_ids=self.input_data["decoder_token_ids"],
            decoder_padding_mask=self.input_data["decoder_padding_mask"],
        )
        self.assertEqual(keras.ops.shape(hidden_states), (2, 16, 32))
        self_attention_cache, cross_attention_cache = cache
        # self_attention_cache: (batch, num_layers, 2,
        # seq, kv_heads, head_dim)
        self.assertEqual(
            keras.ops.shape(self_attention_cache), (2, 2, 2, 16, 2, 8)
        )
        # cross_attention_cache: (batch, num_layers, 2,
        # enc_seq, kv_heads, head_dim)
        self.assertEqual(
            keras.ops.shape(cross_attention_cache),
            (2, 2, 2, 16, 2, 8),
        )

    @pytest.mark.large
    def test_saved_model(self):
        backbone = T5Gemma2Backbone(**self.backbone_kwargs)
        self.run_model_saving_test(
            cls=T5Gemma2Seq2SeqLM,
            init_kwargs={"backbone": backbone},
            input_data=self.input_data,
        )
