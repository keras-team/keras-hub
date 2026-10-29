import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.t5.t5_backbone import T5Backbone
from keras_hub.src.models.t5.t5_tokenizer import T5Tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_t5


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_preset(self):
        model = T5Backbone.from_preset("hf://google/flan-t5-small")
        self.assertFalse(model.tie_embedding_weights)
        token_ids = np.array([[13959, 1566, 12, 2968, 10, 8774, 1]])
        padding_mask = np.ones_like(token_ids)
        outputs = model.predict(
            {
                "encoder_token_ids": token_ids,
                "encoder_padding_mask": padding_mask,
                "decoder_token_ids": token_ids[:, :3],
                "decoder_padding_mask": padding_mask[:, :3],
            }
        )
        self.assertEqual(
            outputs["encoder_sequence_output"].shape, (1, 7, model.hidden_dim)
        )
        self.assertEqual(
            outputs["decoder_sequence_output"].shape, (1, 3, model.hidden_dim)
        )

    @pytest.mark.large
    def test_tokenizer(self):
        tokenizer = T5Tokenizer.from_preset("hf://google/flan-t5-small")
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertEqual(tokenizer.end_token_id, 1)

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://google-t5/t5-small",
            load_weights=False,
        )
        self.assertIsInstance(model, T5Backbone)

    @pytest.mark.large
    def test_dtype_propagation(self):
        model = T5Backbone.from_preset(
            "hf://google-t5/t5-small",
            load_weights=False,
            dtype="bfloat16",
        )
        self.assertEqual(model.dtype_policy.name, "bfloat16")


class TestConvertBackboneConfig(TestCase):
    def setUp(self):
        # Mirrors `google/flan-t5-small`.
        self.flan_config = {
            "model_type": "t5",
            "vocab_size": 32128,
            "d_model": 512,
            "d_kv": 64,
            "d_ff": 1024,
            "num_layers": 8,
            "num_decoder_layers": 8,
            "num_heads": 6,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-6,
            "feed_forward_proj": "gated-gelu",
            "relative_attention_num_buckets": 32,
            "relative_attention_max_distance": 128,
            "tie_word_embeddings": False,
        }

    def test_original_t5_defaults(self):
        # `google-t5/t5-small` omits `feed_forward_proj`,
        # `tie_word_embeddings`, `num_decoder_layers` and
        # `relative_attention_max_distance`, so HF defaults must apply.
        config = {
            "model_type": "t5",
            "vocab_size": 32128,
            "d_model": 512,
            "d_kv": 64,
            "d_ff": 2048,
            "num_layers": 6,
            "num_heads": 8,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-6,
            "relative_attention_num_buckets": 32,
        }
        keras_config = convert_t5.convert_backbone_config(config)
        self.assertEqual(
            keras_config,
            {
                "vocabulary_size": 32128,
                "num_layers": 6,
                "num_heads": 8,
                "hidden_dim": 512,
                "intermediate_dim": 2048,
                "key_value_dim": 64,
                "dropout": 0.1,
                "activation": "relu",
                "use_gated_activation": False,
                "layer_norm_epsilon": 1e-6,
                "tie_embedding_weights": True,
            },
        )

    def test_flan_t5(self):
        keras_config = convert_t5.convert_backbone_config(self.flan_config)
        self.assertEqual(keras_config["activation"], "gelu")
        self.assertTrue(keras_config["use_gated_activation"])
        self.assertFalse(keras_config["tie_embedding_weights"])
        self.assertEqual(keras_config["num_layers"], 8)
        self.assertEqual(keras_config["key_value_dim"], 64)

    def test_transformers_v5_config(self):
        # transformers>=5 forces `tie_word_embeddings=True` and stores whether
        # decoder outputs are scaled in `scale_decoder_outputs`.
        config = {
            **self.flan_config,
            "tie_word_embeddings": True,
            "scale_decoder_outputs": False,
        }
        keras_config = convert_t5.convert_backbone_config(config)
        self.assertFalse(keras_config["tie_embedding_weights"])
        config["scale_decoder_outputs"] = True
        keras_config = convert_t5.convert_backbone_config(config)
        self.assertTrue(keras_config["tie_embedding_weights"])

    def test_asymmetric_layers_raise(self):
        config = {**self.flan_config, "num_decoder_layers": 4}
        with self.assertRaisesRegex(ValueError, "num_decoder_layers"):
            convert_t5.convert_backbone_config(config)

    def test_relative_attention_raise(self):
        config = {**self.flan_config, "relative_attention_num_buckets": 64}
        with self.assertRaisesRegex(ValueError, "relative_attention"):
            convert_t5.convert_backbone_config(config)
        config = {**self.flan_config, "relative_attention_max_distance": 256}
        with self.assertRaisesRegex(ValueError, "relative_attention"):
            convert_t5.convert_backbone_config(config)

    def test_feed_forward_proj_raise(self):
        for feed_forward_proj in ("gated-foo", "foo", "double-gelu", "a-b-c"):
            config = {
                **self.flan_config,
                "feed_forward_proj": feed_forward_proj,
            }
            with self.assertRaisesRegex(ValueError, "feed_forward_proj"):
                convert_t5.convert_backbone_config(config)
