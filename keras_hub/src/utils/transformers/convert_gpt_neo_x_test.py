from unittest.mock import patch

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gpt_neo_x.gpt_neo_x_backbone import GPTNeoXBackbone
from keras_hub.src.models.gpt_neo_x.gpt_neo_x_causal_lm import GPTNeoXCausalLM
from keras_hub.src.models.gpt_neo_x.gpt_neo_x_tokenizer import GPTNeoXTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import gelu_approximate
from keras_hub.src.utils.transformers import convert_gpt_neo_x
from keras_hub.src.utils.transformers.convert_gpt_neo_x import (
    convert_backbone_config,
)


class ConvertGPTNeoXTest(TestCase):
    def setUp(self):
        # A complete config in the shape of EleutherAI/pythia-14m. The tests
        # below start from it and change only what they are about, so a
        # KeyError can only come from the key under test.
        self.hf_config = {
            "hidden_act": "gelu",
            "vocab_size": 10,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "hidden_size": 1,
            "intermediate_size": 1,
            "max_position_embeddings": 1,
            "layer_norm_eps": 1.0,
            "tie_word_embeddings": False,
            "use_parallel_residual": True,
            "rotary_pct": 0.25,
            "rotary_emb_base": 10000,
        }

    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = GPTNeoXCausalLM.from_preset("hf://EleutherAI/pythia-14m")
        prompt_ids = [1276, 310, 634, 7583, 6882, 2092, 32]
        # Hugging Face float32 greedy output for `prompt_ids`. Stops at
        # eight: every step kept has a top-2 logit margin above 0.1; the
        # ninth is a 0.007 near-tie that already flips in Hugging Face's
        # default float16.
        new_ids = [187, 187, 42, 452, 247, 2257, 273, 794]
        n = len(prompt_ids)
        total = n + len(new_ids)

        model.preprocessor = None
        model.compile(sampler="greedy")

        pad_ids = np.zeros((1, total), dtype="int32")
        pad_ids[0, :n] = prompt_ids
        pad_mask = np.zeros((1, total), dtype="int32")
        pad_mask[0, :n] = 1

        k_out = model.generate(
            {"token_ids": pad_ids, "padding_mask": pad_mask},
            stop_token_ids=None,
        )
        k_gen = np.asarray(ops.convert_to_numpy(k_out["token_ids"]))
        k_gen = k_gen[0, n:total]
        self.assertEqual(k_gen.tolist(), new_ids)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://EleutherAI/pythia-14m",
            load_weights=False,
        )
        self.assertIsInstance(model, GPTNeoXCausalLM)
        self.assertFalse(model.preprocessor.add_start_token)
        model = Backbone.from_preset(
            "hf://EleutherAI/pythia-14m",
            load_weights=False,
        )
        self.assertIsInstance(model, GPTNeoXBackbone)
        self.assertFalse(model.tie_word_embeddings)
        self.assertEqual(model.get_config()["activation"], "gelu")
        self.assertEqual(model.rotary_percentage, 0.25)
        self.assertEqual(model.rotary_max_wavelength, 10000)

    def test_config_without_hidden_act(self):
        config = dict(self.hf_config)
        del config["hidden_act"]
        with self.assertRaisesRegex(KeyError, "hidden_act"):
            convert_backbone_config(config)

    def test_config_without_tie_word_embeddings(self):
        config = dict(self.hf_config)
        del config["tie_word_embeddings"]
        with self.assertRaisesRegex(KeyError, "tie_word_embeddings"):
            convert_backbone_config(config)

    def test_config_without_rotary_pct(self):
        config = dict(self.hf_config)
        del config["rotary_pct"]
        # The fallback key is the one named.
        with self.assertRaisesRegex(KeyError, "partial_rotary_factor"):
            convert_backbone_config(config)

    def test_config_without_rotary_emb_base(self):
        config = dict(self.hf_config)
        del config["rotary_emb_base"]
        with self.assertRaisesRegex(KeyError, "rope_theta"):
            convert_backbone_config(config)

    def test_config_with_false_parallel_residual(self):
        config = dict(self.hf_config)
        config["use_parallel_residual"] = False
        with self.assertRaises(ValueError):
            convert_backbone_config(config)

    def test_config_with_rope_parameters(self):
        config = dict(self.hf_config)
        del config["rotary_pct"]
        del config["rotary_emb_base"]
        config["hidden_act"] = "gelu_fast"
        config["rope_parameters"] = {
            "partial_rotary_factor": 0.5,
            "rope_theta": 20000,
        }
        config = convert_backbone_config(config)
        self.assertEqual(config["rotary_percentage"], 0.5)
        self.assertEqual(config["rotary_max_wavelength"], 20000)
        self.assertIs(config["activation"], gelu_approximate)

    def test_config_gelu_variants(self):
        for hidden_act in ("gelu_new", "gelu_pytorch_tanh", "gelu_fast"):
            config = {**self.hf_config, "hidden_act": hidden_act}
            config = convert_backbone_config(config)
            self.assertIs(config["activation"], gelu_approximate)
        config = convert_backbone_config(self.hf_config)
        self.assertEqual(config["activation"], "gelu")

    def test_config_hidden_dropout(self):
        config = {**self.hf_config, "hidden_dropout": 0.1}
        self.assertEqual(convert_backbone_config(config)["dropout"], 0.1)
        # gpt-neox-20b spells the key `hidden_dropout_prob`.
        config = {**self.hf_config, "hidden_dropout_prob": 0.2}
        self.assertEqual(convert_backbone_config(config)["dropout"], 0.2)
        config = convert_backbone_config(self.hf_config)
        self.assertEqual(config["dropout"], 0.0)

    def test_config_rotary_values(self):
        config = dict(self.hf_config)
        config["rotary_pct"] = 0.5
        config["rotary_emb_base"] = 20000
        config = convert_backbone_config(config)
        self.assertEqual(config["rotary_percentage"], 0.5)
        self.assertEqual(config["rotary_max_wavelength"], 20000)

    def test_unsupported_hidden_act(self):
        config = {**self.hf_config, "hidden_act": "silu"}
        with self.assertRaises(ValueError):
            convert_backbone_config(config)

    def test_convert_tokenizer_added_tokens(self):
        tokenizer_config = {
            "model": {
                "vocab": {
                    "<|endoftext|>": 0,
                    "<|padding|>": 1,
                    "a": 2,
                    "b": 3,
                    "ab": 4,
                },
                "merges": [["a", "b"]],
            },
            "added_tokens": [
                {"id": 0, "content": "<|endoftext|>"},
                {"id": 1, "content": "<|padding|>"},
                {"id": 5, "content": "   "},
                {"id": 6, "content": "  "},
            ],
        }
        with patch.object(
            convert_gpt_neo_x, "load_json", return_value=tokenizer_config
        ):
            tokenizer = convert_gpt_neo_x.convert_tokenizer(
                GPTNeoXTokenizer, "unused"
            )
        self.assertEqual(tokenizer.vocabulary_size(), 7)
        self.assertEqual(tokenizer.pad_token_id, 1)
        self.assertAllEqual(tokenizer("a   ab  a"), [2, 5, 4, 6, 2])
