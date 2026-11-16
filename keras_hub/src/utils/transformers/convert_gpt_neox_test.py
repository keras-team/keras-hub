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
from keras_hub.src.utils.transformers import convert_gpt_neox
from keras_hub.src.utils.transformers.convert_gpt_neox import (
    convert_backbone_config,
)


class ConvertGPTNeoXTest(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = GPTNeoXCausalLM.from_preset("hf://EleutherAI/pythia-14m")
        prompt_ids = [1276, 310, 634, 7583, 6882, 2092, 32]
        new_ids = [187, 187, 42, 452, 247, 2257, 273, 794, 15, 309]
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
        with self.assertRaises(KeyError):
            convert_backbone_config(
                {
                    "vocab_size": 10,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "hidden_size": 1,
                    "intermediate_size": 1,
                    "max_position_embeddings": 1,
                    "layer_norm_eps": 1.0,
                    "tie_word_embeddings": False,
                }
            )

    def test_config_without_tie_word_embeddings(self):
        with self.assertRaises(KeyError):
            convert_backbone_config(
                {
                    "hidden_act": "gelu_fast",
                    "vocab_size": 10,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "hidden_size": 1,
                    "intermediate_size": 1,
                    "max_position_embeddings": 1,
                    "layer_norm_eps": 1.0,
                }
            )

    def test_config_with_false_parallel_residual(self):
        with self.assertRaises(ValueError):
            convert_backbone_config(
                {
                    "use_parallel_residual": False,
                    "hidden_act": "gelu",
                    "vocab_size": 10,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "hidden_size": 1,
                    "intermediate_size": 1,
                    "max_position_embeddings": 1,
                    "layer_norm_eps": 1.0,
                    "tie_word_embeddings": False,
                }
            )

    def test_config_with_rope_parameters(self):
        config = convert_backbone_config(
            {
                "hidden_act": "gelu_fast",
                "vocab_size": 10,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "hidden_size": 1,
                "intermediate_size": 1,
                "max_position_embeddings": 1,
                "layer_norm_eps": 1.0,
                "tie_word_embeddings": False,
                "rope_parameters": {
                    "partial_rotary_factor": 0.5,
                    "rope_theta": 20000,
                },
            }
        )
        self.assertEqual(config["rotary_percentage"], 0.5)
        self.assertEqual(config["rotary_max_wavelength"], 20000)

    def test_unsupported_hidden_act(self):
        with self.assertRaises(ValueError):
            convert_backbone_config(
                {
                    "hidden_act": "silu",
                    "vocab_size": 10,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "hidden_size": 1,
                    "intermediate_size": 1,
                    "max_position_embeddings": 1,
                    "layer_norm_eps": 1.0,
                }
            )

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
            convert_gpt_neox, "load_json", return_value=tokenizer_config
        ):
            tokenizer = convert_gpt_neox.convert_tokenizer(
                GPTNeoXTokenizer, "unused"
            )
        self.assertEqual(tokenizer.vocabulary_size(), 7)
        self.assertEqual(tokenizer.pad_token_id, 1)
        self.assertAllEqual(tokenizer("a   ab  a"), [2, 5, 4, 6, 2])
