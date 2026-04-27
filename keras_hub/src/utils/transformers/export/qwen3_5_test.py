"""End-to-end export test for Qwen3.5 → HuggingFace Transformers.

Verifies:
  1. Export produces valid config.json, model.safetensors, tokenizer files
  2. Config fields match between KerasHub and HF
  3. Forward-pass logits are numerically identical (round-trip parity)
"""

import json
import os

import keras.ops as ops
import numpy as np
from transformers import AutoModelForCausalLM

from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_causal_lm import Qwen3_5CausalLM
from keras_hub.src.models.qwen3_5.qwen3_5_causal_lm_preprocessor import (
    Qwen3_5CausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_5.qwen3_5_tokenizer import Qwen3_5Tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.export.hf_exporter import (
    export_to_safetensors,
)


class TestQwen3_5Export(TestCase):
    def test_export_to_hf(self):
        # 1. Setup Tokenizer Assets (BPE)
        vocab = {
            "<|endoftext|>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "The": 3,
            "quick": 4,
            "brown": 5,
            "fox": 6,
            "T": 7,
            "h": 8,
            "e": 9,
            "q": 10,
            "u": 11,
            "i": 12,
            "c": 13,
            "k": 14,
            " ": 15,
        }
        merges = ["q u", "i c", "k "]

        temp_dir = self.get_temp_dir()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        merges_path = os.path.join(temp_dir, "merges.txt")

        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        with open(merges_path, "w") as f:
            f.write("\n".join(merges))

        tokenizer = Qwen3_5Tokenizer(
            vocabulary=vocab_path,
            merges=merges_path,
            has_vision_tokens=False,
        )

        # 2. Create a small text-only Backbone
        # 4 layers: layers 0,1,2 = linear_attention, layer 3 = full_attention
        backbone = Qwen3_5Backbone(
            vocabulary_size=len(vocab),
            num_layers=4,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            hidden_dim=64,
            intermediate_dim=128,
            partial_rotary_factor=0.25,
            rope_max_wavelength=10000,
            layer_norm_epsilon=1e-6,
            dropout=0,
            tie_word_embeddings=False,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
        )

        # 3. Create Preprocessor & CausalLM
        preprocessor = Qwen3_5CausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=32
        )
        keras_model = Qwen3_5CausalLM(
            backbone=backbone, preprocessor=preprocessor
        )

        # 4. Set Random Weights for reproducibility
        rng = np.random.default_rng(42)
        weights = keras_model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        keras_model.set_weights(weights)

        # 5. Export to Hugging Face format
        export_path = os.path.join(temp_dir, "export_task")
        export_to_safetensors(keras_model, export_path)

        # Patch config for our tiny vocab
        config_path = os.path.join(export_path, "config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)
        cfg["eos_token_id"] = 2  # <|im_end|>
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # 6. Load with Hugging Face Transformers
        hf_model = AutoModelForCausalLM.from_pretrained(export_path)

        # 7. Verify Configuration
        hf_config = hf_model.config
        # get_text_config() works for both multimodal (nested) and
        # text-only (flat) config structures.
        text_cfg = hf_config.get_text_config()

        self.assertEqual(
            text_cfg.vocab_size,
            backbone.vocabulary_size,
            "Vocabulary sizes do not match",
        )
        self.assertEqual(
            text_cfg.num_hidden_layers,
            backbone.num_layers,
            "Number of layers do not match",
        )
        self.assertEqual(
            text_cfg.num_attention_heads,
            backbone.num_query_heads,
            "Number of query heads do not match",
        )
        self.assertEqual(
            text_cfg.num_key_value_heads,
            backbone.num_key_value_heads,
            "Number of KV heads do not match",
        )
        self.assertEqual(
            text_cfg.hidden_size,
            backbone.hidden_dim,
            "Hidden dimensions do not match",
        )
        self.assertEqual(
            text_cfg.intermediate_size,
            backbone.intermediate_dim,
            "Intermediate dimensions do not match",
        )
        self.assertEqual(
            text_cfg.rms_norm_eps,
            backbone.layer_norm_epsilon,
            "Layer norm epsilon does not match",
        )
        self.assertEqual(
            text_cfg.head_dim,
            backbone.head_dim,
            "Head dimensions do not match",
        )
        self.assertEqual(
            text_cfg.linear_num_key_heads,
            backbone.linear_num_key_heads,
            "Linear num key heads do not match",
        )
        self.assertEqual(
            text_cfg.linear_num_value_heads,
            backbone.linear_num_value_heads,
            "Linear num value heads do not match",
        )

        # 8. Compare Logits (round-trip numerical parity)
        import torch

        input_ids = np.array([[3, 4]])  # "The quick"

        keras_inputs = {
            "token_ids": input_ids,
            "padding_mask": np.ones_like(input_ids),
        }
        keras_logits = keras_model(keras_inputs)

        hf_inputs = {"input_ids": torch.tensor(input_ids)}
        hf_logits = hf_model(**hf_inputs).logits

        keras_logits_np = ops.convert_to_numpy(keras_logits)
        hf_logits_np = hf_logits.detach().cpu().numpy()

        self.assertAllClose(keras_logits_np, hf_logits_np, atol=1e-3, rtol=1e-3)
