"""CPU unit tests for the neutral KerasHub HF config used by vLLM.

Verifies the config we write resolves through transformers' ``AutoConfig``
without borrowing any model family's name.
"""

import json
import os
import tempfile

from transformers import AutoConfig

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm.hf_config import KERAS_HUB_ARCHITECTURE
from keras_hub.src.vllm.hf_config import KERAS_HUB_MODEL_TYPE
from keras_hub.src.vllm.hf_config import register_hf_config


class HFConfigTest(TestCase):
    def test_neutral_names(self):
        # No borrowed model family in the names we write.
        self.assertEqual(KERAS_HUB_MODEL_TYPE, "keras_hub")
        self.assertEqual(KERAS_HUB_ARCHITECTURE, "KerasHubForCausalLM")

    def test_register_is_idempotent(self):
        # Safe to call from setup_vllm_model and again from each worker import.
        register_hf_config()
        register_hf_config()

    def test_autoconfig_round_trip_carries_dims(self):
        register_hf_config()
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(
                {
                    "architectures": [KERAS_HUB_ARCHITECTURE],
                    "model_type": KERAS_HUB_MODEL_TYPE,
                    "keras_hub_preset": "gpt2_base_en",
                    "vocab_size": 50257,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                    "num_key_value_heads": 12,
                    "head_dim": 64,
                    "hidden_size": 768,
                },
                f,
            )

        cfg = AutoConfig.from_pretrained(tmp)
        self.assertEqual(type(cfg).__name__, "KerasHubConfig")
        self.assertEqual(cfg.model_type, "keras_hub")
        # vLLM reads these for KV-cache profiling.
        self.assertEqual(cfg.num_hidden_layers, 12)
        self.assertEqual(cfg.num_key_value_heads, 12)
        self.assertEqual(cfg.head_dim, 64)
        self.assertEqual(cfg.vocab_size, 50257)
        # The preset (what actually selects the model) survives.
        self.assertEqual(cfg.keras_hub_preset, "gpt2_base_en")

    def test_defaults_fill_missing_kv_fields(self):
        # A sparse config still yields the fields vLLM expects (no None).
        register_hf_config()
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(
                {
                    "model_type": KERAS_HUB_MODEL_TYPE,
                    "hidden_size": 512,
                    "num_attention_heads": 8,
                },
                f,
            )
        cfg = AutoConfig.from_pretrained(tmp)
        # num_key_value_heads defaults to num_attention_heads; head_dim derived.
        self.assertEqual(cfg.num_key_value_heads, 8)
        self.assertEqual(cfg.head_dim, 512 // 8)
