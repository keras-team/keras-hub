import json
import os

import keras.ops as ops
import numpy as np
from transformers import Qwen2ForCausalLM

from keras_hub.src.models.qwen.qwen_backbone import QwenBackbone
from keras_hub.src.models.qwen.qwen_causal_lm import QwenCausalLM
from keras_hub.src.models.qwen.qwen_causal_lm_preprocessor import (
    QwenCausalLMPreprocessor,
)
from keras_hub.src.models.qwen.qwen_tokenizer import QwenTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.export.hf_exporter import (
    export_to_safetensors,
)


class TestQwenExport(TestCase):
    def test_export_to_hf(self):
        # 1. Setup Tokenizer Assets (BPE)
        # We need a vocab that covers the prompt "The quick"
        # Qwen uses specific bytes for spaces (like Ġ or space char).
        vocab = {
            "<|endoftext|>": 0,
            "The": 1,
            "quick": 2,
            "brown": 3,
            "fox": 4,
            # Add partials/characters to ensure fallback works
            "T": 5,
            "h": 6,
            "e": 7,
            "q": 8,
            "u": 9,
            "i": 10,
            "c": 11,
            "k": 12,
            " ": 13,  # Space
        }
        # Add a dummy merge to satisfy initialization
        merges = ["q u", "i c", "k"]

        temp_dir = self.get_temp_dir()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        merges_path = os.path.join(temp_dir, "merges.txt")

        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        with open(merges_path, "w") as f:
            f.write("\n".join(merges))

        # Instantiate Keras Tokenizer
        tokenizer = QwenTokenizer(vocabulary=vocab_path, merges=merges_path)

        # 2. Create a small Backbone
        # Use small dimensions for speed and RAM efficiency
        backbone = QwenBackbone(
            vocabulary_size=len(vocab),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            rope_max_wavelength=10000,
            rope_scaling_factor=1.0,
            layer_norm_epsilon=1e-6,
            dropout=0,
        )

        # 3. Create Preprocessor & Model
        preprocessor = QwenCausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=32
        )
        keras_model = QwenCausalLM(backbone=backbone, preprocessor=preprocessor)

        # 4. Set Random Weights
        # Ensures verification of weight transfer logic
        rng = np.random.default_rng(42)
        weights = keras_model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        keras_model.set_weights(weights)

        # 5. Export to Hugging Face format
        export_path = os.path.join(temp_dir, "export_task")
        export_to_safetensors(keras_model, export_path)

        # Patch the config for EOS token consistency with our tiny vocab
        config_path = os.path.join(export_path, "config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)

        cfg["eos_token_id"] = 0  # <|endoftext|>

        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # 6. Load with Hugging Face Transformers
        # Use Qwen2 classes
        hf_model = Qwen2ForCausalLM.from_pretrained(export_path)

        # 7. Verify Configuration
        hf_config = hf_model.config

        self.assertEqual(
            hf_config.vocab_size,
            backbone.vocabulary_size,
            "Vocabulary sizes do not match",
        )
        self.assertEqual(
            hf_config.num_hidden_layers,
            backbone.num_layers,
            "Number of layers do not match",
        )
        self.assertEqual(
            hf_config.num_attention_heads,
            backbone.num_query_heads,
            "Number of query heads do not match",
        )
        self.assertEqual(
            hf_config.num_key_value_heads,
            backbone.num_key_value_heads,
            "Number of KV heads do not match",
        )
        self.assertEqual(
            hf_config.hidden_size,
            backbone.hidden_dim,
            "Hidden dimensions do not match",
        )
        self.assertEqual(
            hf_config.intermediate_size,
            backbone.intermediate_dim,
            "Intermediate dimensions do not match",
        )
        self.assertEqual(
            hf_config.rms_norm_eps,
            backbone.layer_norm_epsilon,
            "Layer norm epsilon does not match",
        )

        # 8. Compare Outputs (Generation via IDs)

        # ID 1="The", ID 2="quick" (based on your dummy vocab)
        input_ids = np.array([[1, 2]])

        keras_inputs = {
            "token_ids": input_ids,
            "padding_mask": np.ones_like(input_ids),
        }
        keras_logits = keras_model(keras_inputs)

        # HF Generation
        import torch

        hf_inputs = {"input_ids": torch.tensor(input_ids)}
        hf_logits = hf_model(**hf_inputs).logits

        # Verify Logits Match
        keras_logits_np = ops.convert_to_numpy(keras_logits)
        hf_logits_np = hf_logits.detach().cpu().numpy()

        self.assertAllClose(keras_logits_np, hf_logits_np, atol=1e-3, rtol=1e-3)
