import os
import json
import numpy as np
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import keras.ops as ops
from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.models.qwen3.qwen3_causal_lm import Qwen3CausalLM
from keras_hub.src.models.qwen3.qwen3_causal_lm_preprocessor import (
    Qwen3CausalLMPreprocessor,
)
from keras_hub.src.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.export.hf_exporter import (
    export_to_safetensors,
)


class TestQwen3Export(TestCase):

    def test_export_to_hf(self):
        # 1. Setup Dummy Tokenizer Assets (BPE)
        vocab = {
            # Special Tokens
            "<|endoftext|>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            
            # Base Characters
            "Ġ": 3, "q": 4, "u": 5, "i": 6, "c": 7, "k": 8,
            
            # Merged Tokens
            "qu": 9, 
            "ic": 10,
            
            # Full Words
            "The": 11, "quick": 12, "brown": 13, "fox": 14
        }
        
        merges = ["q u", "i c"] # Merges imply "qu" and "ic" exist
        
        temp_dir = self.get_temp_dir()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        merges_path = os.path.join(temp_dir, "merges.txt")
        
        with open(vocab_path, "w") as f: json.dump(vocab, f)
        with open(merges_path, "w") as f: f.write("\n".join(merges))

        tokenizer = Qwen3Tokenizer(vocabulary=vocab_path, merges=merges_path)

        # 2. Create Tiny Qwen3 Backbone
        backbone = Qwen3Backbone(
            vocabulary_size=len(vocab),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            head_dim=16,
            rope_max_wavelength=10000,
            rope_scaling_factor=1.0,
            layer_norm_epsilon=1e-6,
            dropout=0,
        )

        # 3. Create Model
        preprocessor = Qwen3CausalLMPreprocessor(tokenizer=tokenizer, sequence_length=32)
        keras_model = Qwen3CausalLM(backbone=backbone, preprocessor=preprocessor)

        # 4. Randomize Weights
        rng = np.random.default_rng(42)
        weights = keras_model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        keras_model.set_weights(weights)

        # 5. Export
        export_path = os.path.join(temp_dir, "export_task")
        export_to_safetensors(keras_model, export_path)
        
        # Patch config for dummy vocab compatibility
        config_path = os.path.join(export_path, "config.json")
        with open(config_path, "r") as f: cfg = json.load(f)
        cfg["eos_token_id"] = 0
        with open(config_path, "w") as f: json.dump(cfg, f, indent=2)

        # 6. Load with Hugging Face (Qwen2 class works for Qwen3)
        hf_model = AutoModelForCausalLM.from_pretrained(export_path, trust_remote_code=True)
        hf_tokenizer = AutoTokenizer.from_pretrained(export_path)

        # 7. Verify Config
        hf_config = hf_model.config
        self.assertEqual(hf_config.vocab_size, backbone.vocabulary_size)
        self.assertEqual(hf_config.num_hidden_layers, backbone.num_layers)
        self.assertEqual(hf_config.num_attention_heads, backbone.num_query_heads)
        self.assertEqual(hf_config.num_key_value_heads, backbone.num_key_value_heads)
        self.assertEqual(hf_config.hidden_size, backbone.hidden_dim)
        self.assertEqual(hf_config.intermediate_size, backbone.intermediate_dim)

        # 8. Compare Logits
        # Using raw IDs to bypass tokenizer quirks with dummy vocab
        input_ids = np.array([[1, 2, 4]]) 
        
        keras_inputs = {
            "token_ids": input_ids,
            "padding_mask": np.ones_like(input_ids)
        }
        keras_logits = keras_model(keras_inputs)

        import torch
        hf_inputs = {"input_ids": torch.tensor(input_ids)}
        hf_logits = hf_model(**hf_inputs).logits

        keras_logits_np = ops.convert_to_numpy(keras_logits)
        hf_logits_np = hf_logits.detach().cpu().numpy()

        self.assertAllClose(keras_logits_np, hf_logits_np, atol=1e-3, rtol=1e-3)