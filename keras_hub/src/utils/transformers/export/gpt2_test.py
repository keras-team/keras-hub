import json
import os

import numpy as np
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer as HFGPT2Tokenizer

from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_hub.src.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.export.hf_exporter import (
    export_to_safetensors,
)


class TestGPT2Export(TestCase):
    def test_export_to_hf(self):
        # 1. Setup Tokenizer Assets
        vocab = {
            "<|endoftext|>": 0,
            "The": 1,
            "Ġquick": 2,
            "Ġbrown": 3,
            "Ġfox": 4,
        }
        # Minimal merges file (required for initialization)
        merges = ["Ġ q", "u i", "c k"]

        temp_dir = self.get_temp_dir()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        merges_path = os.path.join(temp_dir, "merges.txt")

        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        with open(merges_path, "w") as f:
            f.write("\n".join(merges))

        tokenizer = GPT2Tokenizer(vocabulary=vocab_path, merges=merges_path)

        # 2. Create a small backbone (small GPT-2)
        backbone = GPT2Backbone(
            vocabulary_size=len(vocab),
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
            layer_norm_epsilon=1e-5,
            dropout=0,
        )

        # 3. Create preprocessor & model
        preprocessor = GPT2CausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=32
        )
        keras_model = GPT2CausalLM(backbone=backbone, preprocessor=preprocessor)

        # 4. Set Random Weights
        rng = np.random.default_rng(42)
        weights = keras_model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        keras_model.set_weights(weights)

        # 5. Export to Hugging Face format
        export_path = os.path.join(temp_dir, "export_task")
        export_to_safetensors(keras_model, export_path)

        config_path = os.path.join(export_path, "config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)

        # Patch the IDs to match our tiny vocab
        cfg["bos_token_id"] = 0
        cfg["eos_token_id"] = 0

        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # 6. Load with Hugging Face Transformers
        hf_model = GPT2LMHeadModel.from_pretrained(export_path)
        hf_tokenizer = HFGPT2Tokenizer.from_pretrained(export_path)

        # 7. Verify Configuration
        hf_config = hf_model.config

        self.assertEqual(
            hf_config.vocab_size,
            backbone.vocabulary_size,
            "Vocabulary sizes do not match",
        )
        self.assertEqual(
            hf_config.n_layer,
            backbone.num_layers,
            "Number of layers do not match",
        )
        self.assertEqual(
            hf_config.n_head, backbone.num_heads, "Number of heads do not match"
        )
        self.assertEqual(
            hf_config.n_embd,
            backbone.hidden_dim,
            "Hidden dimensions do not match",
        )
        self.assertEqual(
            hf_config.n_inner,
            backbone.intermediate_dim,
            "Intermediate dimensions do not match",
        )

        # 8. Compare Outputs
        prompt = "The quick"

        # Keras Generation
        keras_output = keras_model.generate(prompt, max_length=5)

        # HF Generation
        input_ids = hf_tokenizer.encode(prompt, return_tensors="pt")
        output_ids = hf_model.generate(
            input_ids, max_length=8, do_sample=False, pad_token_id=0
        )
        hf_output = hf_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print(f"Keras output: '{keras_output}'")
        print(f"HF output: '{hf_output}'")

        self.assertEqual(
            keras_output, hf_output, "Generated outputs do not match"
        )
