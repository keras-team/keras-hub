import os

import numpy as np
import torch
from transformers import GemmaForCausalLM
from transformers import GemmaTokenizer

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_hub.src.models.gemma.gemma_tokenizer import (
    GemmaTokenizer as KerasGemmaTokenizer,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.export_gemma_to_safetensor import (
    export_to_hf,
)


class TestGemmaExport(TestCase):
    def test_export_to_hf(self):
        # Load tokenizer from preset
        tokenizer = KerasGemmaTokenizer.from_preset("gemma_2b_en")

        # Create a small backbone
        backbone = GemmaBackbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=512,
            intermediate_dim=2048,
            head_dim=128,
        )
        # Create preprocessor
        preprocessor = GemmaCausalLMPreprocessor(tokenizer=tokenizer)

        # Create the causal LM model
        keras_model = GemmaCausalLM(
            backbone=backbone, preprocessor=preprocessor
        )

        # Set all weights to 1.0
        weights = keras_model.get_weights()
        for i in range(len(weights)):
            weights[i] = np.ones_like(weights[i])
        keras_model.set_weights(weights)

        # Export to Hugging Face format
        export_path = os.path.join(self.get_temp_dir(), "export_small_model")
        export_to_hf(keras_model, export_path)
        # Load Hugging Face model and tokenizer
        hf_model = GemmaForCausalLM.from_pretrained(export_path)
        hf_tokenizer = GemmaTokenizer.from_pretrained(export_path)

        # Verify configuration
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
            "Number of key value heads do not match",
        )
        self.assertEqual(
            hf_config.hidden_size,
            backbone.hidden_dim,
            "Hidden dimensions do not match",
        )
        self.assertEqual(
            hf_config.intermediate_size,
            backbone.intermediate_dim,
            "Intermediate sizes do not match",
        )
        self.assertEqual(
            hf_config.head_dim,
            backbone.head_dim,
            "Head dimensions do not match",
        )
        self.assertEqual(
            hf_config.max_position_embeddings,
            8192,
            "Max position embeddings do not match",
        )

        # Verify key weights are all ones
        state_dict = hf_model.state_dict()
        keys_to_check = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        for key in keys_to_check:
            self.assertTrue(
                torch.all(state_dict[key] == 1.0),
                f"Weight {key} is not all ones",
            )

        # Verify tokenizer compatibility
        self.assertEqual(
            hf_tokenizer.vocab_size,
            tokenizer.vocabulary_size(),
            "Tokenizer vocabulary sizes do not match",
        )

        # Compare generated outputs
        prompt = "All Hail RCB"
        keras_output = keras_model.generate(prompt, max_length=50)
        input_ids = hf_tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = hf_model.generate(
                input_ids, max_length=50, do_sample=False
            )
        hf_output = hf_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        self.assertEqual(
            keras_output, hf_output, "Generated outputs do not match"
        )
