import os

import numpy as np
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from keras_hub.src.models.llama.llama_backbone import LlamaBackbone
from keras_hub.src.models.llama.llama_causal_lm import LlamaCausalLM
from keras_hub.src.models.llama.llama_causal_lm_preprocessor import (
    LlamaCausalLMPreprocessor,
)
from keras_hub.src.models.llama.llama_tokenizer import LlamaTokenizer
from keras_hub.src.tests.test_case import TestCase


class TestLlamaExport(TestCase):
    def test_export_to_hf(self):
        proto = os.path.join(
            os.path.dirname(__file__),
            "../../../tests/test_data",
            "llama_export_vocab.spm",
        )
        tokenizer = LlamaTokenizer(proto=proto)
        # Create a small backbone
        backbone = LlamaBackbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=512,
            intermediate_dim=1024,
            head_dim=128,
            layer_norm_epsilon=1e-6,
            rope_max_wavelength=10000,
        )
        # Create preprocessor
        preprocessor = LlamaCausalLMPreprocessor(tokenizer=tokenizer)
        # Create the causal LM model
        keras_model = LlamaCausalLM(
            backbone=backbone, preprocessor=preprocessor
        )
        # Set all weights to random values
        rng = np.random.default_rng(42)
        weights = keras_model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        keras_model.set_weights(weights)
        # Export to Hugging Face format using the new methods
        export_path_backbone = os.path.join(
            self.get_temp_dir(), "export_backbone"
        )
        backbone.export_to_transformers(export_path_backbone)
        export_path_tokenizer = os.path.join(
            self.get_temp_dir(), "export_tokenizer"
        )
        preprocessor.tokenizer.export_to_transformers(export_path_tokenizer)
        export_path_task = os.path.join(self.get_temp_dir(), "export_task")
        keras_model.export_to_transformers(export_path_task)
        # Load Hugging Face models and tokenizer
        hf_backbone = AutoModel.from_pretrained(export_path_backbone)
        hf_tokenizer_fast = AutoTokenizer.from_pretrained(export_path_tokenizer)
        hf_tokenizer_slow = AutoTokenizer.from_pretrained(
            export_path_tokenizer, use_fast=False
        )
        hf_full_model = AutoModelForCausalLM.from_pretrained(export_path_task)
        # Verify configuration
        hf_config = hf_backbone.config
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
            hf_config.max_position_embeddings,
            4096,
            "Max position embeddings do not match",
        )
        self.assertEqual(
            hf_config.rms_norm_eps,
            backbone.layer_norm_epsilon,
            "RMS norm epsilon does not match",
        )
        self.assertEqual(
            hf_config.rope_theta,
            backbone.rope_max_wavelength,
            "RoPE theta does not match",
        )
        # Verify tokenizer compatibility
        self.assertEqual(
            hf_tokenizer_fast.vocab_size,
            tokenizer.vocabulary_size(),
            "Tokenizer vocabulary sizes do not match",
        )
        # Compare generated outputs using full model
        prompt = "the quick"
        # Set seed for reproducibility
        rng = np.random.default_rng(42)
        # Generate with Keras model
        keras_output = keras_model.generate(prompt, max_length=20)
        # Generate with HuggingFace model using fast tokenizer
        input_ids_fast = hf_tokenizer_fast.encode(prompt, return_tensors="pt")
        output_ids_fast = hf_full_model.generate(
            input_ids_fast, max_length=20, do_sample=False
        )
        hf_fast_output = hf_tokenizer_fast.decode(
            output_ids_fast[0], skip_special_tokens=True
        )
        # Generate with HuggingFace model using slow tokenizer
        input_ids_slow = hf_tokenizer_slow.encode(prompt, return_tensors="pt")
        output_ids_slow = hf_full_model.generate(
            input_ids_slow, max_length=20, do_sample=False
        )
        hf_slow_output = hf_tokenizer_slow.decode(
            output_ids_slow[0], skip_special_tokens=True
        )
        # Debug print to see the actual outputs
        print(f"Keras output: '{keras_output}'")
        print(f"HF fast output: '{hf_fast_output}'")
        print(f"HF slow output: '{hf_slow_output}'")
        self.assertEqual(
            keras_output,
            hf_fast_output,
            "Generated outputs do not match (fast)",
        )
        self.assertEqual(
            keras_output,
            hf_slow_output,
            "Generated outputs do not match (slow)",
        )
