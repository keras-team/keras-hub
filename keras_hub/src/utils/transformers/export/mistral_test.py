import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.models.mistral.mistral_causal_lm import MistralCausalLM
from keras_hub.src.models.mistral.mistral_causal_lm_preprocessor import (
    MistralCausalLMPreprocessor,
)
from keras_hub.src.models.mistral.mistral_tokenizer import MistralTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.export.hf_exporter import (
    export_to_safetensors,
)


class TestMistralExport(TestCase):
    def test_export_to_hf(self):
        # 1. Create tokenizer from test vocab
        proto = os.path.join(self.get_test_data_dir(), "mistral_test_vocab.spm")
        tokenizer = MistralTokenizer(proto=proto)

        # 2. Create a small backbone
        backbone = MistralBackbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            rope_max_wavelength=10000,
            layer_norm_epsilon=1e-6,
            sliding_window=512,
        )

        # 3. Create preprocessor & model
        preprocessor = MistralCausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=16
        )
        keras_model = MistralCausalLM(
            backbone=backbone, preprocessor=preprocessor
        )

        # 4. Set all weights to deterministic random values
        rng = np.random.default_rng(42)
        weights = keras_model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        keras_model.set_weights(weights)

        # 5. Export to Hugging Face format
        export_path = os.path.join(self.get_temp_dir(), "export_task")
        export_to_safetensors(keras_model, export_path)

        # 6. Verify exported files
        exported = os.listdir(export_path)
        self.assertIn("config.json", exported)
        self.assertIn("model.safetensors", exported)
        self.assertIn("tokenizer.model", exported)
        self.assertIn("tokenizer_config.json", exported)

        # 7. Load with Hugging Face Transformers
        # use_fast=False: the tiny test vocab is not a standard BPE/Unigram
        # SentencePiece model, so the fast tokenizer conversion fails.
        hf_tokenizer = AutoTokenizer.from_pretrained(
            export_path, use_fast=False
        )
        hf_model = AutoModelForCausalLM.from_pretrained(export_path)

        # 8. Verify configuration
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
            "Number of key-value heads do not match",
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
            "Layer norm epsilons do not match",
        )
        self.assertEqual(
            hf_config.rope_theta,
            backbone.rope_max_wavelength,
            "RoPE theta values do not match",
        )
        self.assertEqual(
            hf_config.sliding_window,
            backbone.sliding_window,
            "Sliding window values do not match",
        )

        # 9. Test tokenizer functionality
        # Use in-vocabulary text; the tiny test vocab only contains a handful
        # of words so out-of-vocabulary text maps to UNK and HF/KerasHub
        # differ in how they handle UNK (HF strips it, KerasHub keeps it).
        test_text = "the quick brown fox"
        keras_tokens = tokenizer(test_text)
        hf_tokens = hf_tokenizer.encode(test_text, add_special_tokens=False)
        # Compare tokens (handling potential type differences)
        keras_tokens_list = keras_tokens.numpy().tolist()
        self.assertEqual(
            keras_tokens_list,
            hf_tokens,
            "Tokenizer outputs do not match",
        )

        # 10. Test model inference - verify shapes match
        input_ids = torch.tensor([hf_tokens], dtype=torch.long)
        hf_outputs = hf_model(input_ids=input_ids, return_dict=True)

        # Check output shape
        self.assertEqual(
            hf_outputs.logits.shape[-1],
            backbone.vocabulary_size,
            "Output vocabulary size does not match",
        )
