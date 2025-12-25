import os

import numpy as np
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM
from keras_hub.src.models.gemma3.gemma3_causal_lm_preprocessor import (
    Gemma3CausalLMPreprocessor,
)
from keras_hub.src.models.gemma3.gemma3_tokenizer import Gemma3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class TestGemma3Export(TestCase):
    def test_export_to_hf(self):
        proto = os.path.join(self.get_test_data_dir(), "gemma3_test_vocab.spm")
        tokenizer = Gemma3Tokenizer(proto=proto)

        # Create a small backbone (text-only, no vision encoder)
        backbone = Gemma3Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=896,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=128,
            intermediate_dim=256,
            head_dim=64,
            query_head_dim_normalize=True,
            use_query_key_norm=True,
            use_post_ffw_norm=True,
            use_post_attention_norm=True,
            attention_logit_soft_cap=None,
            final_logit_soft_cap=None,
            use_sliding_window_attention=False,
            sliding_window_size=4096,
            vision_encoder=None,  # TODO: enable for vision models
            layer_norm_epsilon=1e-6,
            dropout=0,
        )

        # Create preprocessor
        preprocessor = Gemma3CausalLMPreprocessor(tokenizer=tokenizer)

        # Create the causal LM model
        keras_model = Gemma3CausalLM(
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
        # Note: We only test the slow tokenizer because the test vocab file
        # is not compatible with the fast tokenizer (Unigram vs BPE mismatch).
        # Using fast tokenizer raises: "You're trying to run a `Unigram` model
        # but you're file was trained with a different algorithm"
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
            hf_config.head_dim,
            backbone.head_dim,
            "Head dimensions do not match",
        )
        self.assertEqual(
            hf_config.tie_word_embeddings,
            backbone.token_embedding.tie_weights,
            "Tie word embeddings do not match",
        )

        # Verify tokenizer compatibility (using slow tokenizer)
        self.assertEqual(
            hf_tokenizer_slow.vocab_size,
            tokenizer.vocabulary_size(),
            "Tokenizer vocabulary sizes do not match",
        )

        # Compare generated outputs using full model
        # Test with small input since we set the seed, we expect same outcome
        prompt = "the quick"

        # Generate with Keras model
        keras_output = keras_model.generate(prompt, max_length=20)

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
        print(f"HF slow output: '{hf_slow_output}'")

        self.assertEqual(
            keras_output,
            hf_slow_output,
            "Generated outputs do not match (slow)",
        )
