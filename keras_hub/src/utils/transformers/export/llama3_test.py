import os

import keras.ops as ops
import numpy as np
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM

from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.models.llama3.llama3_causal_lm import Llama3CausalLM
from keras_hub.src.models.llama3.llama3_causal_lm_preprocessor import (
    Llama3CausalLMPreprocessor,
)
from keras_hub.src.models.llama3.llama3_tokenizer import Llama3Tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.export.hf_exporter import (
    export_to_safetensors,
)


class TestLlama3Export(TestCase):
    def setUp(self):
        # Build a complete BPE vocab: base chars + all intermediate merged
        # tokens referenced in the merges list, plus Llama3 special tokens.
        # The tokenizers library validates that every merge component exists
        # in the vocabulary, so all intermediate tokens must be present.
        vocab_tokens = [
            # base characters
            "Ġ",
            "a",
            "t",
            "i",
            "b",
            "p",
            "l",
            "n",
            "e",
            "o",
            "r",
            "h",
            "!",
            # intermediate merged tokens (produced by earlier merges)
            "Ġa",
            "Ġt",
            "Ġi",
            "Ġb",
            "ai",
            "pl",
            "ne",
            "Ġat",
            "po",
            "rt",
            "Ġth",
            "air",
            "pla",
            "port",
            "Ġai",
            "Ġair",
            "plane",
            # Llama3 special tokens
            "<|end_of_text|>",
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ]
        self.vocab = {t: i for i, t in enumerate(vocab_tokens)}
        self.merges = [
            "Ġ a",
            "Ġ t",
            "Ġ i",
            "Ġ b",
            "a i",
            "p l",
            "n e",
            "Ġa t",
            "p o",
            "r t",
            "Ġt h",
            "ai r",
            "pl a",
            "po rt",
            "Ġai r",
            "Ġa i",
            "pla ne",
        ]

    def test_export_to_hf(self):
        # 1. Create tokenizer
        tokenizer = Llama3Tokenizer(vocabulary=self.vocab, merges=self.merges)

        # 2. Create a small backbone
        backbone = Llama3Backbone(
            vocabulary_size=len(self.vocab),
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=64,
            intermediate_dim=128,
            head_dim=32,
        )

        # 3. Create preprocessor & model
        preprocessor = Llama3CausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=16
        )
        keras_model = Llama3CausalLM(
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
        self.assertIn("tokenizer.json", exported)
        self.assertIn("tokenizer_config.json", exported)
        self.assertNotIn("vocabulary.json", exported)
        self.assertNotIn("merges.txt", exported)

        # 7. Load with Hugging Face Transformers
        hf_tokenizer = AutoTokenizer.from_pretrained(export_path)
        hf_model = LlamaForCausalLM.from_pretrained(export_path)

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

        # 9. Verify tokenizer
        self.assertEqual(
            hf_tokenizer.bos_token,
            tokenizer.start_token,
            "BOS tokens do not match",
        )
        self.assertEqual(
            hf_tokenizer.eos_token,
            tokenizer.end_token,
            "EOS tokens do not match",
        )

        # 10. Compare logits between Keras and HuggingFace models
        # Use token IDs within the vocab range
        input_ids = np.array([[13, 14, 15, 0]], dtype=np.int32)

        keras_inputs = {
            "token_ids": input_ids,
            "padding_mask": np.ones_like(input_ids),
        }
        keras_logits = keras_model(keras_inputs)

        import torch

        hf_inputs = {"input_ids": torch.tensor(input_ids)}
        hf_logits = hf_model(**hf_inputs).logits

        keras_logits_np = ops.convert_to_numpy(keras_logits)
        hf_logits_np = hf_logits.detach().cpu().numpy()

        self.assertAllClose(keras_logits_np, hf_logits_np, atol=1e-3, rtol=1e-3)
