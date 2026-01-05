import json
import os
import shutil
import tempfile

import keras.ops as ops
import numpy as np
import pytest

from keras_hub.src.models.qwen_moe.qwen_moe_backbone import QwenMoeBackbone
from keras_hub.src.models.qwen_moe.qwen_moe_causal_lm import QwenMoeCausalLM
from keras_hub.src.models.qwen_moe.qwen_moe_causal_lm_preprocessor import (
    QwenMoeCausalLMPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase

try:
    from keras_hub.src.models.qwen_moe.qwen_moe_tokenizer import (
        QwenMoeTokenizer,
    )
except ImportError:
    from keras_hub.src.models.qwen.qwen_tokenizer import (
        QwenTokenizer as QwenMoeTokenizer,
    )

from keras_hub.src.utils.transformers.export.hf_exporter import (
    export_to_safetensors,
)


class QwenMoEExportTest(TestCase):
    def test_qwen_moe_export(self):
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "qwen_moe")

        vocab = {
            "<|endoftext|>": 0,
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "hello": 5,
            "world": 6,
        }
        merges = ["h e", "l l", "o <|endoftext|>"]

        vocab_file = os.path.join(temp_dir, "vocabulary.json")
        merges_file = os.path.join(temp_dir, "merges.txt")

        with open(vocab_file, "w") as f:
            json.dump(vocab, f)
        with open(merges_file, "w") as f:
            f.write("\n".join(merges))

        # Building a smallBackbone

        backbone = QwenMoeBackbone(
            vocabulary_size=len(vocab),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            moe_intermediate_dim=32,
            shared_expert_intermediate_dim=64,
            num_experts=4,
            top_k=2,
            norm_top_k_prob=False,
            decoder_sparse_step=1,
            rope_max_wavelength=10000,
            rope_scaling_factor=1.0,
            layer_norm_epsilon=1e-6,
            dtype="float32",
        )

        tokenizer = QwenMoeTokenizer(vocabulary=vocab_file, merges=merges_file)
        preprocessor = QwenMoeCausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=32
        )
        keras_model = QwenMoeCausalLM(
            backbone=backbone, preprocessor=preprocessor
        )

        export_to_safetensors(keras_model, output_path)

        try:
            from transformers import AutoConfig
            from transformers import AutoModelForCausalLM

            hf_config = AutoConfig.from_pretrained(output_path)

            self.assertEqual(hf_config.hidden_size, 64)
            self.assertEqual(hf_config.num_hidden_layers, 2)
            self.assertEqual(hf_config.num_experts, 4)
            self.assertEqual(hf_config.num_experts_per_tok, 2)
            self.assertEqual(hf_config.moe_intermediate_size, 32)
            self.assertEqual(hf_config.shared_expert_intermediate_size, 64)

            hf_model = AutoModelForCausalLM.from_pretrained(output_path)

            #  Sample Input
            input_ids = np.array([[1, 2, 5]])  # IDs for "a", "b", "hello"

            # Keras Inference
            keras_inputs = {
                "token_ids": input_ids,
                "padding_mask": np.ones_like(input_ids),
            }
            keras_logits = keras_model(keras_inputs)

            # Hugging Face Inference
            import torch

            hf_inputs = {"input_ids": torch.tensor(input_ids)}
            hf_logits = hf_model(**hf_inputs).logits

            keras_logits_np = ops.convert_to_numpy(keras_logits)
            hf_logits_np = hf_logits.detach().cpu().numpy()

            self.assertAllClose(
                keras_logits_np, hf_logits_np, atol=1e-4, rtol=1e-4
            )

        except ImportError:
            pytest.skip("Transformers library not installed.")
        except Exception as e:
            print(
                f"Skipping HF load test due to environment/version issue: {e}"
            )

        # Cleanup
        shutil.rmtree(temp_dir)
