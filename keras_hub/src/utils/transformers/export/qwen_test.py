import os
import shutil
import tempfile

import keras.ops as ops
from absl.testing import parameterized
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from keras_hub.src.models.qwen.qwen_causal_lm import QwenCausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.export.hf_exporter import (
    export_to_safetensors,
)


class QwenExportTest(TestCase):
    @parameterized.named_parameters(
        # Use a small preset for testing
        ("qwen2_0.5b_en", "qwen2.5_0.5b_en"),
    )
    def test_qwen_export(self, preset):
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, preset)

        # 1. Load Keras Model
        keras_model = QwenCausalLM.from_preset(preset)

        # 2. Export
        export_to_safetensors(keras_model, output_path)

        # 3. Load HF Model
        hf_model = AutoModelForCausalLM.from_pretrained(output_path)
        hf_tokenizer = AutoTokenizer.from_pretrained(output_path)

        # 4. Verify Config
        self.assertEqual(
            keras_model.backbone.hidden_dim, hf_model.config.hidden_size
        )
        self.assertEqual(
            keras_model.backbone.num_layers, hf_model.config.num_hidden_layers
        )
        self.assertEqual(
            keras_model.backbone.vocabulary_size, hf_model.config.vocab_size
        )

        # 5. Verify Outputs (Logits)
        prompt = "Hello, world!"
        token_ids = ops.array(keras_model.preprocessor.tokenizer([prompt]))
        padding_mask = ops.ones_like(token_ids, dtype="int32")

        keras_inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
        keras_logits = keras_model(keras_inputs)

        hf_inputs = hf_tokenizer(prompt, return_tensors="pt")
        hf_logits = hf_model(**hf_inputs).logits

        keras_logits_np = ops.convert_to_numpy(keras_logits)
        hf_logits_np = hf_logits.detach().cpu().numpy()

        # Higher tolerance might be needed due to RoPE/Transposition differences
        self.assertAllClose(keras_logits_np, hf_logits_np, atol=1e-3, rtol=1e-3)

        shutil.rmtree(temp_dir)
