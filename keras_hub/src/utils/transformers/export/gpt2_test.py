import os
import shutil
import tempfile

import keras.ops as ops
from absl.testing import parameterized
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.export.hf_exporter import (
    export_to_safetensors,
)


class GPT2ExportTest(TestCase):
    @parameterized.named_parameters(
        ("gpt2_base_en", "gpt2_base_en"),
    )
    def test_gpt2_export(self, preset):
        # Create a temporary directory to save the converted model.
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, preset)

        # Load Keras model.
        keras_model = GPT2CausalLM.from_preset(preset)

        # Export to Hugging Face format.
        export_to_safetensors(keras_model, output_path)

        # Load the converted model with Hugging Face Transformers.
        hf_model = AutoModelForCausalLM.from_pretrained(output_path)
        hf_tokenizer = AutoTokenizer.from_pretrained(output_path)

        # Assertions for config parameters.
        self.assertEqual(
            keras_model.backbone.hidden_dim, hf_model.config.hidden_size
        )
        self.assertEqual(
            keras_model.backbone.num_layers, hf_model.config.n_layer
        )
        self.assertEqual(keras_model.backbone.num_heads, hf_model.config.n_head)
        self.assertEqual(
            keras_model.backbone.intermediate_dim, hf_model.config.n_inner
        )
        self.assertEqual(
            keras_model.backbone.vocabulary_size, hf_model.config.vocab_size
        )
        self.assertEqual(
            keras_model.backbone.max_sequence_length,
            hf_model.config.n_positions,
        )

        # Test logits.
        prompt = "Hello, my name is"
        token_ids = ops.array(keras_model.preprocessor.tokenizer([prompt]))
        padding_mask = ops.ones_like(token_ids, dtype="int32")
        keras_inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
        keras_logits = keras_model(keras_inputs)

        hf_inputs = hf_tokenizer(prompt, return_tensors="pt")
        hf_logits = hf_model(**hf_inputs).logits

        keras_logits_np = ops.convert_to_numpy(keras_logits)
        hf_logits_np = hf_logits.detach().cpu().numpy()

        self.assertAllClose(keras_logits_np, hf_logits_np, atol=1e-3, rtol=1e-3)

        # Clean up the temporary directory.
        shutil.rmtree(temp_dir)