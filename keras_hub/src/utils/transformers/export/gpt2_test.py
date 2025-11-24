import os
import shutil
import sys
import tempfile
from os.path import abspath
from os.path import dirname

# import keras
import numpy as np
import tensorflow as tf

# import torch
from absl.testing import parameterized
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# Add the project root to the Python path.
sys.path.insert(
    0, dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
)

from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.utils.transformers.export.hf_exporter import (
    export_to_safetensors,
)


def to_numpy(x):
    # Torch tensor
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()

    # TF tensor
    if hasattr(x, "numpy"):
        return x.numpy()

    # Numpy
    if isinstance(x, np.ndarray):
        return x

    # KerasTensor or ragged wrapper → convert to TF → numpy
    try:
        import tensorflow as tf

        return tf.convert_to_tensor(x).numpy()
    except Exception:
        pass

    raise TypeError(f"Cannot convert value of type {type(x)} to numpy")


class GPT2ExportTest(tf.test.TestCase, parameterized.TestCase):
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
        token_ids = tf.constant(
            keras_model.preprocessor.tokenizer(tf.constant([prompt]))
        )
        padding_mask = tf.ones_like(token_ids, dtype=tf.int32)
        keras_inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
        keras_logits = keras_model(keras_inputs)

        hf_inputs = hf_tokenizer(prompt, return_tensors="pt")
        hf_logits = hf_model(**hf_inputs).logits
        print(hf_logits)

        # Compare logits.
        # Keras logits are (batch_size, sequence_length, vocab_size)
        # HF logits are (batch_size, sequence_length, vocab_size)
        # We need to convert Keras logits to numpy and then to torch tensor
        # for comparison.
        # Convert Keras logits (TF) -> numpy
        keras_logits_np = to_numpy(keras_logits)

        # Convert HF logits (Torch, possibly MPS) -> numpy
        hf_logits_np = to_numpy(hf_logits)

        self.assertAllClose(keras_logits_np, hf_logits_np, atol=1e-3, rtol=1e-3)

        # Clean up the temporary directory.
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    tf.test.main()
