import pytest
import torch
from transformers import GemmaForCausalLM, GemmaTokenizer
from keras_hub.src.utils.transformers.export_gemma_to_safetensor import export_to_hf
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM

class TestGemmaExport(TestCase):
    @pytest.fixture(autouse=True)
    def setup_tmp_path(self, tmp_path):
        """Set up the tmp_path fixture as an instance attribute before each test."""
        self.tmp_path = tmp_path

    @pytest.mark.large
    def test_export_to_hf(self):
        # Load Keras model
        keras_model = GemmaCausalLM.from_preset("gemma_2b_en")
        input_text = "All hail RCB"
        max_length = 25

        # Export to Hugging Face format using self.tmp_path
        export_path = self.tmp_path / "export_to_hf"
        export_to_hf(keras_model, export_path)

        # Load Hugging Face model and tokenizer
        hf_model = GemmaForCausalLM.from_pretrained(export_path)
        hf_tokenizer = GemmaTokenizer.from_pretrained(export_path)

        # Generate text with Keras model
        keras_output = keras_model.generate(input_text, max_length=max_length)

        # Generate text with Hugging Face model
        hf_inputs = hf_tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            hf_outputs = hf_model.generate(**hf_inputs, max_length=max_length, do_sample=False)
        hf_output_text = hf_tokenizer.decode(hf_outputs[0], skip_special_tokens=True)

        self.assertEqual(keras_output, hf_output_text)