import pytest
from keras import ops

from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.models.llama3.llama3_causal_lm import Llama3CausalLM
from keras_hub.src.models.llama3.llama3_causal_lm_preprocessor import (
    Llama3CausalLMPreprocessor,
)
from keras_hub.src.models.llama3.llama3_tokenizer import Llama3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Llama3CausalLMTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|end_of_text|>", "<|begin_of_text|>"]
        self.vocab += ["<|start_header_id|>", "<|end_header_id|>"]
        self.vocab += ["<|eot_id|>"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.preprocessor = Llama3CausalLMPreprocessor(
            Llama3Tokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=7,
        )
        self.vocabulary_size = self.preprocessor.tokenizer.vocabulary_size()
        self.backbone = Llama3Backbone(
            vocabulary_size=self.vocabulary_size,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Llama3CausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 7, self.vocabulary_size),
        )

    def test_generate(self):
        self.run_generate_test(
            cls=Llama3CausalLM,
            init_kwargs=self.init_kwargs,
            prompt=" airplane at airport",
        )

    def test_generate_strip_prompt(self):
        causal_lm = Llama3CausalLM(**self.init_kwargs)
        prompt = " airplane at airport"
        output = causal_lm.generate(prompt, strip_prompt=True)
        self.assertFalse(output.startswith(prompt))

    def test_early_stopping(self):
        self.run_early_stopping_test(
            cls=Llama3CausalLM,
            init_kwargs=self.init_kwargs,
            prompt=[" airplane at airport", " airplane"],
        )

    def test_generate_compilation(self):
        self.run_generate_compilation_test(
            cls=Llama3CausalLM,
            init_kwargs=self.init_kwargs,
            prompt=" airplane at airport",
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Llama3CausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        """Test LiteRT export for Llama3CausalLM with small test model."""
        model = Llama3CausalLM(**self.init_kwargs)

        # Convert boolean padding_mask to int32 for LiteRT compatibility
        input_data = self.input_data.copy()
        if "padding_mask" in input_data:
            input_data["padding_mask"] = ops.cast(
                input_data["padding_mask"], "int32"
            )

        expected_output_shape = (
            2,
            7,
            self.preprocessor.tokenizer.vocabulary_size(),
        )

        self.run_litert_export_test(
            model=model,
            input_data=input_data,
            expected_output_shape=expected_output_shape,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-3, "mean": 1e-5}},
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Llama3CausalLM.presets:
            self.run_preset_test(
                cls=Llama3CausalLM,
                preset=preset,
                input_data=self.input_data,
            )
