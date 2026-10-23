import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import pytest

from keras_hub.src.models.qwen3_moe.qwen3_moe_backbone import Qwen3MoeBackbone
from keras_hub.src.models.qwen3_moe.qwen3_moe_causal_lm import Qwen3MoeCausalLM
from keras_hub.src.models.qwen3_moe.qwen3_moe_causal_lm_preprocessor import (
    Qwen3MoeCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_moe.qwen3_moe_tokenizer import Qwen3MoeTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3MoeCausalLMTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<|im_end|>", "<|endoftext|>", "!"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.preprocessor = Qwen3MoeCausalLMPreprocessor(
            Qwen3MoeTokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=7,
        )
        self.vocabulary_size = self.preprocessor.tokenizer.vocabulary_size()
        self.backbone = Qwen3MoeBackbone(
            vocabulary_size=self.vocabulary_size,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            moe_intermediate_dim=4,
            head_dim=2,
            # shared_expert_intermediate_dim=16,
            num_experts=4,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Qwen3MoeCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 7, self.vocabulary_size),
        )

    def test_generate(self):
        self.run_generate_test(
            cls=Qwen3MoeCausalLM,
            init_kwargs=self.init_kwargs,
            prompt=" airplane at airport",
        )

    def test_generate_strip_prompt(self):
        causal_lm = Qwen3MoeCausalLM(**self.init_kwargs)
        prompt = " airplane at airport"
        output = causal_lm.generate(prompt, strip_prompt=True)
        self.assertFalse(output.startswith(prompt))

    def test_early_stopping(self):
        self.run_early_stopping_test(
            cls=Qwen3MoeCausalLM,
            init_kwargs=self.init_kwargs,
            prompt=[" airplane at airport", " airplane"],
        )

    def test_generate_compilation(self):
        self.run_generate_compilation_test(
            cls=Qwen3MoeCausalLM,
            init_kwargs=self.init_kwargs,
            prompt=" airplane at airport",
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3MoeCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.xfail(
        condition=keras.backend.backend() == "torch",
        reason="litert-torch cannot lower aten._assert_async from MoE routing.",
    )
    def test_litert_export(self):
        self.run_litert_export_test(
            cls=Qwen3MoeCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen3MoeCausalLM.presets:
            self.run_preset_test(
                cls=Qwen3MoeCausalLM,
                preset=preset,
                input_data=self.input_data,
            )
