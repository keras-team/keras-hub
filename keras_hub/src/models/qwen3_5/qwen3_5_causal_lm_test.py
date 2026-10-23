import pytest

from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_causal_lm import Qwen3_5CausalLM
from keras_hub.src.models.qwen3_5.qwen3_5_causal_lm_preprocessor import (
    Qwen3_5CausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_5.qwen3_5_tokenizer import Qwen3_5Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3_5CausalLMTest(TestCase):
    def setUp(self):
        self.merges = ["\u0120 a", "\u0120 t", "\u0120 i", "\u0120 b", "a i"]
        self.merges += ["p l", "n e", "\u0120a t", "p o", "r t", "\u0120t h"]
        self.merges += ["ai r", "pl a", "po rt", "\u0120ai r", "\u0120a i"]
        self.merges += ["pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
        self.vocab += ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
        self.vocab += ["<|video_pad|>", "!"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.preprocessor = Qwen3_5CausalLMPreprocessor(
            Qwen3_5Tokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=7,
        )
        self.backbone = Qwen3_5Backbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=4,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            head_dim=8,
            intermediate_dim=16,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            partial_rotary_factor=0.25,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=4,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Qwen3_5CausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 7, 37),
        )

    def test_generate(self):
        self.run_generate_test(
            cls=Qwen3_5CausalLM,
            init_kwargs=self.init_kwargs,
            prompt=" airplane at airport",
        )

    def test_generate_strip_prompt(self):
        causal_lm = Qwen3_5CausalLM(**self.init_kwargs)
        prompt = " airplane at airport"
        output = causal_lm.generate(prompt, strip_prompt=True)
        self.assertFalse(output.startswith(prompt))

    def test_early_stopping(self):
        self.run_early_stopping_test(
            cls=Qwen3_5CausalLM,
            init_kwargs=self.init_kwargs,
            prompt=[" airplane at airport", " airplane"],
        )

    def test_generate_compilation(self):
        self.run_generate_compilation_test(
            cls=Qwen3_5CausalLM,
            init_kwargs=self.init_kwargs,
            prompt=" airplane at airport",
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3_5CausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen3_5CausalLM.presets:
            self.run_preset_test(
                cls=Qwen3_5CausalLM,
                preset=preset,
                input_data=self.input_data,
            )
