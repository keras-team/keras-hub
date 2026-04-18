from unittest.mock import patch

import pytest
from keras import ops

from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_backbone import (
    Qwen3_5MoeBackbone,
)
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_causal_lm import (
    Qwen3_5MoeCausalLM,
)
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_causal_lm_preprocessor import (
    Qwen3_5MoeCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_tokenizer import (
    Qwen3_5MoeTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3_5MoeCausalLMTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "\u0120air", "plane", "\u0120at", "port"]
        self.vocab += ["<|endoftext|>"]
        self.vocab += [
            "<|im_start|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ]
        self.vocab += ["<|im_end|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = [
            "\u0120 a",
            "\u0120 t",
            "\u0120 i",
            "\u0120 b",
            "a i",
            "p l",
            "n e",
        ]
        self.merges += [
            "\u0120a t",
            "p o",
            "r t",
            "\u0120t h",
            "ai r",
            "pl a",
            "po rt",
        ]
        self.merges += ["\u0120ai r", "\u0120a i", "pla ne"]
        self.preprocessor = Qwen3_5MoeCausalLMPreprocessor(
            Qwen3_5MoeTokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=7,
        )
        self.backbone = Qwen3_5MoeBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=4,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            head_dim=8,
            moe_intermediate_dim=8,
            shared_expert_intermediate_size=8,
            num_experts=4,
            top_k=2,
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
            router_aux_loss_coefficient=0.01,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Qwen3_5MoeCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 7, 13),
        )

    def test_generate(self):
        causal_lm = Qwen3_5MoeCausalLM(**self.init_kwargs)
        # String input.
        prompt = " airplane at airport"
        output = causal_lm.generate(" airplane at airport")
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        self.assertAllEqual(
            outputs["token_ids"][:, :5],
            prompt_ids["token_ids"][:, :5],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :5],
            prompt_ids["padding_mask"][:, :5],
        )

    def test_generate_strip_prompt(self):
        causal_lm = Qwen3_5MoeCausalLM(**self.init_kwargs)
        prompt = " airplane at airport"
        output = causal_lm.generate(prompt, strip_prompt=True)
        self.assertFalse(output.startswith(prompt))

    def test_early_stopping(self):
        causal_lm = Qwen3_5MoeCausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = [" airplane at airport", " airplane"]
            output = causal_lm.generate(prompt)
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = Qwen3_5MoeCausalLM(**self.init_kwargs)
        causal_lm.generate(" airplane at airport")
        first_fn = causal_lm.generate_function
        causal_lm.generate(" airplane at airport")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3_5MoeCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen3_5MoeCausalLM.presets:
            self.run_preset_test(
                cls=Qwen3_5MoeCausalLM,
                preset=preset,
                input_data=self.input_data,
            )
