from unittest.mock import patch

import pytest
from keras import ops

from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_backbone import (
    Qwen3_5MoeBackbone,
)
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_causal_lm import (
    Qwen3_5MoeCausalLM,
)
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_causal_lm_preprocessor import (  # noqa: E501
    Qwen3_5MoeCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_tokenizer import (
    Qwen3_5MoeTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3_5MoeCausalLMTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
        self.vocab += ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
        self.vocab += ["<|video_pad|>", "!"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
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
            expected_output_shape=(2, 7, 37),
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

    def test_cache_shapes(self):
        """Verify that _build_cache produces caches with correct shapes."""
        causal_lm = Qwen3_5MoeCausalLM(**self.init_kwargs)
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        token_ids = prompt_ids["token_ids"]
        padding_mask = prompt_ids["padding_mask"]

        hidden_states, cache = causal_lm._build_cache(token_ids, padding_mask)
        kv_cache, conv_cache, recurrent_cache = cache

        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        bb = self.backbone

        # KV cache: (batch, num_layers, 2, max_length, num_kv_heads, head_dim)
        expected_kv_shape = (
            batch_size,
            bb.num_layers,
            2,
            max_length,
            bb.num_key_value_heads,
            bb.head_dim,
        )
        self.assertEqual(ops.shape(kv_cache), expected_kv_shape)

        # Conv cache: (batch, num_layers, conv_dim, kernel_dim - 1)
        linear_key_dim = bb.linear_num_key_heads * bb.linear_key_head_dim
        linear_val_dim = bb.linear_num_value_heads * bb.linear_value_head_dim
        conv_dim = linear_key_dim * 2 + linear_val_dim
        expected_conv_shape = (
            batch_size,
            bb.num_layers,
            conv_dim,
            bb.linear_conv_kernel_dim - 1,
        )
        self.assertEqual(ops.shape(conv_cache), expected_conv_shape)

        # Recurrent cache: (batch, num_layers, num_value_heads,
        #                    key_head_dim, value_head_dim)
        expected_recurrent_shape = (
            batch_size,
            bb.num_layers,
            bb.linear_num_value_heads,
            bb.linear_key_head_dim,
            bb.linear_value_head_dim,
        )
        self.assertEqual(ops.shape(recurrent_cache), expected_recurrent_shape)

    def test_cache_output_consistency(self):
        """Verify cached decoding produces correct output shapes."""
        causal_lm = Qwen3_5MoeCausalLM(**self.init_kwargs)
        prompt_ids = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        token_ids = prompt_ids["token_ids"]
        padding_mask = prompt_ids["padding_mask"]

        # Build cache from prompt (this runs the full forward pass).
        hidden_states, cache = causal_lm._build_cache(token_ids, padding_mask)
        kv_cache, conv_cache, recurrent_cache = cache

        # Simulate one autoregressive step (what generate_step does).
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)
        batch_size = ops.shape(token_ids)[0]
        next_token = ops.slice(token_ids, [0, index - 1], [batch_size, 1])
        cached_logits, _, new_cache = causal_lm.call_with_cache(
            next_token, cache, index
        )

        # Verify output shape: (batch_size, 1, vocab_size).
        vocab_size = self.backbone.vocabulary_size
        self.assertEqual(ops.shape(cached_logits), (batch_size, 1, vocab_size))

        # Verify cache structure is preserved after update.
        new_kv, new_conv, new_recurrent = new_cache
        self.assertEqual(ops.shape(new_kv), ops.shape(kv_cache))
        self.assertEqual(ops.shape(new_conv), ops.shape(conv_cache))
        self.assertEqual(ops.shape(new_recurrent), ops.shape(recurrent_cache))

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
