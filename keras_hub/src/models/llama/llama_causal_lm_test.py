import os
from unittest.mock import patch

import keras
import pytest
from keras import ops

from keras_hub.src.kv_press.knorm_press import KnormPress
from keras_hub.src.kv_press.random_press import RandomPress
from keras_hub.src.kv_press.streaming_llm_press import StreamingLLMPress
from keras_hub.src.models.llama.llama_backbone import LlamaBackbone
from keras_hub.src.models.llama.llama_causal_lm import LlamaCausalLM
from keras_hub.src.models.llama.llama_causal_lm_preprocessor import (
    LlamaCausalLMPreprocessor,
)
from keras_hub.src.models.llama.llama_tokenizer import LlamaTokenizer
from keras_hub.src.tests.test_case import TestCase


class LlamaCausalLMTest(TestCase):
    def setUp(self):
        self.preprocessor = LlamaCausalLMPreprocessor(
            LlamaTokenizer(
                # Generated using create_llama_test_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "llama_test_vocab.spm"
                )
            ),
            sequence_length=8,
        )
        self.backbone = LlamaBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
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
        self.train_data = (["the quick brown fox", "the earth is round"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=LlamaCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, 10),
        )

    def test_generate(self):
        causal_lm = LlamaCausalLM(**self.init_kwargs)
        # String input.
        prompt = "the quick brown fox"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["token_ids"][:, :5],
            prompt_ids["token_ids"][:, :5],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :5],
            prompt_ids["padding_mask"][:, :5],
        )

    def test_early_stopping(self):
        causal_lm = LlamaCausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["the quick brown fox", "the earth"]
            output = causal_lm.generate(prompt)
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = LlamaCausalLM(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate("the quick brown fox")
        first_fn = causal_lm.generate_function
        causal_lm.generate("the quick brown fox")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    def test_call_with_cache_position_index_decoupling(self):
        # `position_index` (used for rotary embeddings) must be decoupled
        # from `cache_update_index` (the physical cache write slot / causal
        # mask offset) -- this is what lets a compressed, shorter-than-
        # original cache still assign tokens their true original position.
        causal_lm = LlamaCausalLM(**self.init_kwargs)
        prompt = "the quick brown fox"
        preprocessed = self.preprocessor.generate_preprocess([prompt])
        token_ids = preprocessed["token_ids"]
        padding_mask = preprocessed["padding_mask"]
        hidden_states, cache, cache_index_offset = causal_lm._build_cache(
            token_ids, padding_mask
        )
        self.assertEqual(cache_index_offset, 0)
        next_token = ops.slice(token_ids, [0, 0], [1, 1])

        # Same `cache_update_index` both times, but different
        # `position_index` -- isolates the effect of the rotary embedding
        # from cache addressing/masking.
        _, hidden_a, _ = causal_lm.call_with_cache(
            next_token, cache, 2, position_index=2
        )
        _, hidden_b, _ = causal_lm.call_with_cache(
            next_token, cache, 2, position_index=5
        )
        self.assertNotAllClose(hidden_a, hidden_b)

        # Omitting `position_index` falls back to `cache_update_index`,
        # matching pre-compression behavior exactly.
        _, hidden_c, _ = causal_lm.call_with_cache(next_token, cache, 2)
        self.assertAllClose(hidden_a, hidden_c)

    def test_kv_cache_compression_shrinks_cache_and_offsets_index(self):
        if keras.config.backend() != "torch":
            self.skipTest("Compression during generate() is torch-only.")
        causal_lm = LlamaCausalLM(**self.init_kwargs)
        causal_lm.compile(press=KnormPress(compression_ratio=0.5))
        prompt = "the quick brown fox"
        preprocessed = self.preprocessor.generate_preprocess([prompt])
        token_ids = preprocessed["token_ids"]
        padding_mask = preprocessed["padding_mask"]
        max_length = int(token_ids.shape[1])

        _, cache, cache_index_offset = causal_lm._build_cache(
            token_ids, padding_mask
        )
        real_length = int(ops.sum(ops.cast(padding_mask, "int32")))
        expected_keep_len = max(1, round(real_length * 0.5))
        # The buffer keeps `expected_keep_len` prompt slots *plus* one free
        # slot per token generation is about to write -- without that
        # reserve the decode loop would overwrite what was just retained.
        expected_reserve = max_length - real_length
        self.assertEqual(cache.shape[3], expected_keep_len + expected_reserve)
        self.assertEqual(cache_index_offset, real_length - expected_keep_len)

    def test_press_string_identifier(self):
        causal_lm = LlamaCausalLM(**self.init_kwargs)
        causal_lm.compile(press="knorm")
        self.assertIsInstance(causal_lm.press, KnormPress)

    def test_generate_with_kv_cache_compression_ratio_zero_matches_baseline(
        self,
    ):
        # A `compression_ratio=0.0` press is a true no-op end-to-end, so
        # generation must match the uncompressed baseline exactly. The two
        # models share the same (already-built) backbone, so weights match.
        baseline = LlamaCausalLM(**self.init_kwargs)
        baseline.compile(sampler="greedy")
        compressed = LlamaCausalLM(**self.init_kwargs)
        compressed.compile(
            sampler="greedy",
            press=StreamingLLMPress(compression_ratio=0.0),
        )

        prompt = "the quick brown fox"
        baseline_output = baseline.generate(prompt, stop_token_ids=None)
        compressed_output = compressed.generate(prompt, stop_token_ids=None)
        self.assertEqual(baseline_output, compressed_output)

    def test_generate_with_kv_cache_compression(self):
        if keras.config.backend() != "torch":
            self.skipTest("Compression during generate() is torch-only.")
        for press in (
            RandomPress(compression_ratio=0.5),
            KnormPress(compression_ratio=0.5),
            StreamingLLMPress(compression_ratio=0.5, n_sink=1),
        ):
            causal_lm = LlamaCausalLM(**self.init_kwargs)
            causal_lm.compile(sampler="greedy", press=press)
            prompt = "the quick brown fox"
            output = causal_lm.generate(prompt, stop_token_ids=None)
            self.assertTrue(prompt in output)

    def test_generate_with_kv_cache_compression_mixed_length_batch(self):
        if keras.config.backend() != "torch":
            self.skipTest("Compression during generate() is torch-only.")
        causal_lm = LlamaCausalLM(**self.init_kwargs)
        causal_lm.compile(
            sampler="greedy",
            press=StreamingLLMPress(compression_ratio=0.5, n_sink=1),
        )
        prompts = ["the quick brown fox", "the earth"]
        output = causal_lm.generate(prompts, stop_token_ids=None)
        for prompt, generated in zip(prompts, output):
            self.assertTrue(prompt in generated)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=LlamaCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=LlamaCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in LlamaCausalLM.presets:
            self.run_preset_test(
                cls=LlamaCausalLM,
                preset=preset,
                input_data=self.input_data,
            )

    def test_score_logits(self):
        # Setup prompts, models, and associated expected shapes.
        prompts = ["the quick brown fox", "the quick brown fox"]
        causal_lm = LlamaCausalLM(**self.init_kwargs)
        expected_score_shape = (2, 8, 10)

        # Preprocess prompts to get tokenized representations and padding masks.
        preprocessed_prompts = causal_lm.preprocessor.generate_preprocess(
            prompts
        )
        token_ids = preprocessed_prompts["token_ids"]
        padding_mask = preprocessed_prompts["padding_mask"]

        # Get the scores and assert their shape.
        scores = causal_lm.score(
            token_ids=token_ids,
            padding_mask=padding_mask,
            scoring_mode="logits",
        )

        self.assertEqual(ops.shape(scores), expected_score_shape)

    def test_score_loss(self):
        # Setup prompts, models, and associated expected shapes.
        prompts = ["the quick brown fox", "the quick brown fox"]
        causal_lm = LlamaCausalLM(**self.init_kwargs)
        expected_score_shape = (2, 8)

        # Preprocess prompts to get tokenized representations and padding masks.
        preprocessed_prompts = causal_lm.preprocessor.generate_preprocess(
            prompts
        )
        token_ids = preprocessed_prompts["token_ids"]
        padding_mask = preprocessed_prompts["padding_mask"]
        target_ids = ops.roll(token_ids, shift=-1, axis=1)

        # Get the scores and assert their shape.
        scores = causal_lm.score(
            token_ids=token_ids,
            padding_mask=padding_mask,
            scoring_mode="loss",
            target_ids=target_ids,
        )

        self.assertEqual(ops.shape(scores), expected_score_shape)

    def test_score_layer_intercept_fn_exfiltration(self):
        # Setup prompts, models, and associated expected shapes.
        prompts = ["the quick brown fox", "the quick brown fox"]
        causal_lm = LlamaCausalLM(**self.init_kwargs)
        expected_embedded_shape = (2, 8, 8)
        expected_score_shape = (2, 8, 10)

        # Preprocess prompts to get tokenized representations and padding masks.
        preprocessed_prompts = causal_lm.preprocessor.generate_preprocess(
            prompts
        )
        token_ids = preprocessed_prompts["token_ids"]
        padding_mask = preprocessed_prompts["padding_mask"]

        # Setup a custom intercept function that extracts the embeddings to a
        # a variable from the embeddings layer and otherwise asserts on shapes.
        embedded_prompts = None

        def layer_intercept_fn_for_testing(x, i):
            if i == -1:
                nonlocal embedded_prompts
                embedded_prompts = x
            else:
                nonlocal expected_embedded_shape
                self.assertEqual(ops.shape(x), expected_embedded_shape)
            return x

        # Get the scores.
        scores = causal_lm.score(
            token_ids=token_ids,
            padding_mask=padding_mask,
            scoring_mode="logits",
            layer_intercept_fn=layer_intercept_fn_for_testing,
        )

        # Assert shapes for info exfiltrated into the parent context.
        self.assertEqual(ops.shape(embedded_prompts), expected_embedded_shape)
        self.assertEqual(ops.shape(scores), expected_score_shape)
