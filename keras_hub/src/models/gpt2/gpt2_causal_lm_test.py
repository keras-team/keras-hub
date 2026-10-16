from unittest.mock import patch

import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.kv_press.knorm_press import KnormPress
from keras_hub.src.kv_press.random_press import RandomPress
from keras_hub.src.kv_press.streaming_llm_press import StreamingLLMPress
from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_hub.src.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_hub.src.tests.test_case import TestCase


class GPT2CausalLMTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|endoftext|>"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.preprocessor = GPT2CausalLMPreprocessor(
            GPT2Tokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=8,
        )
        self.vocabulary_size = self.preprocessor.tokenizer.vocabulary_size()
        self.backbone = GPT2Backbone(
            vocabulary_size=self.vocabulary_size,
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            max_sequence_length=self.preprocessor.sequence_length,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=GPT2CausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, self.vocabulary_size),
        )

    def test_generate(self):
        causal_lm = GPT2CausalLM(**self.init_kwargs)
        # String input.
        prompt = " airplane at airport"
        output = causal_lm.generate(" airplane at airport")
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
        causal_lm = GPT2CausalLM(**self.init_kwargs)
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
            prompt = [" airplane at airport", " airplane"]
            output = causal_lm.generate(prompt)
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = GPT2CausalLM(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate(" airplane at airport")
        first_fn = causal_lm.generate_function
        causal_lm.generate(" airplane at airport")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    def test_call_with_cache_position_index_decoupling(self):
        # `position_index` (used for position embeddings) must be decoupled
        # from `cache_update_index` (the physical cache write slot / causal
        # mask offset) -- this is what lets a compressed, shorter-than-
        # original cache still assign tokens their true original position.
        causal_lm = GPT2CausalLM(**self.init_kwargs)
        prompt = " airplane at airport"
        preprocessed = self.preprocessor.generate_preprocess([prompt])
        token_ids = preprocessed["token_ids"]
        padding_mask = preprocessed["padding_mask"]
        hidden_states, cache, cache_index_offset = causal_lm._build_cache(
            token_ids, padding_mask
        )
        self.assertEqual(cache_index_offset, 0)
        next_token = ops.slice(token_ids, [0, 0], [1, 1])

        # Same `cache_update_index` both times, but different
        # `position_index` -- isolates the effect of the position embedding
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
        causal_lm = GPT2CausalLM(**self.init_kwargs)
        causal_lm.compile(press=KnormPress(compression_ratio=0.5))
        prompt = " airplane at airport"
        preprocessed = self.preprocessor.generate_preprocess([prompt])
        token_ids = preprocessed["token_ids"]
        padding_mask = preprocessed["padding_mask"]
        max_length = int(token_ids.shape[1])

        _, cache, cache_index_offset = causal_lm._build_cache(
            token_ids, padding_mask
        )
        prompt_length = int(ops.sum(ops.cast(padding_mask, "int32")))
        expected_keep_len = max(1, round(prompt_length * 0.5))
        # The buffer keeps `expected_keep_len` prompt slots *plus* one free
        # slot per token generation is about to write -- without that
        # reserve the decode loop would overwrite what was just retained.
        expected_reserve = max_length - prompt_length
        self.assertEqual(cache.shape[3], expected_keep_len + expected_reserve)
        # The last prompt token maps to the final retained slot, so decoding
        # continues into the reserve rather than back over the prompt.
        self.assertEqual(cache_index_offset, prompt_length - expected_keep_len)
        self.assertEqual(
            (prompt_length - 1) - cache_index_offset, expected_keep_len - 1
        )
        # The very last decode step must still land inside the buffer.
        self.assertLess((max_length - 1) - cache_index_offset, cache.shape[3])

    def test_kv_cache_compression_retains_prompt_through_decoding(self):
        # Regression test for the retained prompt being silently clobbered:
        # every retained slot must still be read by the decode loop. If the
        # slot mapping is off, corrupting them leaves the logits untouched.
        if keras.config.backend() != "torch":
            self.skipTest("Compression during generate() is torch-only.")
        prompt = " airplane at airport"
        preprocessed = self.preprocessor.generate_preprocess([prompt])
        padding_mask = preprocessed["padding_mask"]
        prompt_length = int(ops.sum(ops.cast(padding_mask, "int32")))

        def logits_with_corrupted_prompt_cache(corrupt):
            causal_lm = GPT2CausalLM(**self.init_kwargs)
            causal_lm.compile(
                sampler="greedy", press=KnormPress(compression_ratio=0.5)
            )
            keep_len = max(1, round(prompt_length * 0.5))
            seen = []
            original = causal_lm._compress_layer_cache

            def compress_layer_cache(layer_cache, padding_mask=None):
                layer_cache = original(layer_cache, padding_mask)
                if corrupt:
                    corrupted = ops.convert_to_numpy(layer_cache).copy()
                    corrupted[:, :, :keep_len, :, :] = 37.0
                    layer_cache = ops.convert_to_tensor(corrupted)
                return layer_cache

            call_with_cache = causal_lm.call_with_cache

            def spy(
                token_ids,
                cache,
                cache_update_index,
                position_index=None,
                compress_cache=False,
                padding_mask=None,
            ):
                out = call_with_cache(
                    token_ids,
                    cache,
                    cache_update_index,
                    position_index,
                    compress_cache=compress_cache,
                    padding_mask=padding_mask,
                )
                seen.append(ops.convert_to_numpy(out[0]).copy())
                return out

            causal_lm._compress_layer_cache = compress_layer_cache
            causal_lm.call_with_cache = spy
            causal_lm.generate([prompt], stop_token_ids=None)
            return np.concatenate(seen[1:], axis=1)

        clean = logits_with_corrupted_prompt_cache(corrupt=False)
        corrupted = logits_with_corrupted_prompt_cache(corrupt=True)
        self.assertGreater(np.abs(clean - corrupted).max(), 0.0)

    def test_press_string_identifier(self):
        causal_lm = GPT2CausalLM(**self.init_kwargs)
        causal_lm.compile(press="knorm")
        self.assertIsInstance(causal_lm.press, KnormPress)

    def test_generate_with_kv_cache_compression_ratio_zero_matches_baseline(
        self,
    ):
        # A `compression_ratio=0.0` press is a true no-op end-to-end, so
        # generation must match the uncompressed baseline exactly. The two
        # models share the same (already-built) backbone, so weights match.
        baseline = GPT2CausalLM(**self.init_kwargs)
        baseline.compile(sampler="greedy")
        compressed = GPT2CausalLM(**self.init_kwargs)
        compressed.compile(
            sampler="greedy",
            press=StreamingLLMPress(compression_ratio=0.0),
        )

        prompt = " airplane at airport"
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
            causal_lm = GPT2CausalLM(**self.init_kwargs)
            causal_lm.compile(sampler="greedy", press=press)
            prompt = " airplane at airport"
            output = causal_lm.generate(prompt, stop_token_ids=None)
            self.assertTrue(prompt in output)

    def test_generate_with_kv_cache_compression_mixed_length_batch(self):
        if keras.config.backend() != "torch":
            self.skipTest("Compression during generate() is torch-only.")
        causal_lm = GPT2CausalLM(**self.init_kwargs)
        causal_lm.compile(
            sampler="greedy",
            press=StreamingLLMPress(compression_ratio=0.5, n_sink=1),
        )
        prompts = [" airplane at airport", " airplane"]
        output = causal_lm.generate(prompts, stop_token_ids=None)
        for prompt, generated in zip(prompts, output):
            self.assertTrue(prompt in generated)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=GPT2CausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        """Test LiteRT export for GPT2CausalLM with small test model."""
        model = GPT2CausalLM(**self.init_kwargs)

        # Convert boolean padding_mask to int32 for LiteRT compatibility
        input_data = self.input_data.copy()
        if "padding_mask" in input_data:
            input_data["padding_mask"] = ops.cast(
                input_data["padding_mask"], "int32"
            )

        expected_output_shape = (
            2,
            8,
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
        for preset in GPT2CausalLM.presets:
            self.run_preset_test(
                cls=GPT2CausalLM,
                preset=preset,
                input_data=self.input_data,
            )

    def test_score_logits(self):
        # Setup prompts, models, and associated expected shapes.
        prompts = [" airplane at airport", " airplane at airport"]
        causal_lm = GPT2CausalLM(**self.init_kwargs)
        expected_score_shape = (2, 8, self.vocabulary_size)

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
        prompts = [" airplane at airport", " airplane at airport"]
        causal_lm = GPT2CausalLM(**self.init_kwargs)
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
        prompts = [" airplane at airport", " airplane at airport"]
        causal_lm = GPT2CausalLM(**self.init_kwargs)
        expected_embedded_shape = (2, 8, 4)
        expected_score_shape = (2, 8, self.vocabulary_size)

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

    def test_get_quantization_layer_structure(self):
        causal_lm = GPT2CausalLM(**self.init_kwargs)
        structure = causal_lm.get_quantization_layer_structure("gptq")
        self.assertIsInstance(structure, dict)
        self.assertIn("pre_block_layers", structure)
        self.assertIn("sequential_blocks", structure)
        self.assertLen(structure["pre_block_layers"], 1)
        self.assertIsInstance(structure["pre_block_layers"][0], keras.Model)
        self.assertEqual(
            structure["sequential_blocks"], self.backbone.transformer_layers
        )

        self.assertIsNone(causal_lm.get_quantization_layer_structure("int8"))


@pytest.mark.skipif(keras.src.backend.backend() != "jax", reason="JAX only")
@pytest.mark.multi_device
class GPT2CausalLMDistributionTest(TestCase):
    def setUp(self):
        super().setUp()

        self.device_count = len(keras.distribution.list_devices())

        # Initialize kwargs for model creation
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|endoftext|>"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.vocabulary_size = len(self.vocab)

        self.tokenizer = GPT2Tokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )
        self.preprocessor = GPT2CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=8,
        )
        self.backbone = GPT2Backbone(
            vocabulary_size=self.vocabulary_size,
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=4,
            max_sequence_length=8,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }

        self.device_mesh = keras.distribution.DeviceMesh(
            shape=(self.device_count,),
            axis_names=["batch"],
            devices=keras.distribution.list_devices(),
        )

        self.layout_map = keras.distribution.LayoutMap(self.device_mesh)

        self.distribution = keras.distribution.DataParallel(
            self.device_mesh,
            self.layout_map,
        )

    def test_e2e_data_parallel_generate(self):
        with self.distribution.scope():
            causal_lm = GPT2CausalLM(**self.init_kwargs)

            # Pass prompts to match the number of devices.
            prompts = [" airplane at airport"] * self.device_count

            # This should run without errors and use the distribution.
            output = causal_lm.generate(prompts)
            self.assertIsInstance(output, (list, tuple))
            self.assertLen(output, self.device_count)

    def test_e2e_data_parallel_generate_indivisible_error(self):
        with self.distribution.scope():
            if self.device_count < 2:
                pytest.skip("This test requires at least 2 devices")
            keras.distribution.set_distribution(self.distribution)
            causal_lm = GPT2CausalLM(**self.init_kwargs)

            # Pass only 1 prompt, which is not divisible by the number of
            # devices.
            prompt = " airplane at airport"

            with self.assertRaises(Exception):
                causal_lm.generate(prompt)
