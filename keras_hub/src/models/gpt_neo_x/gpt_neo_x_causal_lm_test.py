from unittest.mock import patch

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.gpt_neo_x.gpt_neo_x_backbone import GPTNeoXBackbone
from keras_hub.src.models.gpt_neo_x.gpt_neo_x_causal_lm import GPTNeoXCausalLM
from keras_hub.src.models.gpt_neo_x.gpt_neo_x_causal_lm_preprocessor import (
    GPTNeoXCausalLMPreprocessor,
)
from keras_hub.src.models.gpt_neo_x.gpt_neo_x_tokenizer import GPTNeoXTokenizer
from keras_hub.src.tests.test_case import TestCase


class GPTNeoXCausalLMTest(TestCase):
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
        self.preprocessor = GPTNeoXCausalLMPreprocessor(
            GPTNeoXTokenizer(vocabulary=self.vocab, merges=self.merges),
            sequence_length=8,
        )
        self.vocabulary_size = self.preprocessor.tokenizer.vocabulary_size()
        self.backbone = GPTNeoXBackbone(
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
            cls=GPTNeoXCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, self.vocabulary_size),
        )

    def test_generate(self):
        causal_lm = GPTNeoXCausalLM(**self.init_kwargs)
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

    def test_cached_generate(self):
        # The shared backbone has head_dim 2, so with the default
        # rotary_percentage of 0.25 no dimensions are rotated. Use a
        # backbone with a real rotary span so cache positions matter.
        backbone = GPTNeoXBackbone(
            vocabulary_size=self.vocabulary_size,
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            rotary_percentage=1.0,
            max_sequence_length=8,
        )
        causal_lm = GPTNeoXCausalLM(backbone=backbone, preprocessor=None)
        # Default init gives near-uniform attention, so a wrong query
        # position barely moves the logits. Larger, seeded weights make
        # the cached path's position handling observable.
        rng = np.random.default_rng(0)
        for weight in causal_lm.weights:
            values = rng.normal(0.0, 1.0, weight.shape)
            weight.assign(values.astype("float32"))

        prompt_length = 4
        new_tokens = 4
        total = prompt_length + new_tokens

        pad_ids = np.zeros((1, total), dtype="int32")
        pad_ids[0, :prompt_length] = [1, 2, 3, 4]
        pad_mask = np.zeros((1, total), dtype="int32")
        pad_mask[0, :prompt_length] = 1

        uncached_ids = np.copy(pad_ids)
        uncached_mask = np.copy(pad_mask)
        for i in range(prompt_length, total):
            inputs = {
                "token_ids": uncached_ids,
                "padding_mask": uncached_mask,
            }
            logits = ops.convert_to_numpy(causal_lm(inputs, training=False))
            next_token = np.argmax(logits[0, i - 1, :])
            uncached_ids[0, i] = next_token
            uncached_mask[0, i] = 1

        # Seed the cache at the full padded length, as `generate()` does.
        hidden_states, cache = causal_lm._build_cache(
            ops.convert_to_tensor(pad_ids)
        )

        for i in range(prompt_length, total):
            cache_update_index = i - 1
            token = ops.cast(
                uncached_ids[:, cache_update_index : cache_update_index + 1],
                "int32",
            )
            logits, hidden_states, cache = causal_lm.call_with_cache(
                token, cache, cache_update_index
            )
            inputs = {
                "token_ids": ops.cast(uncached_ids[:, :i], "int32"),
                "padding_mask": ops.cast(uncached_mask[:, :i], "int32"),
            }
            full_logits = causal_lm(inputs, training=False)
            # Fixed code on GPU (TF32) differs by <= 2.0e-2; the bug this
            # guards against (query rotated at position 0) differs by >= 2.7.
            self.assertAllClose(
                logits[:, 0, :],
                full_logits[:, cache_update_index, :],
                atol=0.2,
                rtol=0,
            )

        causal_lm.compile(sampler="greedy")
        out = causal_lm.generate(
            {"token_ids": pad_ids, "padding_mask": pad_mask},
            stop_token_ids=None,
        )
        cached_ids = ops.convert_to_numpy(out["token_ids"])
        self.assertEqual(uncached_ids.tolist(), cached_ids.tolist())

    def test_early_stopping(self):
        causal_lm = GPTNeoXCausalLM(**self.init_kwargs)
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

    def test_generate_with_end_token_at_id_zero(self):
        # Pythia vocabularies put "<|endoftext|>" at id 0. Padding must not
        # reuse that id, or `generate()` stops before the first new token.
        tokens = [t for t in self.vocab if t != "<|endoftext|>"]
        vocab = {"<|endoftext|>": 0, "<|padding|>": 1}
        vocab.update({token: i + 2 for i, token in enumerate(tokens)})
        preprocessor = GPTNeoXCausalLMPreprocessor(
            GPTNeoXTokenizer(vocabulary=vocab, merges=self.merges),
            sequence_length=8,
        )
        backbone = GPTNeoXBackbone(
            vocabulary_size=len(vocab),
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            max_sequence_length=8,
        )
        causal_lm = GPTNeoXCausalLM(
            backbone=backbone, preprocessor=preprocessor
        )
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor the "!" token."""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = vocab["!"]
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            output = causal_lm.generate(" airplane at airport")
            # Start token plus five prompt tokens leaves room for two more.
            self.assertEqual(output, " airplane at airport!!")

    def test_generate_compilation(self):
        causal_lm = GPTNeoXCausalLM(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate(" airplane at airport")
        first_fn = causal_lm.generate_function
        causal_lm.generate(" airplane at airport")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=GPTNeoXCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=GPTNeoXCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            output_thresholds={"*": {"max": 1e-3, "mean": 1e-4}},
        )

    def test_presets(self):
        expected_presets = [
            "pythia_70m_en",
            "pythia_160m_en",
            "pythia_410m_en",
            "pythia_1.4b_en",
            "pythia_2.8b_en",
            "pythia_6.9b_en",
            "pythia_12b_en",
            "gpt_neox_20b_en",
        ]
        for preset in expected_presets:
            self.assertIn(preset, GPTNeoXCausalLM.presets)
            self.assertTrue(
                GPTNeoXCausalLM.presets[preset]["kaggle_handle"].startswith(
                    f"kaggle://keras/gpt_neox/keras/{preset}/"
                )
            )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in GPTNeoXCausalLM.presets:
            self.run_preset_test(
                cls=GPTNeoXCausalLM,
                preset=preset,
                input_data=self.input_data,
            )
