import os
from unittest.mock import patch

import keras
import pytest

from keras_hub.src.models.t5gemma.t5gemma_backbone import T5GemmaBackbone
from keras_hub.src.models.t5gemma.t5gemma_causal_lm import T5GemmaCausalLM
from keras_hub.src.models.t5gemma.t5gemma_causal_lm_preprocessor import (
    T5GemmaCausalLMPreprocessor,
)
from keras_hub.src.models.t5gemma.t5gemma_tokenizer import T5GemmaTokenizer
from keras_hub.src.tests.test_case import TestCase


class T5GemmaCausalLMTest(TestCase):
    def setUp(self):
        self.tokenizer = T5GemmaTokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "gemma_test_vocab.spm"
            ),
        )
        self.preprocessor = T5GemmaCausalLMPreprocessor(
            self.tokenizer,
            sequence_length=8,
        )
        self.backbone = T5GemmaBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            hidden_dim=16,
            intermediate_dim=32,
            num_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            dropout_rate=0.0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            query_pre_attn_scalar=1.0,
            attention_bias=False,
            hidden_activation="gelu_approximate",
            layer_types=["sliding_attention", "full_attention"],
            initializer_range=0.02,
            attention_dropout=0.0,
            sliding_window=4,
            final_logit_softcapping=30.0,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (["the quick brown fox", "the earth is round"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=T5GemmaCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(
                2,
                8,
                self.preprocessor.tokenizer.vocabulary_size(),
            ),
        )

    def test_generate(self):
        causal_lm = T5GemmaCausalLM(**self.init_kwargs)
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

    def test_generate_strip_prompt(self):
        causal_lm = T5GemmaCausalLM(**self.init_kwargs)
        prompt = "the quick brown fox"
        output = causal_lm.generate(prompt, strip_prompt=True)
        self.assertFalse(output.startswith(prompt))

    def test_early_stopping(self):
        causal_lm = T5GemmaCausalLM(**self.init_kwargs)
        call_decoder_with_cache = causal_lm.call_decoder_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            (
                logits,
                hidden_states,
                cache,
            ) = call_decoder_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = (
                keras.ops.ones(
                    (keras.ops.shape(logits)[0], 1, 1), dtype=logits.dtype
                )
                * 1.0e9
            )
            logits = keras.ops.slice_update(logits, (0, 0, index), update)
            return (
                logits,
                hidden_states,
                cache,
            )

        with patch.object(causal_lm, "call_decoder_with_cache", wraps=wrapper):
            prompt = ["the quick brown fox", "the earth is round"]
            output = causal_lm.generate(prompt)
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = T5GemmaCausalLM(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate("the quick brown fox")
        first_fn = causal_lm.generate_function
        causal_lm.generate("the quick brown fox")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=T5GemmaCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in T5GemmaCausalLM.presets:
            self.run_preset_test(
                cls=T5GemmaCausalLM,
                preset=preset,
                input_data=self.input_data,
            )
