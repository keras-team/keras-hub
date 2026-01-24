import os
from unittest.mock import patch

import keras
import pytest

from keras_hub.src.models.t5gemma.t5gemma_backbone import T5GemmaBackbone
from keras_hub.src.models.t5gemma.t5gemma_seq_2_seq_lm import T5GemmaSeq2SeqLM
from keras_hub.src.models.t5gemma.t5gemma_seq_2_seq_lm_preprocessor import (
    T5GemmaSeq2SeqLMPreprocessor,
)
from keras_hub.src.models.t5gemma.t5gemma_tokenizer import T5GemmaTokenizer
from keras_hub.src.tests.test_case import TestCase


class T5GemmaSeq2SeqLMTest(TestCase):
    def setUp(self):
        self.tokenizer = T5GemmaTokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "gemma_test_vocab.spm"
            ),
        )
        self.preprocessor = T5GemmaSeq2SeqLMPreprocessor(
            tokenizer=self.tokenizer,
            encoder_sequence_length=8,
            decoder_sequence_length=10,
        )
        self.backbone = T5GemmaBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            encoder_hidden_dim=16,
            encoder_intermediate_dim=32,
            encoder_num_layers=2,
            encoder_num_attention_heads=2,
            encoder_num_key_value_heads=1,
            encoder_head_dim=8,
            encoder_layer_types=["sliding_attention", "full_attention"],
            decoder_hidden_dim=16,
            decoder_intermediate_dim=32,
            decoder_num_layers=2,
            decoder_num_attention_heads=2,
            decoder_num_key_value_heads=1,
            decoder_head_dim=8,
            decoder_layer_types=["sliding_attention", "full_attention"],
            dropout_rate=0.0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            query_pre_attn_scalar=1.0,
            attention_bias=False,
            hidden_activation="gelu_approximate",
            initializer_range=0.02,
            attention_dropout=0.0,
            sliding_window=4,
            final_logit_softcapping=30.0,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (
            {
                "encoder_text": ["the quick brown fox", "the earth is round"],
                "decoder_text": ["the quick brown fox", "the earth is round"],
            },
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=T5GemmaSeq2SeqLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(
                2,
                10,
                self.preprocessor.tokenizer.vocabulary_size(),
            ),
        )

    def test_generate(self):
        causal_lm = T5GemmaSeq2SeqLM(**self.init_kwargs)
        # String inputs.
        inputs = {
            "encoder_text": "the quick brown fox",
            "decoder_text": "the quick",
        }
        output = causal_lm.generate(inputs)
        self.assertTrue("the quick" in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess(inputs)
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["decoder_token_ids"][:, :3],
            prompt_ids["decoder_token_ids"][:, :3],
        )
        self.assertAllEqual(
            outputs["decoder_padding_mask"][:, :3],
            prompt_ids["decoder_padding_mask"][:, :3],
        )

    def test_early_stopping(self):
        causal_lm = T5GemmaSeq2SeqLM(**self.init_kwargs)
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
            inputs = {
                "encoder_text": [
                    "the quick brown fox",
                    "the earth is round",
                ],
                "decoder_text": ["the quick", "the earth"],
            }
            output = causal_lm.generate(inputs)
            # We should immediately abort and output the prompt.
            self.assertEqual(inputs["decoder_text"], output)

    def test_generate_compilation(self):
        causal_lm = T5GemmaSeq2SeqLM(**self.init_kwargs)
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
            cls=T5GemmaSeq2SeqLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=T5GemmaSeq2SeqLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in T5GemmaSeq2SeqLM.presets:
            self.run_preset_test(
                cls=T5GemmaSeq2SeqLM,
                preset=preset,
                input_data=self.input_data,
            )
