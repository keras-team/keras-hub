import os
from unittest.mock import patch

import pytest
from keras import ops

from keras_hub.src.models.t5.t5_backbone import T5Backbone
from keras_hub.src.models.t5.t5_seq_2_seq_lm import T5Seq2SeqLM
from keras_hub.src.models.t5.t5_seq_2_seq_lm_preprocessor import (
    T5Seq2SeqLMPreprocessor,
)
from keras_hub.src.models.t5.t5_tokenizer import T5Tokenizer
from keras_hub.src.tests.test_case import TestCase


class T5Seq2SeqLMTest(TestCase):
    def setUp(self):
        self.tokenizer = T5Tokenizer(
            # Generated using create_t5_test_proto.py
            proto=os.path.join(self.get_test_data_dir(), "t5_test_vocab.spm")
        )
        self.preprocessor = T5Seq2SeqLMPreprocessor(
            self.tokenizer,
            encoder_sequence_length=12,
            decoder_sequence_length=10,
        )
        self.vocabulary_size = self.tokenizer.vocabulary_size()
        self.backbone = T5Backbone(
            vocabulary_size=self.vocabulary_size,
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (
            {
                "encoder_text": ["the quick brown fox", "the quick brown fox"],
                "decoder_text": ["the earth is round", "the earth is round"],
            },
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_seq_2_seq_lm_basics(self):
        self.run_task_test(
            cls=T5Seq2SeqLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 10, self.vocabulary_size),
        )

    def test_generate(self):
        # String input.
        inputs = {
            "encoder_text": "the quick brown fox",
            "decoder_text": "the earth",
        }
        seq_2_seq_lm = T5Seq2SeqLM(**self.init_kwargs)
        output = seq_2_seq_lm.generate(inputs)
        self.assertTrue("the earth" in output)
        # String tensor input.
        self.assertIsInstance(seq_2_seq_lm.generate("the quick brown fox"), str)

        # Int tensor input.
        seq_2_seq_lm.preprocessor = None
        preprocessed_batch = self.preprocessor.generate_preprocess(inputs)
        outputs = seq_2_seq_lm.generate(preprocessed_batch, stop_token_ids=None)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["decoder_token_ids"][:, :3],
            preprocessed_batch["decoder_token_ids"][:, :3],
        )
        self.assertAllEqual(
            outputs["decoder_padding_mask"][:, :3],
            preprocessed_batch["decoder_padding_mask"][:, :3],
        )

    def test_cached_decoding_matches_uncached_forward_pass(self):
        """Decoding a token at a time must match a single full forward pass."""
        seq_2_seq_lm = T5Seq2SeqLM(**self.init_kwargs)
        decoder_length = 5
        # Use fully unpadded inputs; the uncached forward pass additionally
        # masks padded decoder positions, which cached decoding leaves to the
        # sampler.
        inputs = {
            "encoder_token_ids": ops.array([[4, 9, 5, 7, 1]] * 2, "int32"),
            "encoder_padding_mask": ops.ones((2, 5), "int32"),
            "decoder_token_ids": ops.array([[0, 4, 6, 8, 10]] * 2, "int32"),
            "decoder_padding_mask": ops.ones((2, decoder_length), "int32"),
        }
        expected_logits = seq_2_seq_lm(inputs, training=False)

        _, encoder_hidden_states, self_cache, cross_cache = (
            seq_2_seq_lm._build_cache(
                inputs["encoder_token_ids"],
                inputs["encoder_padding_mask"],
                inputs["decoder_token_ids"],
            )
        )

        for index in range(decoder_length):
            logits, _, self_cache, _ = seq_2_seq_lm.call_decoder_with_cache(
                encoder_hidden_states=encoder_hidden_states,
                encoder_padding_mask=inputs["encoder_padding_mask"],
                decoder_token_ids=inputs["decoder_token_ids"][
                    :, index : index + 1
                ],
                self_attention_cache=self_cache,
                self_attention_cache_update_index=index,
                cross_attention_cache=cross_cache,
                cross_attention_cache_update_index=None,
            )
            self.assertAllClose(
                logits[:, 0, :],
                expected_logits[:, index, :],
                atol=1e-4,
                rtol=1e-4,
            )

    def test_early_stopping(self):
        seq_2_seq_lm = T5Seq2SeqLM(**self.init_kwargs)
        call_decoder_with_cache = seq_2_seq_lm.call_decoder_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            (
                logits,
                hidden_states,
                self_attention_cache,
                cross_attention_cache,
            ) = call_decoder_with_cache(*args, **kwargs)
            index = self.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return (
                logits,
                hidden_states,
                self_attention_cache,
                cross_attention_cache,
            )

        with patch.object(
            seq_2_seq_lm, "call_decoder_with_cache", wraps=wrapper
        ):
            inputs = {
                "encoder_text": ["the quick brown fox", "the quick brown fox"],
                "decoder_text": ["the earth", "the"],
            }
            output = seq_2_seq_lm.generate(inputs)
            # We should immediately abort and output the prompt.
            self.assertAllEqual(inputs["decoder_text"], output)

    def test_beam_search(self):
        seq_2_seq_lm = T5Seq2SeqLM(**self.init_kwargs)
        seq_2_seq_lm.compile(sampler="beam")
        seq_2_seq_lm.generate("the quick brown fox")

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=T5Seq2SeqLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in T5Seq2SeqLM.presets:
            self.run_preset_test(
                cls=T5Seq2SeqLM,
                preset=preset,
                input_data=self.input_data,
            )
