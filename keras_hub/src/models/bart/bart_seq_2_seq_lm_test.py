from unittest.mock import patch

import pytest
from keras import ops

from keras_hub.src.models.bart.bart_backbone import BartBackbone
from keras_hub.src.models.bart.bart_seq_2_seq_lm import BartSeq2SeqLM
from keras_hub.src.models.bart.bart_seq_2_seq_lm_preprocessor import (
    BartSeq2SeqLMPreprocessor,
)
from keras_hub.src.models.bart.bart_tokenizer import BartTokenizer
from keras_hub.src.tests.test_case import TestCase


class BartSeq2SeqLMTest(TestCase):
    def setUp(self):
        self.vocab = ["<s>", "<pad>", "</s>", "air", "Ġair", "plane", "Ġat"]
        self.vocab += ["port", "<mask>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.preprocessor = BartSeq2SeqLMPreprocessor(
            BartTokenizer(vocabulary=self.vocab, merges=self.merges),
            encoder_sequence_length=12,
            decoder_sequence_length=10,
        )
        self.backbone = BartBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            max_sequence_length=12,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (
            {
                "encoder_text": [
                    " airplane at airport",
                    " airplane at airport",
                ],
                "decoder_text": [" airplane airport", " airplane airport"],
            },
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=BartSeq2SeqLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 10, 9),
        )

    def test_generate(self):
        # String input.
        inputs = {
            "encoder_text": " airplane at airport",
            "decoder_text": " airplane at",
        }
        seq_2_seq_lm = BartSeq2SeqLM(**self.init_kwargs)
        output = seq_2_seq_lm.generate(inputs)
        self.assertTrue(" airplane at" in output)
        # String tensor input.
        self.assertIsInstance(
            seq_2_seq_lm.generate(" airplane at airport"), str
        )

        # Int tensor input.
        seq_2_seq_lm.preprocessor = None
        preprocessed_batch = self.preprocessor.generate_preprocess(inputs)
        outputs = seq_2_seq_lm.generate(preprocessed_batch, stop_token_ids=None)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["decoder_token_ids"][:, :5],
            preprocessed_batch["decoder_token_ids"][:, :5],
        )
        self.assertAllEqual(
            outputs["decoder_padding_mask"][:, :5],
            preprocessed_batch["decoder_padding_mask"][:, :5],
        )

    def test_early_stopping(self):
        seq_2_seq_lm = BartSeq2SeqLM(**self.init_kwargs)
        call_decoder_with_cache = seq_2_seq_lm.call_decoder_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            (
                logits,
                hidden_states,
                self_attention_cache,
                cross_attention_cache,
            ) = call_decoder_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
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
                "encoder_text": [
                    " airplane at airport",
                    " airplane at airport",
                ],
                "decoder_text": [" airplane at", " airplane"],
            }
            output = seq_2_seq_lm.generate(inputs)
            # We should immediately abort and output the prompt.
            self.assertAllEqual(inputs["decoder_text"], output)

    def test_generate_compilation(self):
        seq_2_seq_lm = BartSeq2SeqLM(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        seq_2_seq_lm.generate(" airplane at airport")
        first_fn = seq_2_seq_lm.generate_function
        seq_2_seq_lm.generate(" airplane at airport")
        second_fn = seq_2_seq_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        seq_2_seq_lm.compile(sampler="greedy")
        self.assertIsNone(seq_2_seq_lm.generate_function)

    def test_beam_search(self):
        seq_2_seq_lm = BartSeq2SeqLM(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
        )
        seq_2_seq_lm.compile(sampler="beam")
        seq_2_seq_lm.generate(" airplane at airport")

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BartSeq2SeqLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=BartSeq2SeqLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BartSeq2SeqLM.presets:
            self.run_preset_test(
                cls=BartSeq2SeqLM,
                preset=preset,
                input_data=self.input_data,
            )
