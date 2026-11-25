from unittest.mock import patch

import pytest
from keras import ops

from keras_hub.src.models.whisper.whisper_audio_to_text import (
    WhisperAudioToText,
)
from keras_hub.src.models.whisper.whisper_audio_to_text_preprocessor import (
    WhisperAudioToTextPreprocessor,
)
from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.samplers.greedy_sampler import GreedySampler
from keras_hub.src.tests.mocks.mock_whisper import mock_audio
from keras_hub.src.tests.mocks.mock_whisper import mock_whisper_audio_converter
from keras_hub.src.tests.mocks.mock_whisper import mock_whisper_tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu

# Control token ids of the mock tokenizer. Every other id (`0..29`) is a
# regular BPE token.
CONTROL_TOKEN_IDS = [30, 31, 32, 33, 34]
EOS = 30


class WhisperAudioToTextTest(TestCase):
    def setUp(self):
        self.tokenizer = mock_whisper_tokenizer()
        self.preprocessor = WhisperAudioToTextPreprocessor(
            audio_converter=mock_whisper_audio_converter(),
            tokenizer=self.tokenizer,
            decoder_sequence_length=8,
        )
        self.backbone = WhisperBackbone(
            vocabulary_size=self.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            num_mels=80,
            max_encoder_sequence_length=100,
            max_decoder_sequence_length=8,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.train_data = (
            {
                "audio": mock_audio((2, 16000)),
                "text": [" airplane", " airport"],
            },
        )
        self.input_data = self.preprocessor(self.train_data[0])[0]

    def _model_that_always_extends(self):
        """A model that cannot stop at the first step or emit control tokens.

        This makes random-weight greedy generation always add text.
        """
        return WhisperAudioToText(
            **self.init_kwargs,
            suppress_tokens=CONTROL_TOKEN_IDS[1:],
            begin_suppress_tokens=[EOS],
        )

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=WhisperAudioToText,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, self.tokenizer.vocabulary_size()),
        )

    def test_default_sampler_is_greedy(self):
        seq_2_seq_lm = WhisperAudioToText(**self.init_kwargs)
        self.assertIsInstance(seq_2_seq_lm.sampler, GreedySampler)

    def test_generate(self):
        seq_2_seq_lm = self._model_that_always_extends()
        audio = mock_audio((1, 16000))

        # A forced transcript prefix stays in the output.
        output = seq_2_seq_lm.generate({"audio": audio, "text": [" air"]})
        self.assertIsInstance(output, list)
        self.assertLen(output, 1)
        self.assertTrue(output[0].startswith(" air"))
        self.assertGreater(len(output[0]), len(" air"))

        # Greedy decoding is deterministic.
        self.assertEqual(
            seq_2_seq_lm.generate({"audio": audio, "text": [" air"]}), output
        )

        # Unprompted generation.
        output = seq_2_seq_lm.generate({"audio": audio})
        self.assertIsInstance(output, list)
        self.assertLen(output, 1)
        self.assertGreater(len(output[0]), 0)

        # The output type does not depend on key order or a scalar prompt.
        in_order = seq_2_seq_lm.generate({"audio": audio, "text": " air"})
        reversed_order = seq_2_seq_lm.generate({"text": " air", "audio": audio})
        self.assertIsInstance(in_order, list)
        self.assertEqual(in_order, reversed_order)

        # Batched audio with a scalar prompt keeps every output.
        output = seq_2_seq_lm.generate(
            {"audio": mock_audio((2, 16000)), "text": " air"}
        )
        self.assertLen(output, 2)
        for generated in output:
            self.assertTrue(generated.startswith(" air"))

        # Unbatched audio gives a single string.
        output = seq_2_seq_lm.generate(
            {"audio": mock_audio((16000,)), "text": " air"}
        )
        self.assertIsInstance(output, str)
        self.assertTrue(output.startswith(" air"))

    def test_generate_does_not_mutate_inputs(self):
        seq_2_seq_lm = self._model_that_always_extends()
        inputs = {"audio": mock_audio((16000,)), "text": " air"}
        first = seq_2_seq_lm.generate(inputs)
        self.assertEqual(inputs["text"], " air")
        self.assertEqual(inputs["audio"].shape, (16000,))
        self.assertEqual(seq_2_seq_lm.generate(inputs), first)

    def test_generate_without_preprocessor(self):
        seq_2_seq_lm = self._model_that_always_extends()
        inputs = {"audio": mock_audio((1, 16000)), "text": [" air"]}
        preprocessed = self.preprocessor.generate_preprocess(inputs)
        seq_2_seq_lm.preprocessor = None
        outputs = seq_2_seq_lm.generate(preprocessed, stop_token_ids=None)
        # The prefix and the prompt (bos, no-timestamps, " air") are kept.
        self.assertAllEqual(
            outputs["decoder_token_ids"][:, :3],
            preprocessed["decoder_token_ids"][:, :3],
        )
        # Without stop tokens every position is generated.
        self.assertAllEqual(outputs["decoder_padding_mask"], [[True] * 8])

    def test_suppress_tokens(self):
        boosted = 5

        def run(seq_2_seq_lm):
            call_decoder_with_cache = seq_2_seq_lm.call_decoder_with_cache

            def wrapper(*args, **kwargs):
                logits, hidden_states, self_cache, cross_cache = (
                    call_decoder_with_cache(*args, **kwargs)
                )
                update = ops.ones_like(logits)[:, :, boosted] * 1.0e9
                update = ops.expand_dims(update, axis=-1)
                logits = ops.slice_update(logits, (0, 0, boosted), update)
                return logits, hidden_states, self_cache, cross_cache

            with patch.object(
                seq_2_seq_lm, "call_decoder_with_cache", wraps=wrapper
            ):
                preprocessed = self.preprocessor.generate_preprocess(
                    {"audio": mock_audio((2, 16000)), "text": [" air", ""]}
                )
                seq_2_seq_lm.preprocessor = None
                outputs = seq_2_seq_lm.generate(
                    preprocessed, stop_token_ids=None
                )
                seq_2_seq_lm.preprocessor = self.preprocessor
                return ops.convert_to_numpy(outputs["decoder_token_ids"])

        # `suppress_tokens` are never generated.
        token_ids = run(
            WhisperAudioToText(**self.init_kwargs, suppress_tokens=[boosted])
        )
        self.assertNotIn(boosted, token_ids[0, 3:])
        self.assertNotIn(boosted, token_ids[1, 2:])

        # `begin_suppress_tokens` are only masked at each sequence's first
        # generated step (index 3 after a prompt, index 2 without one).
        token_ids = run(
            WhisperAudioToText(
                **self.init_kwargs, begin_suppress_tokens=[boosted]
            )
        )
        self.assertNotEqual(token_ids[0, 3], boosted)
        self.assertAllEqual(token_ids[0, 4:], [boosted] * 4)
        self.assertNotEqual(token_ids[1, 2], boosted)
        self.assertAllEqual(token_ids[1, 3:], [boosted] * 5)

    def test_early_stopping(self):
        seq_2_seq_lm = WhisperAudioToText(**self.init_kwargs)
        call_decoder_with_cache = seq_2_seq_lm.call_decoder_with_cache

        def wrapper(*args, **kwargs):
            logits, hidden_states, self_cache, cross_cache = (
                call_decoder_with_cache(*args, **kwargs)
            )
            index = self.preprocessor.tokenizer.eos_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, self_cache, cross_cache

        with patch.object(
            seq_2_seq_lm, "call_decoder_with_cache", wraps=wrapper
        ):
            inputs = {
                "audio": mock_audio((2, 16000)),
                "text": [" air", " airport"],
            }
            output = seq_2_seq_lm.generate(inputs)
            # With eos forced, no additional tokens are appended.
            self.assertAllEqual(output, inputs["text"])

    def test_generate_compilation(self):
        seq_2_seq_lm = WhisperAudioToText(**self.init_kwargs)
        audio = mock_audio((1, 16000))
        seq_2_seq_lm.generate({"audio": audio})
        first_fn = seq_2_seq_lm.generate_function
        seq_2_seq_lm.generate({"audio": audio})
        second_fn = seq_2_seq_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        seq_2_seq_lm.compile(sampler="greedy")
        self.assertIsNone(seq_2_seq_lm.generate_function)

    def test_beam_search(self):
        seq_2_seq_lm = self._model_that_always_extends()
        seq_2_seq_lm.compile(sampler="beam")
        prompts = [" air", " airport"]
        output = seq_2_seq_lm.generate(
            {"audio": mock_audio((2, 16000)), "text": prompts}
        )
        self.assertIsInstance(output, list)
        self.assertLen(output, 2)
        for prompt, generated in zip(prompts, output):
            self.assertTrue(generated.startswith(prompt))

    def test_cached_decoding_matches_uncached_forward_pass(self):
        """Decoding a token at a time must match a single forward pass."""
        seq_2_seq_lm = WhisperAudioToText(**self.init_kwargs)
        decoder_length = 6
        inputs = {
            "encoder_features": ops.convert_to_tensor(mock_audio((2, 100, 80))),
            "decoder_token_ids": ops.array([[31, 32, 23, 14, 24, 16]] * 2),
            "decoder_padding_mask": ops.ones((2, decoder_length), "int32"),
        }
        expected_logits = seq_2_seq_lm(inputs, training=False)

        _, encoder_hidden_states, self_cache, cross_cache = (
            seq_2_seq_lm._build_cache(
                inputs["encoder_features"], inputs["decoder_token_ids"]
            )
        )
        atol = 5e-3 if running_on_gpu() else 1e-4
        for index in range(decoder_length):
            logits, _, self_cache, _ = seq_2_seq_lm.call_decoder_with_cache(
                encoder_hidden_states=encoder_hidden_states,
                decoder_token_ids=inputs["decoder_token_ids"][
                    :, index : index + 1
                ],
                self_attention_cache=self_cache,
                self_attention_cache_update_index=index,
                cross_attention_cache=cross_cache,
                cross_attention_cache_update_index=None,
            )
            self.assertAllClose(
                logits[:, 0, :], expected_logits[:, index, :], atol=atol
            )

    def test_decoder_padding_mask_matches_forward_pass(self):
        seq_2_seq_lm = WhisperAudioToText(**self.init_kwargs)
        # Left-padded decoder inputs, as a user may pass without a
        # preprocessor.
        padding_mask = ops.array([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        inputs = {
            "encoder_features": ops.convert_to_tensor(mock_audio((2, 100, 80))),
            "decoder_token_ids": ops.array([[30, 30, 31, 32, 23, 16]] * 2),
            "decoder_padding_mask": padding_mask,
        }
        expected_logits = seq_2_seq_lm(inputs, training=False)
        encoder_hidden_states = seq_2_seq_lm.call_encoder(
            inputs["encoder_features"]
        )
        atol = 5e-3 if running_on_gpu() else 1e-4
        # Without caches.
        logits, _, _, _ = seq_2_seq_lm.call_decoder_with_cache(
            encoder_hidden_states=encoder_hidden_states,
            decoder_token_ids=inputs["decoder_token_ids"],
            decoder_padding_mask=padding_mask,
        )
        self.assertAllClose(logits, expected_logits, atol=atol)
        # Prefilling the caches.
        self_cache, cross_cache = seq_2_seq_lm._initialize_cache(
            encoder_hidden_states, inputs["decoder_token_ids"]
        )
        logits, _, _, _ = seq_2_seq_lm.call_decoder_with_cache(
            encoder_hidden_states=encoder_hidden_states,
            decoder_token_ids=inputs["decoder_token_ids"],
            decoder_padding_mask=padding_mask,
            self_attention_cache=self_cache,
            self_attention_cache_update_index=0,
            cross_attention_cache=cross_cache,
            cross_attention_cache_update_index=0,
        )
        self.assertAllClose(logits, expected_logits, atol=atol)

    def test_call_decoder_with_cache_requires_both_caches(self):
        seq_2_seq_lm = WhisperAudioToText(**self.init_kwargs)
        encoder_hidden_states = seq_2_seq_lm.call_encoder(
            ops.convert_to_tensor(mock_audio((1, 100, 80)))
        )
        token_ids = ops.array([[31, 32]])
        self_cache, _ = seq_2_seq_lm._initialize_cache(
            encoder_hidden_states, token_ids
        )
        with self.assertRaisesRegex(ValueError, "both be set or both be"):
            seq_2_seq_lm.call_decoder_with_cache(
                encoder_hidden_states=encoder_hidden_states,
                decoder_token_ids=token_ids,
                self_attention_cache=self_cache,
                self_attention_cache_update_index=0,
            )

    def test_asymmetric_decoder(self):
        backbone = WhisperBackbone(
            vocabulary_size=self.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            decoder_num_layers=1,
            decoder_num_heads=4,
            decoder_intermediate_dim=8,
            max_encoder_sequence_length=100,
            max_decoder_sequence_length=8,
        )
        self.assertLen(backbone.decoder_transformer_layers, 1)
        seq_2_seq_lm = WhisperAudioToText(
            backbone=backbone, preprocessor=self.preprocessor
        )
        output = seq_2_seq_lm.generate({"audio": mock_audio((1, 16000))})
        self.assertLen(output, 1)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=WhisperAudioToText,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=WhisperAudioToText,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_serialization(self):
        instance = WhisperAudioToText(
            **self.init_kwargs,
            suppress_tokens=[1, 2],
            begin_suppress_tokens=[EOS],
        )
        self.run_serialization_test(instance=instance)

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=WhisperAudioToText,
            preset="whisper_tiny_en",
            input_data=self.input_data,
        )
        seq_2_seq_lm = WhisperAudioToText.from_preset("whisper_tiny_en")
        output = seq_2_seq_lm.generate({"audio": mock_audio((16000,))})
        self.assertIsInstance(output, str)
