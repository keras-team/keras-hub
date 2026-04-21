from unittest.mock import patch

import keras
import numpy as np
import pytest

from keras_hub.src.models.whisper.whisper_audio_converter import (
    WhisperAudioConverter,
)
from keras_hub.src.models.whisper.whisper_audio_to_text import (
    WhisperAudioToText,
)
from keras_hub.src.models.whisper.whisper_audio_to_text_preprocessor import (
    WhisperAudioToTextPreprocessor,
)
from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.models.whisper.whisper_tokenizer import WhisperTokenizer
from keras_hub.src.tests.test_case import TestCase


class WhisperAudioToTextTest(TestCase):
    def setUp(self):
        vocab = ["!", "air", "Ġair", "plane", "Ġat", "port", "<|endoftext|>"]
        vocab = dict([(token, i) for i, token in enumerate(vocab)])
        merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        merges += ["Ġai r", "Ġa i", "pla ne"]
        special_tokens = {
            "<|startoftranscript|>": 9,
            "<|endoftext|>": 10,
            "<|notimestamps|>": 11,
            "<|transcribe|>": 12,
            "<|translate|>": 13,
        }
        self.tokenizer = WhisperTokenizer(
            vocabulary=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )
        self.audio_converter = WhisperAudioConverter(
            num_mels=80,
            num_fft_bins=400,
            stride=160,
            sampling_rate=16000,
            max_audio_length=1,
        )
        self.preprocessor = WhisperAudioToTextPreprocessor(
            audio_converter=self.audio_converter,
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
                "audio": np.random.normal(size=(2, 16000)).astype("float32"),
                "text": [" airplane", " airport"],
            },
        )
        self.input_data = self.preprocessor(self.train_data[0])[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=WhisperAudioToText,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, self.tokenizer.vocabulary_size()),
        )

    def test_generate(self):
        inputs = {
            "audio": np.random.normal(size=(1, 16000)).astype("float32"),
            "text": " air",
        }
        seq_2_seq_lm = WhisperAudioToText(**self.init_kwargs)
        output = seq_2_seq_lm.generate(inputs)
        # The prompt should be preserved as a prefix of the output text.
        self.assertTrue(output.startswith(" air") or " air" in output)

        seq_2_seq_lm.preprocessor = None
        preprocessed = self.preprocessor.generate_preprocess(inputs)
        outputs = seq_2_seq_lm.generate(preprocessed, stop_token_ids=None)
        # The first two tokens (bos + " air") should match the prompt.
        self.assertAllEqual(
            outputs["decoder_token_ids"][:, :2],
            preprocessed["decoder_token_ids"][:, :2],
        )

    def test_early_stopping(self):
        seq_2_seq_lm = WhisperAudioToText(**self.init_kwargs)
        call_decoder_with_cache = seq_2_seq_lm.call_decoder_with_cache

        def wrapper(*args, **kwargs):
            logits, hidden_states, self_cache, cross_cache = (
                call_decoder_with_cache(*args, **kwargs)
            )
            index = self.preprocessor.tokenizer.eos_token_id
            update = keras.ops.ones_like(logits)[:, :, index] * 1.0e9
            update = keras.ops.expand_dims(update, axis=-1)
            logits = keras.ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, self_cache, cross_cache

        with patch.object(
            seq_2_seq_lm, "call_decoder_with_cache", wraps=wrapper
        ):
            inputs = {
                "audio": np.random.normal(size=(2, 16000)).astype("float32"),
                "text": [" air", " airport"],
            }
            output = seq_2_seq_lm.generate(inputs)
            # With eos forced, no additional tokens should be appended.
            for prompt, generated in zip(inputs["text"], output):
                self.assertTrue(generated.startswith(prompt))

    def test_generate_compilation(self):
        seq_2_seq_lm = WhisperAudioToText(**self.init_kwargs)
        audio = np.random.normal(size=(1, 16000)).astype("float32")
        seq_2_seq_lm.generate({"audio": audio})
        first_fn = seq_2_seq_lm.generate_function
        seq_2_seq_lm.generate({"audio": audio})
        second_fn = seq_2_seq_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        seq_2_seq_lm.compile(sampler="greedy")
        self.assertIsNone(seq_2_seq_lm.generate_function)

    def test_beam_search(self):
        seq_2_seq_lm = WhisperAudioToText(**self.init_kwargs)
        seq_2_seq_lm.compile(sampler="beam")
        audio = np.random.normal(size=(1, 16000)).astype("float32")
        seq_2_seq_lm.generate({"audio": audio})

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=WhisperAudioToText,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in WhisperAudioToText.presets:
            self.run_preset_test(
                cls=WhisperAudioToText,
                preset=preset,
                input_data=self.input_data,
            )

    def test_serialization(self):
        instance = WhisperAudioToText(**self.init_kwargs)
        self.run_serialization_test(instance=instance)
