import os
from unittest.mock import patch

import keras
import numpy as np
import pytest

from keras_hub.src.models.moonshine.moonshine_audio_converter import (
    MoonshineAudioConverter,
)
from keras_hub.src.models.moonshine.moonshine_audio_to_text import (
    MoonshineAudioToText,
)
from keras_hub.src.models.moonshine.moonshine_audio_to_text_preprocessor import (  # noqa: E501
    MoonshineAudioToTextPreprocessor,
)
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_tokenizer import (
    MoonshineTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineAudioToTextTest(TestCase):
    def setUp(self):
        self.tokenizer = MoonshineTokenizer(
            proto=os.path.join(self.get_test_data_dir(), "llama_test_vocab.spm")
        )
        self.vocab_size = 10
        hidden_dim = 32
        self.audio_converter = MoonshineAudioConverter(
            sampling_rate=16000,
            do_normalize=False,
            padding_value=0.0,
        )
        self.preprocessor = MoonshineAudioToTextPreprocessor(
            audio_converter=self.audio_converter,
            tokenizer=self.tokenizer,
            decoder_sequence_length=10,
        )
        self.backbone = MoonshineBackbone(
            vocabulary_size=self.vocab_size,
            hidden_dim=hidden_dim,
            filter_dim=hidden_dim,
            encoder_num_layers=2,
            decoder_num_layers=2,
            encoder_num_heads=4,
            decoder_num_heads=4,
            intermediate_dim=hidden_dim * 4,
            feedforward_expansion_factor=4,
            encoder_use_swiglu_activation=False,
            decoder_use_swiglu_activation=True,
            max_position_embeddings=2048,
            pad_head_dim_to_multiple_of=None,
            partial_rotary_factor=0.62,
            dropout=0.0,
            initializer_range=0.02,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        # NOTE: Since keras.ops.convert_to_tensor() does not support
        # dtype="string" for the JAX and PyTorch backends, the only way to pass
        # inputs that aren't a mix of tensors and non-tensors is to use a
        # library-specific function. Using np.random.normal here as a substitute
        # to a librosa.load() call.
        self.train_data = (
            {
                "audio": np.random.normal(size=(2, 16000, 1)),
                "text": ["quick brown", "earth is round"],
            },
        )
        self.input_data = self.preprocessor(self.train_data[0])[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=MoonshineAudioToText,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 10, self.tokenizer.vocabulary_size()),
        )

    def test_generate(self):
        inputs = {"audio": keras.random.normal((1, 16000, 1)), "text": "quick"}
        seq_2_seq_lm = MoonshineAudioToText(**self.init_kwargs)
        output = seq_2_seq_lm.generate(inputs)
        self.assertTrue("quick" in output)

        seq_2_seq_lm.preprocessor = None
        preprocessed = self.preprocessor.generate_preprocess(inputs)
        outputs = seq_2_seq_lm.generate(preprocessed, stop_token_ids=None)
        self.assertAllEqual(
            outputs["decoder_token_ids"][:, :2],
            preprocessed["decoder_token_ids"][:, :2],
        )

    def test_early_stopping(self):
        seq_2_seq_lm = MoonshineAudioToText(**self.init_kwargs)
        call_decoder_with_cache = seq_2_seq_lm.call_decoder_with_cache

        def wrapper(*args, **kwargs):
            logits, hidden_states, self_cache, cross_cache = (
                call_decoder_with_cache(*args, **kwargs)
            )
            index = self.preprocessor.tokenizer.end_token_id
            update = keras.ops.ones_like(logits)[:, :, index] * 1.0e9
            update = keras.ops.expand_dims(update, axis=-1)
            logits = keras.ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, self_cache, cross_cache

        with patch.object(
            seq_2_seq_lm, "call_decoder_with_cache", wraps=wrapper
        ):
            inputs = {
                "audio": keras.random.normal((2, 16000, 1)),
                "text": ["quick", "earth"],
            }
            output = seq_2_seq_lm.generate(inputs)
            self.assertAllEqual(inputs["text"], output)

    def test_generate_compilation(self):
        seq_2_seq_lm = MoonshineAudioToText(**self.init_kwargs)
        seq_2_seq_lm.generate({"audio": keras.random.normal((1, 16000, 1))})
        first_fn = seq_2_seq_lm.generate_function
        seq_2_seq_lm.generate({"audio": keras.random.normal((1, 16000, 1))})
        second_fn = seq_2_seq_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        seq_2_seq_lm.compile(sampler="greedy")
        self.assertIsNone(seq_2_seq_lm.generate_function)

    def test_beam_search(self):
        seq_2_seq_lm = MoonshineAudioToText(**self.init_kwargs)
        seq_2_seq_lm.compile(sampler="beam")
        seq_2_seq_lm.generate({"audio": keras.random.normal((1, 16000, 1))})

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MoonshineAudioToText,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.skip(
        reason="TODO: Bug with MoonshineAudioToText liteRT export"
    )
    def test_litert_export(self):
        self.run_litert_export_test(
            cls=MoonshineAudioToText,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MoonshineAudioToText.presets:
            self.run_preset_test(
                cls=MoonshineAudioToText,
                preset=preset,
                input_data=self.input_data,
            )

    def test_serialization(self):
        instance = MoonshineAudioToText(**self.init_kwargs)
        self.run_serialization_test(instance=instance)
