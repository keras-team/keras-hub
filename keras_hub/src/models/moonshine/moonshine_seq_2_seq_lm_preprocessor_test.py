import os

import keras
import pytest

from keras_hub.src.models.moonshine.moonshine_audio_converter import (
    MoonshineAudioConverter,
)
from keras_hub.src.models.moonshine.moonshine_seq_2_seq_lm_preprocessor import (
    MoonshineSeq2SeqLMPreprocessor,
)
from keras_hub.src.models.moonshine.moonshine_tokenizer import (
    MoonshineTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineSeq2SeqLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = MoonshineTokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "moonshine_test_vocab.spm"
            )
        )
        self.audio_converter = MoonshineAudioConverter(filter_dim=32)
        self.init_kwargs = {
            "audio_converter": self.audio_converter,
            "tokenizer": self.tokenizer,
            "encoder_sequence_length": None,
            "decoder_sequence_length": 8,
        }
        self.input_data = (
            {
                "audio": keras.random.normal((1, 16000, 1)),
                "text": ["the quick brown fox"],
            },
        )

    def test_preprocessor_basics(self):
        preprocessor = MoonshineSeq2SeqLMPreprocessor(**self.init_kwargs)
        output = preprocessor.call(self.input_data)
        x_out, y_out, sample_weight_out = output
        self.assertIn("encoder_input_values", x_out)
        self.assertIn("encoder_padding_mask", x_out)
        self.assertIn("decoder_token_ids", x_out)
        self.assertIn("decoder_padding_mask", x_out)
        self.assertAllEqual(
            keras.ops.shape(x_out["encoder_input_values"]), (1, 40, 32)
        )
        self.assertAllEqual(
            keras.ops.shape(x_out["encoder_padding_mask"]), (1, 40)
        )
        self.assertAllEqual(keras.ops.shape(x_out["decoder_token_ids"]), (1, 8))
        self.assertAllEqual(
            keras.ops.shape(x_out["decoder_padding_mask"]), (1, 8)
        )
        self.assertAllEqual(keras.ops.shape(y_out), (1, 8))
        self.assertAllEqual(keras.ops.shape(sample_weight_out), (1, 8))

    def test_generate_preprocess(self):
        preprocessor = MoonshineSeq2SeqLMPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(self.input_data)
        self.assertIn("encoder_input_values", output)
        self.assertAllClose(output["decoder_token_ids"].shape, [1, 8])

    def test_generate_postprocess(self):
        preprocessor = MoonshineSeq2SeqLMPreprocessor(**self.init_kwargs)
        input_data = {
            "decoder_token_ids": keras.ops.ones((1, 5), dtype="int32"),
            "decoder_padding_mask": keras.ops.ones((1, 5)),
        }
        output = preprocessor.generate_postprocess(input_data)
        self.assertIsInstance(output, list)
        self.assertIsInstance(output[0], str)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MoonshineSeq2SeqLMPreprocessor.presets:
            self.run_preset_test(
                cls=MoonshineSeq2SeqLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )

    def test_serialization(self):
        preprocessor = MoonshineSeq2SeqLMPreprocessor(**self.init_kwargs)
        self.run_serialization_test(instance=preprocessor)
