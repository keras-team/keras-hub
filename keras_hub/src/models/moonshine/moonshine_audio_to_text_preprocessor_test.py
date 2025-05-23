import os

import keras
import numpy as np
import pytest

from keras_hub.src.models.moonshine.moonshine_audio_converter import (
    MoonshineAudioConverter,
)
from keras_hub.src.models.moonshine.moonshine_audio_to_text_preprocessor import (  # noqa: E501
    MoonshineAudioToTextPreprocessor,
)
from keras_hub.src.models.moonshine.moonshine_tokenizer import (
    MoonshineTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineAudioToTextPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = MoonshineTokenizer(
            proto=os.path.join(self.get_test_data_dir(), "llama_test_vocab.spm")
        )
        self.audio_converter = MoonshineAudioConverter()
        self.init_kwargs = {
            "audio_converter": self.audio_converter,
            "tokenizer": self.tokenizer,
            "decoder_sequence_length": 8,
        }
        # NOTE: Since keras.ops.convert_to_tensor() does not support
        # dtype="string" for the JAX and PyTorch backends, the only way to pass
        # inputs that aren't a mix of tensors and non-tensors is to use a
        # library-specific function. Using np.random.normal here as a substitute
        # to a librosa.load() call.
        self.input_data = (
            {
                "audio": np.random.normal(size=(1, 16000, 1)),
                "text": ["the quick brown fox"],
            },
        )

    def test_preprocessor_basics(self):
        preprocessor = MoonshineAudioToTextPreprocessor(**self.init_kwargs)
        output = preprocessor.call(self.input_data[0])
        x_out, y_out, sample_weight_out = output
        self.assertIn("encoder_input_values", x_out)
        self.assertIn("encoder_padding_mask", x_out)
        self.assertIn("decoder_token_ids", x_out)
        self.assertIn("decoder_padding_mask", x_out)
        self.assertAllEqual(
            keras.ops.shape(x_out["encoder_input_values"]),
            (1, 16000, 1),
        )
        self.assertAllEqual(
            keras.ops.shape(x_out["encoder_padding_mask"]),
            (1, 16000),
        )
        self.assertAllEqual(keras.ops.shape(x_out["decoder_token_ids"]), (1, 8))
        self.assertAllEqual(
            keras.ops.shape(x_out["decoder_padding_mask"]), (1, 8)
        )
        self.assertAllEqual(keras.ops.shape(y_out), (1, 8))
        self.assertAllEqual(keras.ops.shape(sample_weight_out), (1, 8))

    def test_generate_preprocess(self):
        preprocessor = MoonshineAudioToTextPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(self.input_data[0])
        self.assertIn("encoder_input_values", output)
        self.assertAllEqual(
            keras.ops.shape(output["encoder_input_values"]),
            (1, 16000, 1),
        )
        self.assertAllClose(output["decoder_token_ids"].shape, [1, 8])

    def test_generate_postprocess(self):
        preprocessor = MoonshineAudioToTextPreprocessor(**self.init_kwargs)
        input_data = {
            "decoder_token_ids": keras.ops.ones((1, 5), dtype="int32"),
            "decoder_padding_mask": keras.ops.ones((1, 5)),
        }
        output = preprocessor.generate_postprocess(input_data)
        self.assertIsInstance(output, list)
        self.assertIsInstance(output[0], str)

    def test_generate_postprocess_batched(self):
        preprocessor = MoonshineAudioToTextPreprocessor(**self.init_kwargs)
        batch_size = 3
        sequence_length = 5
        input_data = {
            "decoder_token_ids": keras.ops.ones(
                (batch_size, sequence_length), dtype="int32"
            ),
            "decoder_padding_mask": keras.ops.ones(
                (batch_size, sequence_length)
            ),
        }
        output = preprocessor.generate_postprocess(input_data)
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), batch_size)
        for item in output:
            self.assertIsInstance(item, str)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MoonshineAudioToTextPreprocessor.presets:
            self.run_preset_test(
                cls=MoonshineAudioToTextPreprocessor,
                preset=preset,
                input_data=self.input_data[0],
            )

    def test_serialization(self):
        instance = MoonshineAudioToTextPreprocessor(**self.init_kwargs)
        self.run_serialization_test(instance=instance)
