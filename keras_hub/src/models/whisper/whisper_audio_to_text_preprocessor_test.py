import keras
import numpy as np
import pytest

from keras_hub.src.models.whisper.whisper_audio_converter import (
    WhisperAudioConverter,
)
from keras_hub.src.models.whisper.whisper_audio_to_text_preprocessor import (
    WhisperAudioToTextPreprocessor,
)
from keras_hub.src.models.whisper.whisper_tokenizer import WhisperTokenizer
from keras_hub.src.tests.test_case import TestCase


class WhisperAudioToTextPreprocessorTest(TestCase):
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
        self.init_kwargs = {
            "audio_converter": self.audio_converter,
            "tokenizer": self.tokenizer,
            "decoder_sequence_length": 8,
        }
        # `np.random.normal` stands in for a real audio waveform since
        # `keras.ops.convert_to_tensor` does not support string dtypes across
        # backends.
        self.input_data = (
            {
                "audio": np.random.normal(size=(1, 16000)).astype("float32"),
                "text": [" airplane at airport"],
            },
        )

    def test_preprocessor_basics(self):
        preprocessor = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        x_out, y_out, sample_weight_out = preprocessor(self.input_data[0])
        self.assertIn("encoder_features", x_out)
        self.assertIn("decoder_token_ids", x_out)
        self.assertIn("decoder_padding_mask", x_out)
        self.assertAllEqual(
            keras.ops.shape(x_out["encoder_features"]), (1, 100, 80)
        )
        self.assertAllEqual(keras.ops.shape(x_out["decoder_token_ids"]), (1, 8))
        self.assertAllEqual(
            keras.ops.shape(x_out["decoder_padding_mask"]), (1, 8)
        )
        self.assertAllEqual(keras.ops.shape(y_out), (1, 8))
        self.assertAllEqual(keras.ops.shape(sample_weight_out), (1, 8))
        # First decoder token should be the bos token.
        self.assertEqual(
            int(keras.ops.convert_to_numpy(x_out["decoder_token_ids"])[0, 0]),
            self.tokenizer.bos_token_id,
        )

    def test_generate_preprocess(self):
        preprocessor = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(self.input_data[0])
        self.assertIn("encoder_features", output)
        self.assertAllEqual(
            keras.ops.shape(output["encoder_features"]), (1, 100, 80)
        )
        self.assertAllEqual(
            keras.ops.shape(output["decoder_token_ids"]), (1, 8)
        )
        # The first token should be bos.
        token_ids = keras.ops.convert_to_numpy(output["decoder_token_ids"])[0]
        self.assertEqual(int(token_ids[0]), self.tokenizer.bos_token_id)

    def test_generate_preprocess_without_text(self):
        preprocessor = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(
            {"audio": self.input_data[0]["audio"]}
        )
        token_ids = keras.ops.convert_to_numpy(output["decoder_token_ids"])
        # With no prompt, the first slot should hold the bos token.
        self.assertEqual(int(token_ids[0, 0]), self.tokenizer.bos_token_id)

    def test_generate_preprocess_in_tf_data(self):
        # Under `tf.data.Dataset.map`, `generate_preprocess` is traced in graph
        # mode and the batch dimension is symbolic. Using `int(batch_size)`
        # here would fail; verify the Keras-ops path works.
        import tensorflow as tf

        preprocessor = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        audio = np.random.normal(size=(3, 16000)).astype("float32")
        ds = tf.data.Dataset.from_tensor_slices({"audio": audio}).batch(3)
        ds = ds.map(preprocessor.generate_preprocess)
        output = next(iter(ds))
        self.assertAllEqual(
            keras.ops.shape(output["encoder_features"]), (3, 100, 80)
        )
        # Every sequence should start with the bos token.
        token_ids = keras.ops.convert_to_numpy(output["decoder_token_ids"])
        for i in range(3):
            self.assertEqual(int(token_ids[i, 0]), self.tokenizer.bos_token_id)

    def test_generate_postprocess(self):
        preprocessor = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        input_data = {
            "decoder_token_ids": keras.ops.array([[9, 2, 3, 4, 10]]),
            "decoder_padding_mask": keras.ops.ones((1, 5), dtype="bool"),
        }
        output = preprocessor.generate_postprocess(input_data)
        self.assertIsInstance(output, list)
        self.assertIsInstance(output[0], str)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in WhisperAudioToTextPreprocessor.presets:
            self.run_preset_test(
                cls=WhisperAudioToTextPreprocessor,
                preset=preset,
                input_data=self.input_data[0],
            )

    def test_serialization(self):
        instance = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        self.run_serialization_test(instance=instance)
