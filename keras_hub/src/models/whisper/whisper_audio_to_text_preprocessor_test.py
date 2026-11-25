import keras
import numpy as np
import pytest

from keras_hub.src.models.whisper.whisper_audio_to_text_preprocessor import (
    WhisperAudioToTextPreprocessor,
)
from keras_hub.src.tests.mocks.mock_whisper import mock_audio
from keras_hub.src.tests.mocks.mock_whisper import mock_whisper_audio_converter
from keras_hub.src.tests.mocks.mock_whisper import mock_whisper_tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import tf

# Token ids of the mock tokenizer.
EOS = 30  # Also the pad token.
BOS = 31
NO_TIMESTAMPS = 32
TRANSCRIBE = 33
TRANSLATE = 34
EN = 35
FR = 36
AIR = 23  # " air"
PORT = 16  # "port"


class WhisperAudioToTextPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = mock_whisper_tokenizer()
        self.audio_converter = mock_whisper_audio_converter()
        self.init_kwargs = {
            "audio_converter": self.audio_converter,
            "tokenizer": self.tokenizer,
            "decoder_sequence_length": 8,
        }
        self.input_data = (
            {
                "audio": mock_audio((1, 16000)),
                "text": [" airplane at airport"],
            },
        )

    def test_preprocessor_basics(self):
        x, y, sample_weight = self.run_preprocessor_test(
            cls=WhisperAudioToTextPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            token_id_key="decoder_token_ids",
        )
        self.assertAllEqual(
            keras.ops.shape(x["encoder_features"]), (1, 100, 80)
        )
        # `<|startoftranscript|><|notimestamps|> airplane at airport` + eos.
        self.assertAllEqual(
            x["decoder_token_ids"],
            [[BOS, NO_TIMESTAMPS, 23, 14, 24, 23, 16, EOS]],
        )
        self.assertAllEqual(x["decoder_padding_mask"], [[True] * 8])
        self.assertAllEqual(y, [[NO_TIMESTAMPS, 23, 14, 24, 23, 16, EOS, EOS]])
        self.assertAllEqual(sample_weight, [[True] * 7 + [False]])

    def test_multilingual_prefix(self):
        tokenizer = mock_whisper_tokenizer(multilingual=True)
        preprocessor = WhisperAudioToTextPreprocessor(
            audio_converter=self.audio_converter,
            tokenizer=tokenizer,
        )
        # Defaults to English transcription without timestamps.
        self.assertEqual(
            preprocessor.decoder_prefix_ids,
            [BOS, EN, TRANSCRIBE, NO_TIMESTAMPS],
        )
        preprocessor = WhisperAudioToTextPreprocessor(
            audio_converter=self.audio_converter,
            tokenizer=tokenizer,
            language="fr",
            task="translate",
            return_timestamps=True,
        )
        self.assertEqual(preprocessor.decoder_prefix_ids, [BOS, FR, TRANSLATE])
        x, y, _ = preprocessor(
            {"audio": mock_audio((1, 16000)), "text": [" air"]},
            sequence_length=5,
        )
        self.assertAllEqual(
            x["decoder_token_ids"], [[BOS, FR, TRANSLATE, AIR, EOS]]
        )
        self.assertAllEqual(y, [[FR, TRANSLATE, AIR, EOS, EOS]])
        # Changing the attributes after the packer is built updates it.
        preprocessor.language = "<|en|>"
        preprocessor.task = "transcribe"
        preprocessor.return_timestamps = False
        output = preprocessor.generate_preprocess(
            {"audio": mock_audio((1, 16000))}, sequence_length=5
        )
        self.assertAllEqual(
            output["decoder_token_ids"],
            [[BOS, EN, TRANSCRIBE, NO_TIMESTAMPS, EOS]],
        )

    def test_invalid_prefix_config(self):
        with self.assertRaisesRegex(ValueError, "`task` must be one of"):
            WhisperAudioToTextPreprocessor(**self.init_kwargs, task="caption")
        preprocessor = WhisperAudioToTextPreprocessor(
            **self.init_kwargs, language="fr"
        )
        with self.assertRaisesRegex(ValueError, "English-only"):
            preprocessor.decoder_prefix_ids
        preprocessor = WhisperAudioToTextPreprocessor(
            audio_converter=self.audio_converter,
            tokenizer=mock_whisper_tokenizer(multilingual=True),
            language="xx",
        )
        with self.assertRaisesRegex(ValueError, "Unknown `language`"):
            preprocessor.decoder_prefix_ids

    def test_generate_preprocess(self):
        preprocessor = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(self.input_data[0])
        self.assertAllEqual(
            keras.ops.shape(output["encoder_features"]), (1, 100, 80)
        )
        # No end token is appended, so generation continues after the text.
        self.assertAllEqual(
            output["decoder_token_ids"],
            [[BOS, NO_TIMESTAMPS, 23, 14, 24, 23, 16, EOS]],
        )
        self.assertAllEqual(
            output["decoder_padding_mask"], [[True] * 7 + [False]]
        )

    def test_generate_preprocess_without_text(self):
        preprocessor = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(
            {"audio": mock_audio((2, 16000))}
        )
        # The control prefix is added exactly once.
        self.assertAllEqual(
            output["decoder_token_ids"],
            [[BOS, NO_TIMESTAMPS] + [EOS] * 6] * 2,
        )
        self.assertAllEqual(
            output["decoder_padding_mask"], [[True] * 2 + [False] * 6] * 2
        )

    def test_generate_preprocess_scalar_text_with_batched_audio(self):
        preprocessor = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(
            {"audio": mock_audio((2, 16000)), "text": " air"}
        )
        self.assertAllEqual(
            keras.ops.shape(output["encoder_features"]), (2, 100, 80)
        )
        self.assertAllEqual(
            output["decoder_token_ids"],
            [[BOS, NO_TIMESTAMPS, AIR] + [EOS] * 5] * 2,
        )
        self.assertAllEqual(
            output["decoder_padding_mask"], [[True] * 3 + [False] * 5] * 2
        )

    def test_generate_preprocess_unbatched_audio(self):
        preprocessor = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(
            {"audio": mock_audio((16000,)), "text": " air"}
        )
        self.assertAllEqual(
            keras.ops.shape(output["encoder_features"]), (100, 80)
        )
        self.assertAllEqual(
            output["decoder_token_ids"], [BOS, NO_TIMESTAMPS, AIR] + [EOS] * 5
        )
        self.assertAllEqual(
            output["decoder_padding_mask"], [True] * 3 + [False] * 5
        )

    def test_generate_preprocess_sequence_length_override(self):
        preprocessor = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(
            {"audio": mock_audio((1, 16000)), "text": [" airport"]},
            sequence_length=5,
        )
        self.assertAllEqual(
            output["decoder_token_ids"], [[BOS, NO_TIMESTAMPS, AIR, PORT, EOS]]
        )
        self.assertAllEqual(
            output["decoder_padding_mask"], [[True] * 4 + [False]]
        )

    def test_generate_preprocess_batch_size_mismatch(self):
        preprocessor = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "Batch size mismatch"):
            preprocessor.generate_preprocess(
                {
                    "audio": mock_audio((3, 16000)),
                    "text": [" air", " airport"],
                }
            )

    @pytest.mark.skipif(tf is None, reason="Requires TensorFlow.")
    def test_generate_preprocess_in_tf_data(self):
        # Under `tf.data.Dataset.map`, `generate_preprocess` is traced in graph
        # mode and the batch dimension is symbolic, so it runs the TensorFlow
        # path with `tf.shape`.
        preprocessor = WhisperAudioToTextPreprocessor(**self.init_kwargs)
        audio = mock_audio((3, 16000))
        ds = tf.data.Dataset.from_tensor_slices({"audio": audio}).batch(3)
        output = next(iter(ds.map(preprocessor.generate_preprocess)))
        self.assertAllEqual(
            keras.ops.shape(output["encoder_features"]), (3, 100, 80)
        )
        self.assertAllEqual(
            output["decoder_token_ids"],
            [[BOS, NO_TIMESTAMPS] + [EOS] * 6] * 3,
        )
        self.assertAllEqual(
            output["decoder_padding_mask"], [[True] * 2 + [False] * 6] * 3
        )

        ds = tf.data.Dataset.from_tensor_slices(
            {"audio": audio[:2], "text": [" air", " airport"]}
        ).batch(2)
        output = next(iter(ds.map(preprocessor.generate_preprocess)))
        self.assertAllEqual(
            output["decoder_token_ids"],
            [
                [BOS, NO_TIMESTAMPS, AIR] + [EOS] * 5,
                [BOS, NO_TIMESTAMPS, AIR, PORT] + [EOS] * 4,
            ],
        )
        self.assertAllEqual(
            output["decoder_padding_mask"],
            [[True] * 3 + [False] * 5, [True] * 4 + [False] * 4],
        )

    def test_generate_postprocess(self):
        preprocessor = WhisperAudioToTextPreprocessor(
            audio_converter=self.audio_converter,
            tokenizer=mock_whisper_tokenizer(multilingual=True),
        )
        # Control tokens, language tokens, eos and masked positions are
        # stripped.
        input_data = {
            "decoder_token_ids": np.array(
                [
                    [BOS, EN, TRANSCRIBE, NO_TIMESTAMPS, AIR, PORT, EOS, EOS],
                    [BOS, FR, TRANSLATE, NO_TIMESTAMPS, AIR, AIR, PORT, 9],
                ]
            ),
            "decoder_padding_mask": np.array(
                [[True] * 7 + [False], [True] * 6 + [False] * 2]
            ),
        }
        output = preprocessor.generate_postprocess(input_data)
        self.assertEqual(output, [" airport", " air air"])

        # Unbatched input.
        output = preprocessor.generate_postprocess(
            {
                "decoder_token_ids": np.array([BOS, EN, AIR, PORT, EOS]),
                "decoder_padding_mask": np.array([True] * 4 + [False]),
            }
        )
        self.assertEqual(output, " airport")

    @pytest.mark.skipif(tf is None, reason="Requires TensorFlow.")
    def test_python_matches_tf(self):
        preprocessor = WhisperAudioToTextPreprocessor(
            audio_converter=self.audio_converter,
            tokenizer=mock_whisper_tokenizer(multilingual=True),
            decoder_sequence_length=8,
        )
        for inputs in [
            {"audio": mock_audio((2, 16000))},
            {"audio": mock_audio((2, 16000)), "text": " air"},
            {"audio": mock_audio((2, 16000)), "text": [" air", " airport"]},
            {"audio": mock_audio((16000,)), "text": " airport"},
        ]:
            python_output = preprocessor._generate_preprocess_python(inputs)
            tf_output = preprocessor._generate_preprocess_tf(inputs)
            for key in python_output:
                self.assertAllClose(
                    python_output[key], tf_output[key], atol=1e-5
                )
            python_text = preprocessor._generate_postprocess_python(
                python_output
            )
            tf_text = preprocessor._generate_postprocess_tf(tf_output)
            self.assertEqual(python_text, tf_text)

    def test_serialization(self):
        preprocessor = WhisperAudioToTextPreprocessor(
            audio_converter=self.audio_converter,
            tokenizer=mock_whisper_tokenizer(multilingual=True),
            language="fr",
            task="translate",
            return_timestamps=True,
        )
        self.run_serialization_test(instance=preprocessor)
        restored = WhisperAudioToTextPreprocessor.from_config(
            preprocessor.get_config()
        )
        self.assertEqual(
            restored.decoder_prefix_ids, preprocessor.decoder_prefix_ids
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=WhisperAudioToTextPreprocessor,
            preset="whisper_tiny_en",
            input_data=self.input_data[0],
        )
