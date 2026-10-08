import numpy as np
from keras import ops

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_preprocessor import (
    Qwen3ASRPreprocessor,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_preprocessor import (
    _get_audio_token_length,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import Qwen3ASRTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRPreprocessorTest(TestCase):
    def test_audio_token_length_uses_window_size(self):
        token_length = _get_audio_token_length(np.array([50]), n_window=25)

        self.assertEqual(token_length.tolist(), [7])

    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += [
            "<|audio_pad|>",
            "<|audio_info|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "<|audio_start|>",
            "<|audio_end|>",
            "<asr_text>",
        ]
        self.vocab = sorted(set(self.vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.tokenizer = Qwen3ASRTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )

        self.audio_converter = Qwen3ASRAudioConverter(
            max_audio_length=1.05,
        )

        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "audio_converter": self.audio_converter,
            "sequence_length": 40,
        }

    def test_preprocessor_basics(self):
        expected_token_ids = np.array(
            [4]
            + [3] * 13
            + [1, 31, 22, 32, 31, 24, 31, 22, 32, 31, 24]
            + [5] * 15,
            dtype="int32",
        )
        expected_padding_mask = (
            expected_token_ids != self.vocab["<|endoftext|>"]
        )
        expected_labels = np.array(
            [3] * 13 + [1, 31, 22, 32, 31, 24, 31, 22, 32, 31, 24] + [5] * 16,
            dtype="int32",
        )
        expected_sample_weight = np.array(
            [False] * 19 + [True] * 6 + [False] * 15,
            dtype=bool,
        )
        audio = [np.zeros((16000,), dtype="float32")]
        expected_audio_mel = self.audio_converter(audio)

        self.run_preprocessor_test(
            cls=Qwen3ASRPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data={
                "prompts": [" airplane at airport"],
                "responses": [" airplane at airport"],
                "audio": [np.zeros((16000,), dtype="float32")],
            },
            expected_output=(
                {
                    "token_ids": expected_token_ids[None, :],
                    "padding_mask": expected_padding_mask[None, :],
                    "audio_mel": expected_audio_mel,
                    "audio_mask": np.ones((1, 100), dtype="int32"),
                },
                expected_labels[None, :],
                expected_sample_weight[None, :],
            ),
        )

    def test_with_start_end_token(self):
        expected_token_ids = np.array(
            [0]
            + [4]
            + [3] * 13
            + [1, 31, 22, 32, 31, 24, 31, 22, 32, 31, 24]
            + [5] * 14,
            dtype="int32",
        )
        expected_padding_mask = (
            expected_token_ids != self.vocab["<|endoftext|>"]
        )
        expected_labels = np.array(
            [4]
            + [3] * 13
            + [1, 31, 22, 32, 31, 24, 31, 22, 32, 31, 24]
            + [5] * 15,
            dtype="int32",
        )
        expected_sample_weight = np.array(
            [False] * 20 + [True] * 6 + [False] * 14,
            dtype=bool,
        )

        input_data = {
            "prompts": [" airplane at airport"] * 4,
            "responses": [" airplane at airport"] * 4,
            "audio": [np.ones((16000,), dtype="float32")] * 4,
        }
        preprocessor = Qwen3ASRPreprocessor(
            **self.init_kwargs,
            add_start_token=True,
            add_end_token=True,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(
            x["token_ids"],
            np.tile(expected_token_ids[None, :], (4, 1)),
        )
        self.assertAllEqual(
            x["padding_mask"],
            np.tile(expected_padding_mask[None, :], (4, 1)),
        )
        self.assertAllEqual(
            y,
            np.tile(expected_labels[None, :], (4, 1)),
        )
        self.assertAllEqual(
            sw,
            np.tile(expected_sample_weight[None, :], (4, 1)),
        )

    def test_inference_basics(self):
        input_data = {
            "prompts": [" airplane at airport"],
            "audio": [np.ones((16000,))],
        }
        preprocessor = Qwen3ASRPreprocessor(**self.init_kwargs)
        output = preprocessor(input_data)

        # Check keys
        self.assertIn("token_ids", output)
        self.assertIn("padding_mask", output)
        self.assertIn("audio_mel", output)
        self.assertIn("audio_mask", output)

        # Check shapes
        self.assertEqual(output["token_ids"].shape, (1, 40))
        self.assertEqual(output["padding_mask"].shape, (1, 40))
        self.assertEqual(len(output["audio_mel"].shape), 3)
        self.assertEqual(output["audio_mel"].shape[0], 1)
        self.assertEqual(output["audio_mel"].shape[2], 128)

        self.assertEqual(len(output["audio_mask"].shape), 2)
        self.assertEqual(output["audio_mask"].shape[0], 1)

    def test_generate_preprocess(self):
        input_data = {
            "prompts": " airplane",
            "audio": np.ones((16000,)),
        }
        preprocessor = Qwen3ASRPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(input_data)

        self.assertIn("token_ids", output)
        self.assertIn("padding_mask", output)
        self.assertIn("audio_mel", output)
        self.assertIn("audio_mask", output)

        self.assertAllEqual(
            output["token_ids"],
            np.array(
                [4] + [3] * 13 + [1, 31, 22] + [5] * 23,
                dtype="int32",
            ),
        )
        self.assertAllEqual(
            output["padding_mask"],
            np.array(
                [True] * 17 + [False] * 23,
                dtype=bool,
            ),
        )

    def test_short_audio_uses_padded_audio_token_count(self):
        preprocessor = Qwen3ASRPreprocessor(**self.init_kwargs)
        output = preprocessor(
            {"prompts": [""], "audio": [np.ones(4800, dtype="float32")]}
        )
        token_ids = ops.convert_to_numpy(output["token_ids"])

        self.assertEqual(
            np.count_nonzero(token_ids == self.vocab["<|audio_pad|>"]), 7
        )

    def test_long_audio_uses_full_audio_token_count(self):
        preprocessor = Qwen3ASRPreprocessor(
            tokenizer=self.tokenizer,
            audio_converter=Qwen3ASRAudioConverter(max_audio_length=30),
            sequence_length=600,
        )
        output = preprocessor(
            {
                "prompts": [""],
                "audio": [np.zeros(41 * 16000, dtype="float32")],
            }
        )
        token_ids = ops.convert_to_numpy(output["token_ids"])

        self.assertEqual(
            np.count_nonzero(token_ids == self.vocab["<|audio_pad|>"]), 533
        )

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": np.array(
                [31, 22, 32, 31, 24],
                dtype="int32",
            ),
            "padding_mask": np.array(
                [1, 1, 1, 1, 1],
                dtype=bool,
            ),
        }
        preprocessor = Qwen3ASRPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, " airplane at airport")
