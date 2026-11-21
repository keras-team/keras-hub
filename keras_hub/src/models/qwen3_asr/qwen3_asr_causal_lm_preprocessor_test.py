import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_causal_lm_preprocessor import (
    Qwen3ASRCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_causal_lm_preprocessor import (
    _get_audio_token_length,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import Qwen3ASRTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import tf


class Qwen3ASRCausalLMPreprocessorTest(TestCase):
    def test_audio_token_length_uses_window_size(self):
        token_length = _get_audio_token_length(np.array([50]), n_window=25)

        self.assertEqual(token_length.tolist(), [7])

    def test_audio_token_length_supports_numpy_and_tf(self):
        audio_lengths = np.array([0, 1, 99, 100, 101, 200], dtype="int32")
        expected = np.array([0, 1, 13, 13, 14, 26], dtype="int32")

        numpy_result = _get_audio_token_length(audio_lengths)
        self.assertAllEqual(numpy_result, expected)

        if tf is not None:
            tf_result = _get_audio_token_length(tf.constant(audio_lengths))
            self.assertAllEqual(tf_result, expected)

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

        self.audio_converter = Qwen3ASRAudioConverter()

        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "audio_converter": self.audio_converter,
            "sequence_length": 40,
        }

    def test_preprocessor_basics(self):
        expected_token_ids = np.array(
            [4]
            + [3] * 7
            + [1, 31, 22, 32, 31, 24, 31, 22, 32, 31, 24, 6]
            + [5] * 20,
            dtype="int32",
        )
        expected_padding_mask = (
            expected_token_ids != self.vocab["<|endoftext|>"]
        )
        expected_labels = np.array(
            [3] * 7 + [1, 31, 22, 32, 31, 24, 31, 22, 32, 31, 24, 6] + [5] * 21,
            dtype="int32",
        )
        expected_sample_weight = np.array(
            [False] * 13 + [True] * 6 + [False] * 21,
            dtype=bool,
        )
        audio = [np.zeros((8000,), dtype="float32")]
        expected_audio_mel = self.audio_converter(audio)

        self.run_preprocessor_test(
            cls=Qwen3ASRCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data={
                "prompts": [" airplane at airport"],
                "responses": [" airplane at airport"],
                "audio": [np.zeros((8000,), dtype="float32")],
            },
            expected_output=(
                {
                    "token_ids": expected_token_ids[None, :],
                    "padding_mask": expected_padding_mask[None, :],
                    "audio_mel": expected_audio_mel,
                    "audio_mask": np.ones((1, 50), dtype="int32"),
                },
                expected_labels[None, :],
                expected_sample_weight[None, :],
            ),
        )

    def test_with_start_end_token(self):
        expected_token_ids = np.array(
            [4]
            + [3] * 13
            + [1, 31, 22, 32, 31, 24, 31, 22, 32, 31, 24, 6]
            + [5] * 14,
            dtype="int32",
        )
        expected_padding_mask = (
            expected_token_ids != self.vocab["<|endoftext|>"]
        )
        expected_labels = np.array(
            [3] * 13
            + [1, 31, 22, 32, 31, 24, 31, 22, 32, 31, 24, 6]
            + [5] * 15,
            dtype="int32",
        )
        expected_sample_weight = np.array(
            [False] * 19 + [True] * 6 + [False] * 15,
            dtype=bool,
        )

        input_data = {
            "prompts": [" airplane at airport"] * 4,
            "responses": [" airplane at airport"] * 4,
            "audio": [np.ones((16000,), dtype="float32")] * 4,
        }
        preprocessor = Qwen3ASRCausalLMPreprocessor(
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
        preprocessor = Qwen3ASRCausalLMPreprocessor(**self.init_kwargs)
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
        self.assertTrue(ops.is_tensor(output["audio_mel"]))
        self.assertTrue(ops.is_tensor(output["audio_mask"]))

    def test_generate_preprocess(self):
        input_data = {
            "prompts": " airplane",
            "audio": np.ones((16000,)),
        }
        preprocessor = Qwen3ASRCausalLMPreprocessor(**self.init_kwargs)
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

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_generate_preprocess_tf_data(self):
        preprocessor = Qwen3ASRCausalLMPreprocessor(**self.init_kwargs)
        ds = (
            tf.data.Dataset.from_tensor_slices(
                {
                    "prompts": [" airplane", " airplane <audio> at airport"],
                    "audio": np.zeros((2, 8000), dtype="float32"),
                }
            )
            .batch(2)
            .map(preprocessor.generate_preprocess)
        )
        batch = next(iter(ds))
        self.assertEqual(batch["token_ids"].shape, (2, 40))
        self.assertEqual(batch["padding_mask"].shape, (2, 40))
        self.assertEqual(batch["audio_mel"].shape, (2, 50, 128))
        self.assertEqual(batch["audio_mask"].shape, (2, 50))
        expected_first = np.array(
            [4] + [3] * 7 + [1, 31, 22] + [5] * 29, dtype="int32"
        )
        self.assertAllEqual(batch["token_ids"][0], expected_first)
        expected_second_valid = ops.convert_to_numpy(
            self.tokenizer(
                " airplane <|audio_start|>"
                + "<|audio_pad|>" * 7
                + "<|audio_end|> at airport"
            )
        )
        second_ids = ops.convert_to_numpy(batch["token_ids"][1])
        second_mask = ops.convert_to_numpy(batch["padding_mask"][1]).astype(
            bool
        )
        self.assertAllEqual(second_ids[second_mask], expected_second_valid)

        def gen():
            yield {
                "prompts": " airplane",
                "audio": np.zeros((8000,), dtype="float32"),
            }

        unbatched_ds = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                "prompts": tf.TensorSpec(shape=(), dtype=tf.string),
                "audio": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            },
        ).map(preprocessor.generate_preprocess)
        sample = next(iter(unbatched_ds))
        self.assertEqual(sample["token_ids"].shape, (40,))
        self.assertEqual(sample["audio_mel"].shape, (50, 128))
        self.assertEqual(sample["audio_mask"].shape, (50,))

    def test_truncated_audio_placeholders_raise(self):
        preprocessor = Qwen3ASRCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            audio_converter=self.audio_converter,
            sequence_length=10,
        )
        with self.assertRaisesRegex(ValueError, "audio_pad"):
            preprocessor.generate_preprocess(
                {
                    "prompts": " airplane",
                    "audio": np.zeros((16000,), dtype="float32"),
                },
                sequence_length=10,
            )

        with self.assertRaisesRegex(ValueError, "audio_pad"):
            preprocessor(
                {
                    "prompts": [" airplane"],
                    "responses": [" airplane"],
                    "audio": [np.zeros((16000,), dtype="float32")],
                },
                sequence_length=10,
            )

        if tf is not None:
            ds = (
                tf.data.Dataset.from_tensor_slices(
                    {
                        "prompts": [" airplane"],
                        "audio": np.zeros((1, 16000), dtype="float32"),
                    }
                )
                .batch(1)
                .map(preprocessor.generate_preprocess)
            )
            with self.assertRaises(tf.errors.InvalidArgumentError):
                next(iter(ds))

    def test_audio_placeholder_is_replaced_in_place(self):
        prompt = "transcribe this: <audio> now"
        audio = np.ones((16000,), dtype="float32")
        preprocessor = Qwen3ASRCausalLMPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(
            {"prompts": [prompt], "audio": [audio]}
        )

        formatted_prompt = prompt.replace(
            "<audio>",
            "<|audio_start|>" + "<|audio_pad|>" * 13 + "<|audio_end|>",
        )
        expected_token_ids = ops.convert_to_numpy(
            self.tokenizer(formatted_prompt)
        )
        token_ids = ops.convert_to_numpy(output["token_ids"])[0]
        padding_mask = ops.convert_to_numpy(output["padding_mask"])[0]

        self.assertAllEqual(token_ids[padding_mask], expected_token_ids)

    def test_short_audio_uses_padded_audio_token_count(self):
        preprocessor = Qwen3ASRCausalLMPreprocessor(**self.init_kwargs)
        output = preprocessor(
            {"prompts": [""], "audio": [np.ones(4800, dtype="float32")]}
        )
        token_ids = ops.convert_to_numpy(output["token_ids"])

        self.assertEqual(
            np.count_nonzero(token_ids == self.vocab["<|audio_pad|>"]), 7
        )

    def test_long_audio_uses_full_audio_token_count(self):
        preprocessor = Qwen3ASRCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            audio_converter=Qwen3ASRAudioConverter(),
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
        preprocessor = Qwen3ASRCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, " airplane at airport")

    def test_text_only_when_audio_converter_none(self):
        preprocessor = Qwen3ASRCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            audio_converter=None,
            sequence_length=20,
        )
        output = preprocessor({"prompts": " airplane"})
        self.assertNotIn("audio_mel", output)
        self.assertNotIn("audio_mask", output)
        self.assertEqual(output["token_ids"].shape, (20,))

        with self.assertRaises(ValueError):
            Qwen3ASRCausalLMPreprocessor(**self.init_kwargs)(
                {"prompts": " airplane", "audio": None}
            )

    def test_mismatched_lengths_and_multiple_audio_placeholders_raise(self):
        preprocessor = Qwen3ASRCausalLMPreprocessor(**self.init_kwargs)
        with self.assertRaises(ValueError):
            preprocessor(
                {
                    "prompts": [" airplane", " airport"],
                    "audio": [np.ones((8000,), dtype="float32")],
                }
            )
        with self.assertRaises(ValueError):
            preprocessor(
                {
                    "prompts": ["<audio> and <audio>"],
                    "audio": [np.ones((8000,), dtype="float32")],
                }
            )

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_eager_tf_prompts_with_ragged_audio(self):
        preprocessor = Qwen3ASRCausalLMPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(
            {
                "prompts": tf.constant([" airplane", " airport"]),
                "audio": [
                    np.zeros((8000,), dtype="float32"),
                    np.zeros((16000,), dtype="float32"),
                ],
            }
        )
        self.assertEqual(output["token_ids"].shape, (2, 40))
        self.assertEqual(output["audio_mel"].shape, (2, 100, 128))
        self.assertEqual(int(ops.sum(output["audio_mask"][0])), 50)
        self.assertEqual(int(ops.sum(output["audio_mask"][1])), 100)

    def test_backend_tensor_1d_and_2d_single_clip_audio(self):
        preprocessor = Qwen3ASRCausalLMPreprocessor(**self.init_kwargs)
        for audio in (
            ops.ones((16000,), dtype="float32"),
            ops.ones((1, 16000), dtype="float32"),
        ):
            output = preprocessor.generate_preprocess(
                {"prompts": " airplane", "audio": audio}
            )
            self.assertEqual(int(ops.sum(output["audio_mask"])), 100)
            self.assertEqual(
                np.count_nonzero(
                    ops.convert_to_numpy(output["token_ids"])
                    == self.vocab["<|audio_pad|>"]
                ),
                13,
            )
