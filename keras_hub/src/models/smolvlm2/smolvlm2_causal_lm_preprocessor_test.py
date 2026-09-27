"""Tests for SmolVLM2CausalLMPreprocessor."""

import re

import numpy as np
from keras import ops

from keras_hub.src.models.smolvlm2.smolvlm2_causal_lm_preprocessor import (
    SmolVLM2CausalLMPreprocessor,
)
from keras_hub.src.models.smolvlm2.smolvlm2_causal_lm_preprocessor import (
    _get_image_prompt_string,
)
from keras_hub.src.models.smolvlm2.smolvlm2_causal_lm_preprocessor import (
    _number_to_words,
)
from keras_hub.src.models.smolvlm2.smolvlm2_image_converter import (
    SmolVLM2ImageConverter,
)
from keras_hub.src.models.smolvlm2.smolvlm2_tokenizer import SmolVLM2Tokenizer
from keras_hub.src.models.smolvlm2.smolvlm2_video_converter import (
    SmolVLM2VideoConverter,
)
from keras_hub.src.tests.test_case import TestCase


class SmolVLM2CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += [
            "Ġa t",
            "p o",
            "r t",
            "Ġt h",
            "ai r",
            "pl a",
            "po rt",
        ]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab += ["!"]
        self.vocab += ["<|begin_of_text|>"]
        self.vocab += ["<|end_of_text|>"]
        self.vocab += ["<image>"]
        self.vocab += ["<end_of_utterance>"]
        self.vocab += ["<|im_start|>"]
        self.vocab += ["<|im_end|>"]
        self.vocab += ["<fake_token_around_image>"]
        self.vocab += ["<global-img>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.tokenizer = SmolVLM2Tokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = [" airplane at airport"]

    def test_preprocessor_basics(self):
        # " airplane at airport" tokenizes to [23, 14, 24, 23, 16].
        # call() packs (prompts, responses) = same text duplicated.
        # Packer with seq_length=9 (8+1): [23,14,24,23, 23,14,24,23, 35]
        # → truncated. token_ids[:-1] = [23,14,24,23, 23,14,24,23].
        preprocessor = SmolVLM2CausalLMPreprocessor(**self.init_kwargs)
        x, y, sw = preprocessor(self.input_data)

        self.assertAllEqual(x["token_ids"], [[23, 14, 24, 23, 23, 14, 24, 23]])
        self.assertIn("padding_mask", x)

    def test_with_start_end_token(self):
        input_data = [" airplane at airport"] * 4
        preprocessor = SmolVLM2CausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=True,
            add_end_token=True,
        )
        x, y, sw = preprocessor(input_data)
        # start=34, [23,14,24,23, 23,14,24] truncated to 8 positions.
        self.assertAllEqual(
            x["token_ids"], [[34, 23, 14, 24, 23, 23, 14, 24]] * 4
        )

    def test_generate_preprocess_text_only(self):
        input_data = " airplane at airport"
        preprocessor = SmolVLM2CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)
        # Text-only should NOT have vision keys.
        self.assertNotIn("pixel_values", x)
        self.assertNotIn("vision_indices", x)

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [23, 14, 24, 23, 16, 0, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = SmolVLM2CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, " airplane at airport")

    def test_generate_preprocess_with_image(self):
        """Multimodal prompt with an image produces vision keys."""
        image_converter = SmolVLM2ImageConverter(
            max_image_size=32,
            size=64,
            do_image_splitting=False,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
            interpolation="bicubic",
        )
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=image_converter,
            sequence_length=128,
            image_seq_len=4,  # Small for testing.
        )

        img = np.random.randint(0, 256, size=(20, 20, 3)).astype("uint8")
        prompt = (
            "<|im_start|>User:<image>describe<end_of_utterance>\nAssistant:"
        )

        x = preprocessor.generate_preprocess({"prompts": prompt, "images": img})

        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)
        self.assertIn("pixel_values", x)
        self.assertIn("vision_indices", x)

        # pixel_values should be (1, 32, 32, 3) — single crop, no splitting.
        pixel_values = ops.convert_to_numpy(x["pixel_values"])
        self.assertEqual(pixel_values.shape[1], 32)
        self.assertEqual(pixel_values.shape[2], 32)
        self.assertEqual(pixel_values.shape[3], 3)

        # One row per prompt, with `image_seq_len=4` slots for the crop.
        vision_indices = ops.convert_to_numpy(x["vision_indices"])
        self.assertEqual(vision_indices.shape, (1, 4))

    def test_prompt_expansion_unsplit(self):
        """Unsplit image produces <fake><global-img><image>×N<fake> format."""
        result = _get_image_prompt_string(
            image_seq_len=3,
            image_rows=0,
            image_cols=0,
            fake_token_around_image="<fake_token_around_image>",
            image_token="<image>",
            global_image_token="<global-img>",
        )
        # Should be: <fake><global-img><image><image><image><fake>
        self.assertIn("<fake_token_around_image>", result)
        self.assertIn("<global-img>", result)
        self.assertEqual(result.count("<image>"), 3)

    def test_prompt_expansion_split(self):
        """Split image produces per-patch <row_R_col_C> + global view."""
        result = _get_image_prompt_string(
            image_seq_len=2,
            image_rows=2,
            image_cols=3,
            fake_token_around_image="<fake_token_around_image>",
            image_token="<image>",
            global_image_token="<global-img>",
        )
        # 2×3 = 6 patches + 1 global = 7 sub-images × 2 tokens = 14.
        self.assertEqual(result.count("<image>"), 14)
        # Should contain row/col tags.
        self.assertIn("<row_1_col_1>", result)
        self.assertIn("<row_2_col_3>", result)
        # Should contain global tag.
        self.assertIn("<global-img>", result)

    def test_special_token_tokenization(self):
        """_tokenize_with_special_tokens preserves special tokens."""
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=32,
        )
        if not preprocessor.built:
            preprocessor.build(None)

        text = "<|im_start|> air<end_of_utterance>"
        ids = preprocessor._tokenize_with_special_tokens(text)

        # <|im_start|> should be a single ID.
        start_id = self.tokenizer.start_token_id
        eou_id = self.tokenizer.end_of_utterance_token_id
        self.assertEqual(ids[0], start_id)
        self.assertEqual(ids[-1], eou_id)

    def test_generate_preprocess_with_video(self):
        """Video prompt with <video> produces vision keys."""
        video_converter = SmolVLM2VideoConverter(
            max_image_size=32,
            size=64,
            num_frames=3,
            fps=1,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
            interpolation="bicubic",
        )
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=video_converter,
            sequence_length=512,
            image_seq_len=4,
        )

        # Fake video: 6 frames of 48x64.
        video = np.random.randint(0, 256, size=(6, 48, 64, 3)).astype("uint8")
        prompt = (
            "<|im_start|>User:<video>describe<end_of_utterance>\nAssistant:"
        )

        x = preprocessor.generate_preprocess(
            {"prompts": prompt, "videos": video}
        )

        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)
        self.assertIn("pixel_values", x)
        self.assertIn("vision_indices", x)

        # pixel_values should be (3, 32, 32, 3) — 3 sampled frames.
        pixel_values = ops.convert_to_numpy(x["pixel_values"])
        self.assertEqual(pixel_values.shape[0], 3)
        self.assertEqual(pixel_values.shape[1], 32)
        self.assertEqual(pixel_values.shape[2], 32)
        self.assertEqual(pixel_values.shape[3], 3)

        # One row per prompt: 3 frames x `image_seq_len=4` slots.
        vision_indices = ops.convert_to_numpy(x["vision_indices"])
        self.assertEqual(vision_indices.shape, (1, 12))

    def test_video_prompt_expansion(self):
        """Video prompt string has per-frame timestamps."""
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=32,
            image_seq_len=2,
        )
        if not preprocessor.built:
            preprocessor.build(None)

        prompt = preprocessor._get_video_prompt_string(
            num_frames=3,
            metadata={"fps": 1, "duration": 3},
        )

        # Should contain video intro.
        self.assertIn("three frames", prompt)
        self.assertIn("[H:MM:SS]", prompt)

        # Should have per-frame timestamps.
        self.assertIn("Frame from 00:00:", prompt)
        self.assertIn("Frame from 00:01:", prompt)
        self.assertIn("Frame from 00:02:", prompt)

        # Each frame gets image_seq_len=2 <image> tokens.
        self.assertEqual(prompt.count("<image>"), 6)  # 3 frames × 2

        # Each frame wrapped with <fake_token_around_image>.
        self.assertIn("<fake_token_around_image>", prompt)
        self.assertIn("<global-img>", prompt)

    def _image_preprocessor(self, **overrides):
        kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": SmolVLM2ImageConverter(
                max_image_size=32,
                size=64,
                do_image_splitting=False,
                scale=[1 / 255.0] * 3,
                offset=[0.0] * 3,
                interpolation="bicubic",
            ),
            "sequence_length": 128,
            "image_seq_len": 4,
        }
        kwargs.update(overrides)
        return SmolVLM2CausalLMPreprocessor(**kwargs)

    def test_generate_preprocess_with_batched_images(self):
        """Every prompt/image in a batch is preprocessed, not just the first."""
        preprocessor = self._image_preprocessor()
        images = np.random.randint(0, 256, size=(3, 20, 20, 3)).astype("uint8")
        prompts = [
            "<|im_start|>User:<image>describe<end_of_utterance>\nAssistant:",
            "<|im_start|>User:<image>what is this<end_of_utterance>\n"
            "Assistant:",
            "<|im_start|>User:<image>caption<end_of_utterance>\nAssistant:",
        ]

        x = preprocessor.generate_preprocess(
            {"prompts": prompts, "images": images}
        )

        self.assertEqual(ops.shape(x["token_ids"])[0], 3)
        pixel_values = ops.convert_to_numpy(x["pixel_values"])
        self.assertEqual(pixel_values.shape, (3, 32, 32, 3))
        # Four `<image>` slots per prompt, as flat `batch * seq + pos`
        # offsets into the packed sequence.
        vision_indices = ops.convert_to_numpy(x["vision_indices"])
        self.assertEqual(vision_indices.shape, (3, 4))
        sequence_length = ops.shape(x["token_ids"])[1]
        for i, row in enumerate(vision_indices):
            self.assertTrue(np.all(row // int(sequence_length) == i))

    def test_generate_preprocess_image_count_mismatch(self):
        """Mismatched prompt/image counts must raise, not silently truncate."""
        preprocessor = self._image_preprocessor()
        images = np.random.randint(0, 256, size=(2, 20, 20, 3)).astype("uint8")
        prompts = ["<image>a", "<image>b", "<image>c"]
        with self.assertRaisesRegex(ValueError, "images"):
            preprocessor.generate_preprocess(
                {"prompts": prompts, "images": images}
            )

    def test_generate_preprocess_truncated_image_tokens(self):
        """Too short a sequence must raise instead of misaligning vision."""
        preprocessor = self._image_preprocessor(sequence_length=4)
        img = np.random.randint(0, 256, size=(20, 20, 3)).astype("uint8")
        with self.assertRaisesRegex(ValueError, "image_seq_len"):
            preprocessor.generate_preprocess(
                {"prompts": "<image>describe", "images": img}
            )

    def test_generate_preprocess_uneven_crops_raises(self):
        """Prompts expanding to different crop counts must raise clearly."""
        preprocessor = self._image_preprocessor(
            sequence_length=512,
            image_converter=SmolVLM2ImageConverter(
                max_image_size=32,
                size=64,
                do_image_splitting=True,
                scale=[1 / 255.0] * 3,
                offset=[0.0] * 3,
            ),
        )
        # Crop count follows the aspect ratio: the wide image snaps to a
        # 1x2 grid plus a global view, the square one to a 2x2 grid plus a
        # global view, so the prompts expand to a different number of
        # `<image>` tokens.
        images = [
            np.random.randint(0, 256, size=(16, 64, 3)).astype("uint8"),
            np.random.randint(0, 256, size=(64, 64, 3)).astype("uint8"),
        ]
        with self.assertRaisesRegex(ValueError, "same number"):
            preprocessor.generate_preprocess(
                {"prompts": ["<image>a", "<image>b"], "images": images}
            )

    def test_config_roundtrip(self):
        """`image_seq_len` must survive serialization."""
        preprocessor = self._image_preprocessor(image_seq_len=7)
        config = preprocessor.get_config()
        self.assertEqual(config["image_seq_len"], 7)
        restored = SmolVLM2CausalLMPreprocessor.from_config(config)
        self.assertEqual(restored.image_seq_len, 7)

    def test_call_with_images(self):
        """The training path expands `<image>` and emits vision indices."""
        preprocessor = self._image_preprocessor()
        images = np.random.randint(0, 256, size=(2, 20, 20, 3)).astype("uint8")
        x, y, sw = preprocessor(
            {
                "prompts": ["<image>describe", "<image>caption"],
                "responses": [" airplane", " airport"],
                "images": images,
            }
        )
        pixel_values = ops.convert_to_numpy(x["pixel_values"])
        self.assertEqual(pixel_values.shape, (2, 32, 32, 3))
        vision_indices = ops.convert_to_numpy(x["vision_indices"])
        self.assertEqual(vision_indices.shape, (2, 4))
        token_ids = ops.convert_to_numpy(x["token_ids"])
        image_token_id = self.tokenizer.image_token_id
        self.assertEqual(int((token_ids == image_token_id).sum()), 8)

    def test_call_text_only_emits_empty_vision(self):
        """Text-only training batches carry zero-length `pixel_values`."""
        preprocessor = self._image_preprocessor()
        x, y, sw = preprocessor([" airplane at airport"] * 2)
        pixel_values = ops.convert_to_numpy(x["pixel_values"])
        self.assertEqual(pixel_values.shape, (0, 32, 32, 3))
        self.assertEqual(ops.shape(x["vision_indices"]), (2, 0))

    def test_call_with_splitting_raises(self):
        """Image splitting cannot be resolved inside a `tf.data` graph."""
        preprocessor = self._image_preprocessor(
            image_converter=SmolVLM2ImageConverter(
                max_image_size=32,
                size=64,
                do_image_splitting=True,
                scale=[1 / 255.0] * 3,
                offset=[0.0] * 3,
            )
        )
        img = np.random.randint(0, 256, size=(1, 20, 20, 3)).astype("uint8")
        with self.assertRaisesRegex(ValueError, "do_image_splitting"):
            preprocessor(
                {
                    "prompts": ["<image>describe"],
                    "responses": [" airplane"],
                    "images": img,
                }
            )

    def test_call_with_videos_raises(self):
        """Videos are inference-only."""
        preprocessor = self._image_preprocessor()
        video = np.random.randint(0, 256, size=(1, 3, 20, 20, 3)).astype(
            "uint8"
        )
        with self.assertRaisesRegex(ValueError, "video"):
            preprocessor(
                {
                    "prompts": ["<video>describe"],
                    "responses": [" airplane"],
                    "videos": video,
                }
            )

    def test_generate_preprocess_text_only_respects_add_start_token(self):
        """Text-only generation honours `add_start_token` like other paths."""
        preprocessor = SmolVLM2CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(" airplane at airport")
        # The default `add_start_token=False` adds no `<|im_start|>` (34).
        self.assertAllEqual(x["token_ids"], [23, 14, 24, 23, 16, 0, 0, 0])
        preprocessor = SmolVLM2CausalLMPreprocessor(
            **self.init_kwargs, add_start_token=True
        )
        x = preprocessor.generate_preprocess(" airplane at airport")
        self.assertAllEqual(x["token_ids"], [34, 23, 14, 24, 23, 16, 0, 0])

    def _row_col_preprocessor(self):
        vocab = dict(self.vocab)
        vocab["<row_1_col_1>"] = len(vocab)
        tokenizer = SmolVLM2Tokenizer(vocabulary=vocab, merges=self.merges)
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=8
        )
        return preprocessor, vocab["<row_1_col_1>"]

    def test_generate_postprocess_strips_row_col_tags(self):
        """Crop tags from split images do not leak into decoded text."""
        preprocessor, tag_id = self._row_col_preprocessor()
        x = preprocessor.generate_postprocess(
            {
                "token_ids": [tag_id, 23, 14, 0, 0, 0, 0, 0],
                "padding_mask": [1, 1, 1, 0, 0, 0, 0, 0],
            }
        )
        self.assertAllEqual(x, " airplane")

    def test_row_col_tags_tokenize_to_single_ids(self):
        """Prompts map crop tags to the tokenizer's tag ids."""
        preprocessor, tag_id = self._row_col_preprocessor()
        ids = preprocessor._tokenize_with_special_tokens("<row_1_col_1> air")
        self.assertEqual([int(i) for i in ids], [tag_id, 23])

    def test_number_to_words_matches_num2words(self):
        """The frame count is spelled out like HF's `num2words`."""
        # Expected strings are `num2words(n)` outputs, measured on the VM.
        for number, words in [
            (1, "one"),
            (8, "eight"),
            (20, "twenty"),
            (21, "twenty-one"),
            (32, "thirty-two"),
            (45, "forty-five"),
            (64, "sixty-four"),
            (100, "one hundred"),
            (101, "one hundred and one"),
            (999, "nine hundred and ninety-nine"),
        ]:
            self.assertEqual(_number_to_words(number), words)

    def test_number_to_words_rejects_out_of_range(self):
        """Only `[0, 1000)` is supported; anything else raises."""
        for number in (-1, 1000):
            with self.assertRaisesRegex(ValueError, "number="):
                _number_to_words(number)

    def test_video_timestamps_follow_sampled_indices(self):
        """Timestamps use each sampled frame's source index, as in HF."""
        video_converter = SmolVLM2VideoConverter(
            max_image_size=32,
            size=64,
            num_frames=4,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=video_converter,
            image_seq_len=2,
        )
        video = np.zeros((128, 8, 8, 3), dtype="uint8")
        video_output = preprocessor._preprocess_video(video)
        self.assertEqual(video_output["frames_indices"], [0, 42, 84, 127])
        prompt = preprocessor._get_video_prompt_string(
            num_frames=video_output["num_frames"],
            metadata={"fps": 8},
            frames_indices=video_output["frames_indices"],
        )
        # `index / fps` is 0, 5.25, 10.5 and 15.875 seconds.
        self.assertEqual(
            re.findall(r"Frame from (\d\d:\d\d):", prompt),
            ["00:00", "00:05", "00:10", "00:15"],
        )
        self.assertIn("series of four frames from a 0:00:15 [H:MM:SS]", prompt)

    def test_video_prompt_defaults_to_fps_24(self):
        """Without `fps` metadata, HF assumes 24 fps for the timestamps."""
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=self.tokenizer, image_seq_len=2
        )
        prompt = preprocessor._get_video_prompt_string(num_frames=30)
        self.assertEqual(
            re.findall(r"Frame from (\d\d:\d\d):", prompt),
            ["00:00"] * 24 + ["00:01"] * 6,
        )
        # The duration is `int(29 / 24)` seconds.
        self.assertIn(
            "series of thirty frames from a 0:00:01 [H:MM:SS]", prompt
        )

    def test_video_metadata_frames_indices_map_sampled_frames(self):
        """`video_metadata["frames_indices"]` maps input frames to source."""
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=self.tokenizer, image_seq_len=2
        )
        prompt = preprocessor._get_video_prompt_string(
            num_frames=2,
            metadata={"fps": 2, "frames_indices": [0, 4, 8]},
            frames_indices=[0, 2],
        )
        self.assertEqual(
            re.findall(r"Frame from (\d\d:\d\d):", prompt), ["00:00", "00:04"]
        )
        with self.assertRaisesRegex(ValueError, "one entry per input frame"):
            preprocessor._get_video_prompt_string(
                num_frames=2,
                metadata={"frames_indices": [0]},
                frames_indices=[0, 2],
            )

    def test_video_prompt_uses_metadata_duration(self):
        """`video_metadata["duration"]` sets the duration, as in HF."""
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=self.tokenizer, image_seq_len=2
        )
        prompt = preprocessor._get_video_prompt_string(
            num_frames=4, metadata={"fps": 8, "duration": 20}
        )
        # Without it the duration would be `int(3 / 8)`, i.e. 0:00:00.
        self.assertIn("four frames from a 0:00:20 [H:MM:SS]", prompt)
