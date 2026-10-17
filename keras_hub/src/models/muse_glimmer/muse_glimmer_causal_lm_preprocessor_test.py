import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.models.muse_glimmer.muse_glimmer_causal_lm_preprocessor import (  # noqa: E501
    MuseGlimmerCausalLMPreprocessor,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    MuseGlimmerImageConverter,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_tokenizer import (
    MuseGlimmerTokenizer,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_video_converter import (
    MuseGlimmerVideoConverter,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import tf

LAYOUT_SPECIAL_TOKENS = [
    "<|end_of_text|>",
    "<|begin_of_text|>",
    "<|finetune_right_pad|>",
    "<|eot|>",
    "<|patch|>",
    "<|video|>",
    "<|image_start|>",
    "<|image_end|>",
    "<|vid_start|>",
    "<|vid_end|>",
    "<|vid_frame_separator|>",
]

DELIMITER_TOKENS = [
    "<|image_start|>",
    "<|image_end|>",
    "<|vid_start|>",
    "<|vid_end|>",
    "<|vid_frame_separator|>",
]

# Ids from the HF `MuseGlimmerProcessor` for the vocabulary of
# `_layout_preprocessor`. Each video has 16x16 frames, and a frame gives
# 4 merged vision tokens. The ids `17`, `18` and `16` are the image start,
# patch and image end tokens. The ids `21`, `22`, `20` and `19` are the video
# start, video, separator and video end tokens. The ids `49` and `39` are
# the prompt text. The other ids are the Time text.
HF_LAYOUT_IDS = {
    "image": [17, 18, 18, 18, 18, 16, 49, 39],
    "video_two_groups": [
        *[21, 23, 30, 32, 28, 11, 46, 1, 0, 1, 44, 22, 22, 22, 22, 20],
        *[23, 30, 32, 28, 11, 46, 3, 0, 1, 44, 22, 22, 22, 22, 19, 49, 39],
    ],
    "video_sampled": [
        *[21, 23, 30, 32, 28, 11, 46, 1, 0, 1, 44, 22, 22, 22, 22, 20],
        *[23, 30, 32, 28, 11, 46, 4, 0, 1, 44, 22, 22, 22, 22, 19, 49, 39],
    ],
}


class MuseGlimmerCausalLMPreprocessorTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|end_of_text|>", "<|begin_of_text|>"]
        self.vocab += ["<|finetune_right_pad|>", "<|eot|>", "<|image|>"]
        self.vocab += ["<|video|>"]
        self.vocab += DELIMITER_TOKENS
        self.vocab = sorted(set(self.vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.tokenizer = MuseGlimmerTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.init_kwargs = {"tokenizer": self.tokenizer, "sequence_length": 8}
        self.preprocessor = MuseGlimmerCausalLMPreprocessor(**self.init_kwargs)
        self.image_converter = MuseGlimmerImageConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            max_image_tokens=64,
            scale=1 / 255.0,
        )
        self.image = np.random.randint(0, 255, (8, 8, 3)).astype("float32")
        # Map `image_token_id` to a vocabulary token, so prompts can
        # contain image placeholders.
        self.image_tokenizer = MuseGlimmerTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
            image_token_id=self.vocab["<|image|>"],
            unsplittable_tokens=["<|image|>"],
        )
        self.image_token_id = self.vocab["<|image|>"]
        self.image_preprocessor = MuseGlimmerCausalLMPreprocessor(
            tokenizer=self.image_tokenizer,
            sequence_length=8,
            image_converter=self.image_converter,
        )
        self.video_converter = MuseGlimmerVideoConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            fps=1.0,
            source_fps=1.0,
            num_frames=4,
            max_video_frame_tokens=16,
            scale=1 / 255.0,
        )
        self.video_token_id = self.vocab["<|video|>"]
        self.video_vocab = dict(self.vocab)
        for char in "Time:0123456789.s":
            if char not in self.video_vocab:
                self.video_vocab[char] = len(self.video_vocab)
        self.video_preprocessor = MuseGlimmerCausalLMPreprocessor(
            tokenizer=MuseGlimmerTokenizer(
                vocabulary=self.video_vocab,
                merges=self.merges,
                video_token_id=self.video_token_id,
                unsplittable_tokens=["<|video|>"],
            ),
            sequence_length=8,
            video_converter=self.video_converter,
        )

    def test_text_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=MuseGlimmerCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=([" airplane at airport"],),
        )

    def test_image_preprocessor_basics(self):
        # `run_preprocessor_test` builds a second layer from
        # `self.init_kwargs`.
        self.init_kwargs = {
            "tokenizer": self.image_tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 8,
        }
        # A constant image gives known `pixel_values` after scaling.
        image = np.full((8, 8, 3), 255.0, dtype="float32")
        self.run_preprocessor_test(
            cls=MuseGlimmerCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=(
                {
                    "prompts": ["<|image|> airplane", " air<|image|>"],
                    "images": np.stack([image, image]),
                },
            ),
            expected_output=(
                {
                    "token_ids": [
                        [1, 6, 7, 5, 35, 26, 2, 4],
                        [1, 35, 6, 7, 5, 2, 4, 4],
                    ],
                    "padding_mask": [
                        [1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 0, 0],
                    ],
                    # An 8x8 image gives 4 patches and one merged token.
                    "pixel_values": np.ones((2, 4, 96)),
                    "image_grid_thw": [[[1, 2, 2]], [[1, 2, 2]]],
                    "vision_indices": [[2], [3]],
                },
                [[6, 7, 5, 35, 26, 2, 4, 4], [35, 6, 7, 5, 2, 4, 4, 4]],
                # The image tokens are not labels. The start and end tokens
                # are labels.
                [[1, 0, 1, 1, 1, 1, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0]],
            ),
        )

    def test_video_preprocessor_basics(self):
        # The exact ids of the layout are in the layout tests.
        self.init_kwargs = {
            "tokenizer": self.video_preprocessor.tokenizer,
            "video_converter": self.video_converter,
            "sequence_length": 16,
        }
        video = np.full((2, 8, 8, 3), 255.0, dtype="float32")
        self.run_preprocessor_test(
            cls=MuseGlimmerCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=(
                {
                    "prompts": ["<|video|> airplane", " air<|video|>"],
                    "videos": np.stack([video, video]),
                },
            ),
        )

    def test_call_with_several_images_per_prompt(self):
        x, _, _ = self.image_preprocessor(
            {
                "prompts": "<|image|><|image|> airplane",
                "images": np.stack([self.image, self.image]),
            },
            sequence_length=16,
        )
        self.assertEqual(tuple(x["token_ids"].shape), (16,))
        self.assertEqual(tuple(x["image_grid_thw"].shape), (2, 3))
        # Each image has its own start and end tokens.
        self.assertAllEqual(x["vision_indices"], [2, 5])

    def test_call_truncated_image_tokens_raise(self):
        with self.assertRaisesRegex(ValueError, "increase `sequence_length`"):
            self.image_preprocessor(
                {"prompts": ["<|image|> airplane"], "images": [self.image]},
                sequence_length=1,
            )

    def test_call_placeholder_count_per_prompt_raises(self):
        # The image count matches the placeholders of the whole batch, but
        # not of each prompt.
        with self.assertRaisesRegex(ValueError, "one placeholder per item"):
            self.image_preprocessor(
                {
                    "prompts": ["<|image|><|image|> airplane", " airport"],
                    "images": np.stack([self.image, self.image]),
                }
            )

    def test_text_only_generate_preprocess(self):
        output = self.preprocessor.generate_preprocess(" airplane at airport")
        self.assertIn("token_ids", output)
        self.assertIn("padding_mask", output)

    def test_image_generate_preprocess_expands_placeholder(self):
        tokenizer = self.image_tokenizer
        layout = self.image_preprocessor._image_layout(3)
        ids = self.image_preprocessor._expand_vision_placeholders(
            [[self.image_token_id, 1000, 1001]], [layout], []
        )
        expected = [tokenizer.image_start_token_id]
        expected += [self.image_token_id] * 3
        expected += [tokenizer.image_end_token_id, 1000, 1001]
        self.assertEqual(ids, [expected])

    def test_expand_placeholders_across_prompts(self):
        # The second prompt uses the second image's token count.
        tokenizer = self.image_tokenizer
        start_id = tokenizer.image_start_token_id
        end_id = tokenizer.image_end_token_id
        ids = self.image_preprocessor._expand_vision_placeholders(
            [[self.image_token_id, 1000], [self.image_token_id, 1001]],
            [
                self.image_preprocessor._image_layout(1),
                self.image_preprocessor._image_layout(4),
            ],
            [],
        )
        self.assertEqual(
            ids,
            [
                [start_id, self.image_token_id, end_id, 1000],
                [start_id, *[self.image_token_id] * 4, end_id, 1001],
            ],
        )

    def test_extra_placeholder_raises(self):
        with self.assertRaisesRegex(ValueError, "more image placeholders"):
            self.image_preprocessor._expand_vision_placeholders(
                [[self.image_token_id, self.image_token_id]],
                [self.image_preprocessor._image_layout(1)],
                [],
            )

    def test_extra_image_raises(self):
        with self.assertRaisesRegex(ValueError, "2 image"):
            self.image_preprocessor.generate_preprocess(
                {"prompts": ["<|image|> airplane"], "images": [self.image] * 2}
            )

    def test_truncated_image_tokens_raise(self):
        with self.assertRaisesRegex(ValueError, "Increase `sequence_length`"):
            self.image_preprocessor.generate_preprocess(
                {"prompts": ["<|image|> airplane"], "images": [self.image]},
                sequence_length=1,
            )

    def test_image_generate_preprocess_stacks_media_grids(self):
        output = self.image_preprocessor.generate_preprocess(
            {
                "prompts": ["<|image|><|image|> airplane"],
                "images": [self.image, self.image],
            }
        )
        self.assertEqual(output["image_grid_thw"].shape, (2, 3))
        num_image_tokens = int(
            np.sum(np.asarray(output["token_ids"]) == self.image_token_id)
        )
        self.assertEqual(output["vision_indices"].shape, (num_image_tokens,))

    def test_prompts_dict_without_media(self):
        output = self.preprocessor.generate_preprocess(
            {"prompts": [" airplane at airport"]}
        )
        expected = self.preprocessor.generate_preprocess(
            [" airplane at airport"]
        )
        self.assertAllEqual(output["token_ids"], expected["token_ids"])
        self.assertAllEqual(output["padding_mask"], expected["padding_mask"])

    def test_single_prompt_with_image_is_unbatched(self):
        output = self.image_preprocessor.generate_preprocess(
            {"prompts": "<|image|> airplane", "images": self.image}
        )
        self.assertEqual(tuple(output["token_ids"].shape), (8,))
        self.assertEqual(tuple(output["padding_mask"].shape), (8,))
        self.assertEqual(tuple(output["image_grid_thw"].shape), (1, 3))
        # An 8x8 image gives one merged token after the start token and the
        # image start token.
        self.assertAllEqual(output["vision_indices"], [2])

    def test_images_with_different_sizes(self):
        wide_image = np.random.randint(0, 255, (8, 16, 3)).astype("float32")
        output = self.image_preprocessor.generate_preprocess(
            {
                "prompts": ["<|image|> airplane", "<|image|> airport"],
                "images": [self.image, wide_image],
            }
        )
        self.assertAllEqual(output["image_grid_thw"], [[1, 2, 2], [1, 2, 4]])
        # The second prompt expands its placeholder to two tokens. Its row
        # starts at position 8.
        self.assertAllEqual(output["vision_indices"], [2, 10, 11])

    def test_batched_image_array(self):
        output = self.image_preprocessor.generate_preprocess(
            {
                "prompts": ["<|image|> airplane", "<|image|> airport"],
                "images": np.stack([self.image, self.image]),
            }
        )
        self.assertEqual(tuple(output["token_ids"].shape), (2, 8))
        self.assertEqual(tuple(output["image_grid_thw"].shape), (2, 3))

    def test_single_prompt_with_video_is_unbatched(self):
        video = np.random.randint(0, 255, (2, 8, 8, 3)).astype("float32")
        output = self.video_preprocessor.generate_preprocess(
            {"prompts": "<|video|> airplane", "videos": video},
            sequence_length=16,
        )
        self.assertEqual(tuple(output["token_ids"].shape), (16,))
        self.assertAllEqual(output["image_grid_thw"], [[1, 2, 2]])
        # The vision token follows the start token, the video start token
        # and the 10 Time tokens.
        self.assertAllEqual(output["vision_indices"], [12])

    def test_videos_with_different_sizes(self):
        short_video = np.random.randint(0, 255, (2, 8, 8, 3))
        long_video = np.random.randint(0, 255, (4, 8, 16, 3))
        output = self.video_preprocessor.generate_preprocess(
            {
                "prompts": ["<|video|> airplane", "<|video|> airport"],
                "videos": [
                    short_video.astype("float32"),
                    long_video.astype("float32"),
                ],
            },
            sequence_length=48,
        )
        self.assertAllEqual(output["image_grid_thw"], [[1, 2, 2], [2, 2, 4]])
        # Each group has 10 Time tokens. The second video has 2 groups of 2
        # merged tokens each, with a separator token between them. Its row
        # starts at position 48.
        self.assertAllEqual(output["vision_indices"], [12, 60, 61, 73, 74])

    def _video_preprocessor_with_frames(self, sequence_length):
        # A video with 8 frames at `source_fps=1.0` keeps all 8 frames.
        # Each pair of frames gives one merged token for an 8x8 frame.
        converter = MuseGlimmerVideoConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            fps=1.0,
            source_fps=1.0,
            num_frames=8,
            max_video_frame_tokens=16,
        )
        return MuseGlimmerCausalLMPreprocessor(
            tokenizer=self.video_preprocessor.tokenizer,
            sequence_length=sequence_length,
            video_converter=converter,
        )

    def test_generate_preprocess_with_video_fps_per_video(self):
        preprocessor = self._video_preprocessor_with_frames(128)
        video = np.zeros((8, 8, 8, 3), dtype="float32")
        output = preprocessor.generate_preprocess(
            {
                "prompts": ["<|video|> airplane", "<|video|> airport"],
                "videos": [video, video],
                # 8 frames at 2 fps keep int(8 * 1 / 2) = 4 frames.
                "video_fps": [1.0, 2.0],
            }
        )
        self.assertAllEqual(output["image_grid_thw"], [[4, 2, 2], [2, 2, 2]])
        self.assertEqual(output["vision_indices"].shape[0], 6)

    def test_generate_preprocess_with_video_fps_per_prompt(self):
        preprocessor = self._video_preprocessor_with_frames(128)
        video = np.zeros((8, 8, 8, 3), dtype="float32")
        # The first prompt has two videos and the second prompt has one.
        output = preprocessor.generate_preprocess(
            {
                "prompts": ["<|video|><|video|> airplane", "<|video|> airport"],
                "videos": [video, video, video],
                "video_fps": [1.0, 2.0],
            }
        )
        self.assertAllEqual(
            output["image_grid_thw"], [[4, 2, 2], [4, 2, 2], [2, 2, 2]]
        )

    def test_generate_preprocess_with_video_fps_per_prompt_wrong_size(self):
        preprocessor = self._video_preprocessor_with_frames(128)
        video = np.zeros((8, 8, 8, 3), dtype="float32")
        with self.assertRaisesRegex(ValueError, "one value per video"):
            preprocessor.generate_preprocess(
                {
                    "prompts": [
                        "<|video|><|video|> airplane",
                        "<|video|> airport",
                    ],
                    "videos": [video, video, video],
                    "video_fps": [1.0, 2.0, 3.0, 4.0],
                }
            )

    def test_generate_preprocess_with_one_video_fps(self):
        preprocessor = self._video_preprocessor_with_frames(128)
        video = np.zeros((8, 8, 8, 3), dtype="float32")
        output = preprocessor.generate_preprocess(
            {
                "prompts": ["<|video|> airplane", "<|video|> airport"],
                "videos": [video, video],
                "video_fps": 2.0,
            }
        )
        self.assertAllEqual(output["image_grid_thw"], [[2, 2, 2], [2, 2, 2]])

    def test_call_with_video_fps(self):
        preprocessor = self._video_preprocessor_with_frames(128)
        videos = np.zeros((2, 8, 8, 8, 3), dtype="float32")
        prompts = ["<|video|> airplane", "<|video|> airport"]
        x, _, _ = preprocessor({"prompts": prompts, "videos": videos})
        self.assertAllEqual(x["image_grid_thw"], [[[4, 2, 2]], [[4, 2, 2]]])
        for video_fps in (2.0, [2.0, 2.0], np.array([2.0, 2.0])):
            x, _, _ = preprocessor(
                {"prompts": prompts, "videos": videos, "video_fps": video_fps}
            )
            self.assertAllEqual(x["image_grid_thw"], [[[2, 2, 2]], [[2, 2, 2]]])

    def test_call_with_video_fps_per_video_in_a_prompt(self):
        preprocessor = self._video_preprocessor_with_frames(128)
        videos = np.zeros((1, 2, 8, 8, 8, 3), dtype="float32")
        # One prompt with two videos. Both videos need the same frame count.
        x, _, _ = preprocessor(
            {
                "prompts": ["<|video|><|video|> airplane"],
                "videos": videos,
                "video_fps": [[2.0, 2.0]],
            }
        )
        self.assertAllEqual(x["image_grid_thw"], [[[2, 2, 2], [2, 2, 2]]])

    def test_video_fps_with_wrong_size_raises(self):
        preprocessor = self._video_preprocessor_with_frames(128)
        video = np.zeros((8, 8, 8, 3), dtype="float32")
        with self.assertRaisesRegex(ValueError, "one value per video"):
            preprocessor.generate_preprocess(
                {
                    "prompts": ["<|video|> airplane", "<|video|> airport"],
                    "videos": [video, video],
                    "video_fps": [1.0, 2.0, 3.0],
                }
            )
        with self.assertRaisesRegex(ValueError, "one value per prompt"):
            preprocessor(
                {
                    "prompts": ["<|video|> airplane", "<|video|> airport"],
                    "videos": np.stack([video, video]),
                    "video_fps": [[1.0, 2.0, 3.0]],
                }
            )

    def test_call_with_different_frame_counts_raises(self):
        preprocessor = self._video_preprocessor_with_frames(128)
        video = np.zeros((8, 8, 8, 3), dtype="float32")
        # 8 frames keep 8 or 4 frames at 1 or 2 fps.
        with self.assertRaisesRegex(ValueError, "same output size"):
            preprocessor(
                {
                    "prompts": ["<|video|> airplane", "<|video|> airport"],
                    "videos": np.stack([video, video]),
                    "video_fps": [1.0, 2.0],
                }
            )

    def test_call_with_different_image_sizes_raises(self):
        wide_image = np.zeros((8, 16, 3), dtype="float32")
        with self.assertRaisesRegex(ValueError, r"same size.*\(8, 8, 3\)"):
            self.image_preprocessor(
                {
                    "prompts": ["<|image|> airplane", "<|image|> airport"],
                    "images": [self.image, wide_image],
                }
            )

    def test_call_with_different_video_sizes_raises(self):
        preprocessor = self._video_preprocessor_with_frames(128)
        with self.assertRaisesRegex(ValueError, r"same size.*\(4, 8, 8, 3\)"):
            preprocessor(
                {
                    "prompts": ["<|video|> airplane", "<|video|> airport"],
                    "videos": [
                        np.zeros((8, 8, 8, 3), dtype="float32"),
                        np.zeros((4, 8, 8, 3), dtype="float32"),
                    ],
                }
            )

    def test_text_only_call_adds_empty_vision_inputs(self):
        x, y, sample_weight = self.image_preprocessor([" airplane", " airport"])
        # An 8x8 image patch has `2 * 3 * 4**2 = 96` values.
        self.assertEqual(tuple(x["pixel_values"].shape), (2, 0, 96))
        self.assertEqual(tuple(x["image_grid_thw"].shape), (2, 0, 3))
        self.assertEqual(tuple(x["vision_indices"].shape), (2, 0))
        self.assertEqual(tuple(x["token_ids"].shape), (2, 8))
        x, _, _ = self.image_preprocessor(" airplane")
        self.assertEqual(tuple(x["pixel_values"].shape), (0, 96))
        self.assertEqual(tuple(x["vision_indices"].shape), (0,))

    def test_text_only_call_without_converter_has_no_vision_inputs(self):
        x, _, _ = self.preprocessor([" airplane", " airport"])
        self.assertEqual(sorted(x.keys()), ["padding_mask", "token_ids"])

    def test_text_only_tf_data_adds_empty_vision_inputs(self):
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        dataset = tf.data.Dataset.from_tensor_slices(
            [" airplane", " airport"]
        ).batch(2)
        x, _, _ = next(iter(dataset.map(self.image_preprocessor)))
        self.assertEqual(tuple(x["pixel_values"].shape), (2, 0, 96))
        self.assertEqual(tuple(x["image_grid_thw"].shape), (2, 0, 3))
        self.assertEqual(tuple(x["vision_indices"].shape), (2, 0))

    def test_generate_preprocess_respects_add_start_token_false(self):
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        preprocessor = MuseGlimmerCausalLMPreprocessor(
            tokenizer=self.image_tokenizer,
            sequence_length=8,
            image_converter=self.image_converter,
            add_start_token=False,
        )
        inputs = {"prompts": ["<|image|> airplane"], "images": [self.image]}
        python_ids = preprocessor.generate_preprocess(inputs)["token_ids"]
        self.assertNotEqual(
            int(np.asarray(python_ids)[0, 0]),
            self.image_tokenizer.start_token_id,
        )
        dataset = tf.data.Dataset.from_tensors(
            {"prompts": ["<|image|> airplane"], "images": [self.image]}
        )
        tf_ids = next(iter(dataset.map(preprocessor.generate_preprocess)))[
            "token_ids"
        ]
        self.assertAllEqual(python_ids, tf_ids)

    def test_generate_preprocess_with_per_row_media_stack(self):
        rows = np.stack([np.stack([self.image, self.image])] * 3)
        output = self.image_preprocessor.generate_preprocess(
            {
                "prompts": ["<|image|><|image|> a"] * 3,
                "images": rows,
            }
        )
        self.assertEqual(output["image_grid_thw"].shape, (6, 3))
        self.assertEqual(output["vision_indices"].shape, (6,))

    def test_generate_preprocess_with_nested_media_lists(self):
        wide_image = np.random.randint(0, 255, (8, 16, 3)).astype("float32")
        output = self.image_preprocessor.generate_preprocess(
            {
                "prompts": ["<|image|><|image|> a", "<|image|> b"],
                "images": [[self.image, wide_image], [self.image]],
            }
        )
        self.assertAllEqual(
            output["image_grid_thw"], [[1, 2, 2], [1, 2, 4], [1, 2, 2]]
        )

    def test_generate_preprocess_with_empty_media_is_text_only(self):
        expected = self.image_preprocessor.generate_preprocess([" airplane"])
        for images in (
            [],
            np.zeros((0, 8, 8, 3), "float32"),
            np.zeros((1, 0, 8, 8, 3), "float32"),
        ):
            output = self.image_preprocessor.generate_preprocess(
                {"prompts": [" airplane"], "images": images}
            )
            self.assertAllEqual(output["token_ids"], expected["token_ids"])
            self.assertNotIn("pixel_values", output)

    def test_media_with_wrong_rank_raises(self):
        with self.assertRaisesRegex(ValueError, "Received rank"):
            self.image_preprocessor.generate_preprocess(
                {"prompts": ["<|image|> a"], "images": np.zeros((8, 8))}
            )

    def test_generate_postprocess_drops_vision_placeholders(self):
        text_ids = self.image_tokenizer(" airplane at airport")
        text_ids = [int(token_id) for token_id in np.asarray(text_ids)]
        num_image_tokens = 4
        token_ids = [self.image_token_id] * num_image_tokens + text_ids
        outputs = {
            "token_ids": np.array([token_ids], "int32"),
            "padding_mask": np.ones((1, len(token_ids)), "bool"),
        }
        text = self.image_preprocessor.generate_postprocess(outputs)
        self.assertNotIn("<|image|>", text[0])
        self.assertEqual(text[0], " airplane at airport")

    def _layout_preprocessor(self, video_fps=1.0, num_frames=4):
        """Build a preprocessor with the HF vocabulary layout.

        The vocabulary holds the real token names of the checkpoint. The
        ids match the ones of the HF processor in `HF_LAYOUT_IDS`.
        """
        vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab += list("Time:0123456789.s") + ["Ġ"] + LAYOUT_SPECIAL_TOKENS
        vocab = {token: i for i, token in enumerate(sorted(set(vocab)))}
        tokenizer = MuseGlimmerTokenizer(
            vocabulary=vocab,
            merges=self.merges,
            image_token_id=vocab["<|patch|>"],
            video_token_id=vocab["<|video|>"],
            unsplittable_tokens=LAYOUT_SPECIAL_TOKENS,
        )
        video_converter = MuseGlimmerVideoConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            fps=video_fps,
            source_fps=1.0,
            num_frames=num_frames,
            max_video_frame_tokens=16,
            scale=1 / 255.0,
        )
        image_converter = MuseGlimmerImageConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            max_image_tokens=64,
            scale=1 / 255.0,
        )
        return MuseGlimmerCausalLMPreprocessor(
            tokenizer=tokenizer,
            image_converter=image_converter,
            video_converter=video_converter,
            sequence_length=64,
            add_start_token=False,
        )

    def _layout_ids(self, preprocessor, inputs):
        output = preprocessor.generate_preprocess(inputs)
        token_ids = np.asarray(output["token_ids"])[0]
        padding_mask = np.asarray(output["padding_mask"])[0].astype(bool)
        return token_ids[padding_mask].tolist()

    def _layout_image_inputs(self, prompt="<|patch|> airplane"):
        image = np.random.randint(0, 255, (16, 16, 3)).astype("uint8")
        return {"prompts": [prompt], "images": [image]}

    def _layout_video_inputs(self, total_frames, source_fps, prompt=None):
        video = np.random.randint(0, 255, (total_frames, 16, 16, 3))
        return {
            "prompts": [prompt or "<|video|> airplane"],
            "videos": [video.astype("uint8")],
            "video_fps": source_fps,
        }

    def test_image_layout_matches_hf_ids(self):
        preprocessor = self._layout_preprocessor()
        ids = self._layout_ids(preprocessor, self._layout_image_inputs())
        self.assertEqual(ids, HF_LAYOUT_IDS["image"])

    def test_video_layout_with_two_groups_matches_hf_ids(self):
        preprocessor = self._layout_preprocessor()
        inputs = self._layout_video_inputs(total_frames=4, source_fps=1.0)
        ids = self._layout_ids(preprocessor, inputs)
        self.assertEqual(ids, HF_LAYOUT_IDS["video_two_groups"])

    def test_video_layout_with_sampled_frames_matches_hf_ids(self):
        # 10 frames at 2 fps keep the frames 0, 3, 6 and 9. The second
        # group starts at frame 6, which is 3.0 seconds.
        preprocessor = self._layout_preprocessor()
        inputs = self._layout_video_inputs(total_frames=10, source_fps=2.0)
        ids = self._layout_ids(preprocessor, inputs)
        self.assertEqual(ids, HF_LAYOUT_IDS["video_sampled"])

    def test_vocabulary_without_a_delimiter_token_raises(self):
        for token in DELIMITER_TOKENS:
            vocab = {k: v for k, v in self.video_vocab.items() if k != token}
            with self.assertRaisesRegex(ValueError, token):
                MuseGlimmerTokenizer(
                    vocabulary=vocab,
                    merges=self.merges,
                    video_token_id=self.video_token_id,
                    unsplittable_tokens=["<|video|>"],
                )

    def test_typed_delimiters_are_not_expanded_twice(self):
        preprocessor = self._layout_preprocessor()
        image = self._layout_image_inputs(
            "<|image_start|><|patch|><|image_end|> airplane"
        )
        self.assertEqual(
            self._layout_ids(preprocessor, image), HF_LAYOUT_IDS["image"]
        )
        video = self._layout_video_inputs(
            4, 1.0, "<|vid_start|><|video|><|vid_end|> airplane"
        )
        self.assertEqual(
            self._layout_ids(preprocessor, video),
            HF_LAYOUT_IDS["video_two_groups"],
        )

    def test_layout_special_token_ids_include_delimiters(self):
        tokenizer = self._layout_preprocessor().tokenizer
        for name in (
            "image_start_token_id",
            "image_end_token_id",
            "video_start_token_id",
            "video_end_token_id",
            "video_separator_token_id",
        ):
            token_id = getattr(tokenizer, name)
            self.assertIsNotNone(token_id)
            self.assertIn(token_id, tokenizer.special_token_ids)

    def test_generate_postprocess_drops_layout_tokens(self):
        preprocessor = self._layout_preprocessor()
        ids = HF_LAYOUT_IDS["video_two_groups"]
        outputs = {
            "token_ids": np.array([ids], "int32"),
            "padding_mask": np.ones((1, len(ids)), "bool"),
        }
        text = preprocessor.generate_postprocess(outputs)[0]
        # The delimiters, the vision tokens and the separator are gone.
        # The Time text and the prompt stay.
        self.assertEqual(text, "Time: 0.0sTime: 2.0s airplane")

    def test_python_and_tf_data_layouts_match(self):
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        preprocessor = self._layout_preprocessor()
        cases = [
            self._layout_image_inputs(),
            self._layout_video_inputs(4, 1.0),
            self._layout_video_inputs(10, 2.0),
            self._layout_image_inputs(
                "<|image_start|><|patch|><|image_end|> airplane"
            ),
        ]
        for inputs in cases:
            expected = self._layout_ids(preprocessor, inputs)
            dataset = tf.data.Dataset.from_tensors(inputs)
            output = next(iter(dataset.map(preprocessor.generate_preprocess)))
            token_ids = output["token_ids"].numpy()[0]
            padding_mask = output["padding_mask"].numpy()[0].astype(bool)
            self.assertEqual(token_ids[padding_mask].tolist(), expected)

    def test_python_and_tf_data_training_layouts_match(self):
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        preprocessor = self._layout_preprocessor()
        video = np.random.randint(0, 255, (1, 1, 4, 16, 16, 3)).astype("uint8")
        inputs = {
            "prompts": ["<|video|> airplane"],
            "videos": video,
            "video_fps": 1.0,
        }
        python_x, python_y, _ = preprocessor(inputs)
        dataset = tf.data.Dataset.from_tensors(inputs)
        tf_x, tf_y, _ = next(iter(dataset.map(preprocessor)))
        self.assertAllEqual(python_x["token_ids"], tf_x["token_ids"])
        self.assertAllEqual(python_y, tf_y)
        self.assertAllEqual(python_x["vision_indices"], tf_x["vision_indices"])

    @parameterized.named_parameters(
        # `(total_frames, source_fps, converter_fps)`.
        ("two_groups", 4, 1.0, 1.0),
        ("sampled_frames", 10, 2.0, 1.0),
        ("repeating_timestamp", 7, 3.0, 3.0),
        # 13 / 20 is 0.65 in float64 and 0.6499999762 in float32. HF
        # divides in float32, so the text is 0.6.
        ("float32_rounding", 21, 20.0, 20.0),
    )
    def test_video_layout_matches_the_hf_processor(
        self, total_frames, source_fps, converter_fps
    ):
        pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "MuseGlimmerProcessor"):
            self.skipTest("transformers has no MuseGlimmerProcessor.")
        preprocessor = self._layout_preprocessor(
            video_fps=converter_fps, num_frames=4
        )
        tokenizer = preprocessor.tokenizer
        tokenizer._maybe_initialized_tokenizers()
        hf_tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=tokenizer._tokenizer
        )
        processor = transformers.MuseGlimmerProcessor(
            image_processor=transformers.MuseGlimmerImageProcessor(
                patch_size=4,
                temporal_patch_size=2,
                merge_size=2,
                max_image_tokens=64,
            ),
            video_processor=transformers.MuseGlimmerVideoProcessor(
                patch_size=4,
                temporal_patch_size=2,
                merge_size=2,
                max_video_frame_tokens=16,
                num_frames=4,
                fps=converter_fps,
            ),
            tokenizer=hf_tokenizer,
        )
        inputs = self._layout_video_inputs(total_frames, source_fps)
        metadata = transformers.video_utils.VideoMetadata(
            total_num_frames=total_frames,
            fps=source_fps,
            width=16,
            height=16,
        )
        hf_inputs = processor(
            text=inputs["prompts"],
            videos=inputs["videos"],
            video_metadata=[metadata],
            do_sample_frames=True,
            return_metadata=True,
            return_tensors="pt",
        )
        hf_ids = hf_inputs["input_ids"][0].tolist()
        self.assertEqual(self._layout_ids(preprocessor, inputs), hf_ids)

    def test_image_layout_matches_the_hf_processor(self):
        pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "MuseGlimmerProcessor"):
            self.skipTest("transformers has no MuseGlimmerProcessor.")
        preprocessor = self._layout_preprocessor()
        tokenizer = preprocessor.tokenizer
        tokenizer._maybe_initialized_tokenizers()
        processor = transformers.MuseGlimmerProcessor(
            image_processor=transformers.MuseGlimmerImageProcessor(
                patch_size=4,
                temporal_patch_size=2,
                merge_size=2,
                max_image_tokens=64,
            ),
            video_processor=transformers.MuseGlimmerVideoProcessor(
                patch_size=4, temporal_patch_size=2, merge_size=2
            ),
            tokenizer=transformers.PreTrainedTokenizerFast(
                tokenizer_object=tokenizer._tokenizer
            ),
        )
        inputs = self._layout_image_inputs()
        hf_inputs = processor(
            text=inputs["prompts"], images=inputs["images"], return_tensors="pt"
        )
        hf_ids = hf_inputs["input_ids"][0].tolist()
        self.assertEqual(self._layout_ids(preprocessor, inputs), hf_ids)

    def test_media_without_converter_raises(self):
        with self.assertRaisesRegex(ValueError, "no `video_converter`"):
            self.image_preprocessor.generate_preprocess(
                {
                    "prompts": ["<|image|> airplane"],
                    "videos": np.zeros((2, 8, 8, 3), dtype="float32"),
                }
            )
