import numpy as np

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


class MuseGlimmerCausalLMPreprocessorTest(TestCase):
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
            num_frames=4,
            max_video_frame_tokens=16,
            scale=1 / 255.0,
        )
        self.video_token_id = self.vocab["<|video|>"]
        self.video_preprocessor = MuseGlimmerCausalLMPreprocessor(
            tokenizer=MuseGlimmerTokenizer(
                vocabulary=self.vocab,
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
                        [1, 5, 30, 21, 2, 4, 4, 4],
                        [1, 30, 5, 2, 4, 4, 4, 4],
                    ],
                    "padding_mask": [
                        [1, 1, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                    ],
                    # An 8x8 image gives 4 patches and one merged token.
                    "pixel_values": np.ones((2, 4, 96)),
                    "image_grid_thw": [[[1, 2, 2]], [[1, 2, 2]]],
                    "vision_indices": [[1], [2]],
                },
                [[5, 30, 21, 2, 4, 4, 4, 4], [30, 5, 2, 4, 4, 4, 4, 4]],
                # The image tokens are not labels.
                [[0, 1, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0]],
            ),
        )

    def test_video_preprocessor_basics(self):
        self.init_kwargs = {
            "tokenizer": self.video_preprocessor.tokenizer,
            "video_converter": self.video_converter,
            "sequence_length": 8,
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
            expected_output=(
                {
                    "token_ids": [
                        [1, 6, 30, 21, 2, 4, 4, 4],
                        [1, 30, 6, 2, 4, 4, 4, 4],
                    ],
                    "padding_mask": [
                        [1, 1, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                    ],
                    # Two 8x8 frames form one temporal patch group.
                    "pixel_values": np.ones((2, 4, 96)),
                    "image_grid_thw": [[[1, 2, 2]], [[1, 2, 2]]],
                    "vision_indices": [[1], [2]],
                },
                [[6, 30, 21, 2, 4, 4, 4, 4], [30, 6, 2, 4, 4, 4, 4, 4]],
                [[0, 1, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0]],
            ),
        )

    def test_call_with_several_images_per_prompt(self):
        x, _, _ = self.image_preprocessor(
            {
                "prompts": "<|image|><|image|> airplane",
                "images": np.stack([self.image, self.image]),
            }
        )
        self.assertEqual(tuple(x["token_ids"].shape), (8,))
        self.assertEqual(tuple(x["image_grid_thw"].shape), (2, 3))
        self.assertAllEqual(x["vision_indices"], [1, 2])

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
        ids = self.image_preprocessor._expand_vision_placeholders(
            [[self.image_token_id, 1000, 1001]], [3], []
        )
        self.assertEqual(ids, [[self.image_token_id] * 3 + [1000, 1001]])

    def test_expand_placeholders_across_prompts(self):
        # The second prompt uses the second image's token count.
        ids = self.image_preprocessor._expand_vision_placeholders(
            [[self.image_token_id, 1000], [self.image_token_id, 1001]],
            [1, 4],
            [],
        )
        self.assertEqual(
            ids,
            [[self.image_token_id, 1000], [self.image_token_id] * 4 + [1001]],
        )

    def test_extra_placeholder_raises(self):
        with self.assertRaisesRegex(ValueError, "more image placeholders"):
            self.image_preprocessor._expand_vision_placeholders(
                [[self.image_token_id, self.image_token_id]], [1], []
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
        # An 8x8 image gives one merged token after the start token.
        self.assertAllEqual(output["vision_indices"], [1])

    def test_images_with_different_sizes(self):
        wide_image = np.random.randint(0, 255, (8, 16, 3)).astype("float32")
        output = self.image_preprocessor.generate_preprocess(
            {
                "prompts": ["<|image|> airplane", "<|image|> airport"],
                "images": [self.image, wide_image],
            }
        )
        self.assertAllEqual(output["image_grid_thw"], [[1, 2, 2], [1, 2, 4]])
        # The second prompt expands its placeholder to two tokens.
        self.assertAllEqual(output["vision_indices"], [1, 9, 10])

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
            {"prompts": "<|video|> airplane", "videos": video}
        )
        self.assertEqual(tuple(output["token_ids"].shape), (8,))
        self.assertAllEqual(output["image_grid_thw"], [[1, 2, 2]])
        self.assertAllEqual(output["vision_indices"], [1])

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
            sequence_length=12,
        )
        self.assertAllEqual(output["image_grid_thw"], [[1, 2, 2], [2, 2, 4]])
        # The second video has 2 frame groups of 2 merged tokens each.
        self.assertAllEqual(output["vision_indices"], [1, 13, 14, 15, 16])

    def test_media_without_converter_raises(self):
        with self.assertRaisesRegex(ValueError, "no `video_converter`"):
            self.image_preprocessor.generate_preprocess(
                {
                    "prompts": ["<|image|> airplane"],
                    "videos": np.zeros((2, 8, 8, 3), dtype="float32"),
                }
            )
