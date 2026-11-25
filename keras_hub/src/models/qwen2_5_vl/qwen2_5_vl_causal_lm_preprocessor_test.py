import numpy as np
from keras import ops

from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_causal_lm_preprocessor import (
    Qwen2_5VLCausalLMPreprocessor,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_image_converter import (
    Qwen2_5VLImageConverter,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_tokenizer import (
    Qwen2_5VLTokenizer,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_video_converter import (
    Qwen2_5VLVideoConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen2_5VLCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i"]
        self.merges += ["p l", "n e", "Ġa t", "p o", "r t", "Ġt h"]
        self.merges += ["ai r", "pl a", "po rt", "Ġai r", "Ġa i"]
        self.merges += ["pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
        self.vocab += ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
        self.vocab += ["<|video_pad|>", "!"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.tokenizer = Qwen2_5VLTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = ["airplane at airport"]

    def _image_converter(self):
        return Qwen2_5VLImageConverter(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=1,
            min_pixels=16,
            max_pixels=100,
        )

    def _video_converter(self):
        """Video converter with small bounds so test grids stay tiny."""
        return Qwen2_5VLVideoConverter(
            image_converter=self._image_converter(),
            min_pixels=16,
            max_pixels=100,
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=Qwen2_5VLCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[10, 22, 32, 31, 24, 2, 1, 1]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [[22, 32, 31, 24, 2, 1, 1, 1]],
                [[1, 1, 1, 1, 1, 0, 0, 0]],
            ),
        )

    def test_with_end_token(self):
        input_data = ["airplane at airport"] * 4

        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            **self.init_kwargs,
            add_end_token=True,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[10, 22, 32, 31, 24, 2, 1, 1]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 0, 0]] * 4)
        self.assertAllEqual(y, [[22, 32, 31, 24, 2, 1, 1, 1]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "airplane at airport"
        preprocessor = Qwen2_5VLCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [10, 22, 32, 31, 24, 1, 1, 1])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [10, 22, 32, 31, 24, 1, 1, 1],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = Qwen2_5VLCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "airplane at airport")

    def test_generate_preprocess_with_image(self):
        """Image input: marker expansion, vision_indices, position_ids."""
        image_converter = self._image_converter()
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=image_converter,
            sequence_length=16,
        )

        # 4x4 image → grid [1, 2, 2] → 4 merged tokens (merge_size=1).
        image = np.ones((4, 4, 3), dtype="float32")
        prompt = "<|vision_start|><|image_pad|><|vision_end|> air"

        out = preprocessor.generate_preprocess(
            {"prompts": prompt, "images": [image]}
        )

        token_ids = ops.convert_to_numpy(out["token_ids"])
        image_id = self.tokenizer.image_token_id
        # <|vision_start|>, 4 image pads, <|vision_end|>, then text.
        self.assertEqual(list(token_ids[1:5]), [image_id] * 4)

        self.assertEqual(tuple(out["image_grid_thw"].shape), (1, 3))
        self.assertEqual(
            list(ops.convert_to_numpy(out["image_grid_thw"])[0]), [1, 2, 2]
        )

        # 4-D pixel patches: (4, tps, p, p, C).
        self.assertEqual(tuple(out["pixel_values"].shape), (4, 2, 2, 2, 3))

        # vision_indices mark the 4 image token positions.
        self.assertEqual(
            list(ops.convert_to_numpy(out["vision_indices"])), [1, 2, 3, 4]
        )

        # position_ids: (3, seq_len) unbatched — [t, h, w] channels.
        pos_ids = ops.convert_to_numpy(out["position_ids"])
        self.assertEqual(pos_ids.shape, (3, 16))
        # First token (vision_start): text position 0.
        self.assertEqual(list(pos_ids[:, 0]), [0, 0, 0])
        # Image tokens: h in {0,1}, w in {0,1}, t=0 offset by pos 1.
        self.assertEqual(list(pos_ids[0, 1:5]), [1, 1, 1, 1])
        self.assertEqual(list(pos_ids[1, 1:5]), [1, 1, 2, 2])
        self.assertEqual(list(pos_ids[2, 1:5]), [1, 2, 1, 2])
        # After the vision span, text continues from max(h_m, w_m)=2.
        # vision_end at index 5 → pos 1+2=3.
        self.assertEqual(list(pos_ids[:, 5]), [3, 3, 3])

    def _structured_image(self):
        """Structured non-uniform image (smooth gradient + dark edge), so
        the bicubic+antialias kernel exercises non-trivial weights."""
        xx, yy = np.meshgrid(np.linspace(0, 255, 9), np.linspace(0, 255, 12))
        image = np.stack([xx + yy * 0.3, yy, 255 - xx], -1)
        image[:, :3] *= 0.4
        return np.clip(image, 0, 255)

    def _pixel_values(self, image):
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self._image_converter(),
            sequence_length=32,
        )
        out = preprocessor.generate_preprocess(
            {
                "prompts": "<|vision_start|><|image_pad|><|vision_end|> air",
                "images": [image],
            }
        )
        return ops.convert_to_numpy(out["pixel_values"])

    def test_generate_preprocess_pixel_values_match_hf(self):
        """Pin `pixel_values` against HF-equivalent preprocessing output.

        Expected values were computed offline with HF's
        `Qwen2VLImageProcessorFast` (torchvision bicubic+antialias resize,
        1/255 rescale, CLIP normalization).
        """
        image = self._structured_image().astype("float32")
        # First patch of the resized grid (top-left 2x2), first temporal
        # frame — HF-equivalent values, atol 1e-3.
        patch0 = self._pixel_values(image)[0, 0]
        expected = np.array(
            [
                [
                    [-1.779326, -1.752097, -0.042374],
                    [-1.608532, -1.752097, -0.297906],
                ],
                [
                    [-1.738716, -1.612934, -0.042374],
                    [-1.570418, -1.621489, -0.297906],
                ],
            ],
            dtype="float32",
        )
        self.assertAllClose(patch0, expected, atol=1e-3)

    def test_generate_preprocess_pixel_values_uint8_match_hf(self):
        """uint8 inputs take torchvision's quantized integer resize path.

        Expected values computed offline with HF's
        `Qwen2VLImageProcessorFast` on the uint8 array — HF quantizes
        only uint8 inputs, and its uint8/float32 paths genuinely diverge
        (~0.15 on this fixture), so the dtype dispatch is load-bearing.
        """
        image = self._structured_image().astype("uint8")
        patch0 = self._pixel_values(image)[0, 0]
        expected = np.array(
            [
                [
                    [-1.7776642, -1.752097, -0.04399312],
                    [-1.6170814, -1.752097, -0.29995432],
                ],
                [
                    [-1.7484673, -1.6170272, -0.04399312],
                    [-1.5732862, -1.6320349, -0.29995432],
                ],
            ],
            dtype="float32",
        )
        self.assertAllClose(patch0, expected, atol=1e-5)

    def test_generate_preprocess_pixel_values_int_float32_match_hf(self):
        """Integer-valued float32 inputs take the floating-point path.

        HF quantizes only uint8 dtype inputs, so the same pixel values as
        float32 must NOT go through the integer-quantized resize.
        Expected values computed offline with HF's
        `Qwen2VLImageProcessorFast` on `img_u8.astype("float32")`.
        """
        image = self._structured_image().astype("uint8").astype("float32")
        patch0 = self._pixel_values(image)[0, 0]
        expected = np.array(
            [
                [
                    [-1.7804581, -1.752097, -0.04262114],
                    [-1.6189507, -1.752097, -0.30228978],
                ],
                [
                    [-1.7498758, -1.6170272, -0.04262114],
                    [-1.5772278, -1.6256379, -0.30228978],
                ],
            ],
            dtype="float32",
        )
        self.assertAllClose(patch0, expected, atol=1e-5)

    def test_generate_preprocess_with_multiple_images(self):
        """Two image markers each expand independently."""
        image_converter = self._image_converter()
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=image_converter,
            sequence_length=32,
        )

        # Image 1: 4x4 → grid [1,2,2] → 4 tokens.
        # Image 2: 8x8 → grid [1,4,4] → 16 tokens.
        images = [
            np.ones((4, 4, 3), dtype="float32"),
            np.ones((8, 8, 3), dtype="float32"),
        ]
        prompt = "<|image_pad|> air <|image_pad|>"

        out = preprocessor.generate_preprocess(
            {"prompts": prompt, "images": images}
        )

        token_ids = ops.convert_to_numpy(out["token_ids"])
        image_id = self.tokenizer.image_token_id
        self.assertEqual(int(np.sum(token_ids == image_id)), 4 + 16)
        self.assertEqual(tuple(out["vision_indices"].shape), (4 + 16,))
        self.assertEqual(tuple(out["image_grid_thw"].shape), (2, 3))

    def test_generate_preprocess_with_video(self):
        """Video input: temporal grid produces t*h*w tokens; videos work
        with `image_converter=None` (merge size comes from the video
        converter's inner image converter)."""
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=self._video_converter(),
            sequence_length=32,
        )

        video = np.ones((8, 2, 2, 3), dtype="float32")
        tok = self.tokenizer
        prompt = (
            tok.vision_start_token
            + tok.video_token
            + tok.vision_end_token
            + " air"
        )

        out = preprocessor.generate_preprocess(
            {"prompts": prompt, "videos": [video]}
        )

        token_ids = ops.convert_to_numpy(out["token_ids"])
        video_id = self.tokenizer.video_token_id
        self.assertEqual(int(np.sum(token_ids == video_id)), 16)
        self.assertEqual(
            list(ops.convert_to_numpy(out["image_grid_thw"])[0]), [4, 2, 2]
        )

        # Temporal channel steps by `tokens_per_second` per merged frame
        # (4 tokens each) — HF's default when no fps metadata is given.
        pos_ids = ops.convert_to_numpy(out["position_ids"])
        self.assertEqual(
            list(pos_ids[0, 1:17]),
            [1] * 4 + [5] * 4 + [9] * 4 + [13] * 4,
        )
        self.assertEqual(list(pos_ids[:, 17]), [3, 3, 3])
        self.assertEqual(list(pos_ids[:, 18]), [4, 4, 4])

    def test_generate_preprocess_video_decode_position_base(self):
        """Pad/decode positions rebase to `llm_positions.max() + 1`.

        HF's `mrope_position_deltas` bases generated-token positions on
        the max position across all three channels — including temporal.
        With `time_interval` scaling, the temporal max (13) exceeds the
        spatial continuation (5), so pad positions must start at 14.
        """
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=self._video_converter(),
            sequence_length=32,
        )

        video = np.ones((8, 2, 2, 3), dtype="float32")
        tok = self.tokenizer
        prompt = (
            tok.vision_start_token
            + tok.video_token
            + tok.vision_end_token
            + " air"
        )

        out = preprocessor.generate_preprocess(
            {"prompts": prompt, "videos": [video]}
        )
        pos_ids = ops.convert_to_numpy(out["position_ids"])
        # Real tokens keep the spatial continuation (positions 17, 18).
        self.assertEqual(list(pos_ids[:, 17]), [3, 3, 3])
        self.assertEqual(list(pos_ids[:, 18]), [4, 4, 4])
        # Pad region starts at llm max + 1 = 14, then runs sequentially.
        self.assertEqual(list(pos_ids[:, 19]), [14, 14, 14])
        self.assertEqual(list(pos_ids[:, 31]), [26, 26, 26])

    def test_generate_preprocess_second_per_grid_ts(self):
        """`second_per_grid_ts` scales video temporal positions.

        interval = tokens_per_second * sgt; fractional intervals truncate
        per frame (HF `(arange(t) * interval).long()`).
        """
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=self._video_converter(),
            sequence_length=32,
        )

        video = np.ones((8, 2, 2, 3), dtype="float32")
        tok = self.tokenizer
        prompt = (
            tok.vision_start_token
            + tok.video_token
            + tok.vision_end_token
            + " air"
        )

        # sgt = 0.5 → interval = 4 * 0.5 = 2: positions step by 2.
        out = preprocessor.generate_preprocess(
            {
                "prompts": prompt,
                "videos": [video],
                "second_per_grid_ts": [0.5],
            }
        )
        pos_ids = ops.convert_to_numpy(out["position_ids"])
        self.assertEqual(
            list(pos_ids[0, 1:17]),
            [1] * 4 + [3] * 4 + [5] * 4 + [7] * 4,
        )

        # sgt = 2/3 → interval = 8/3 ≈ 2.67: per-frame truncation gives
        # steps 0, 2, 5, 8.
        out = preprocessor.generate_preprocess(
            {
                "prompts": prompt,
                "videos": [video],
                "second_per_grid_ts": [2.0 / 3.0],
            }
        )
        pos_ids = ops.convert_to_numpy(out["position_ids"])
        self.assertEqual(
            list(pos_ids[0, 1:17]),
            [1] * 4 + [3] * 4 + [6] * 4 + [9] * 4,
        )

        # Custom tokens_per_second scales the interval.
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=self._video_converter(),
            tokens_per_second=2,
            sequence_length=32,
        )
        out = preprocessor.generate_preprocess(
            {
                "prompts": prompt,
                "videos": [video],
                "second_per_grid_ts": [1.0],
            }
        )
        pos_ids = ops.convert_to_numpy(out["position_ids"])
        self.assertEqual(
            list(pos_ids[0, 1:17]),
            [1] * 4 + [3] * 4 + [5] * 4 + [7] * 4,
        )

    def test_second_per_grid_ts_multi_video(self):
        """`second_per_grid_ts` applies per video in prompt order, on
        both the training (`__call__`) and generate paths."""
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=self._video_converter(),
            sequence_length=64,
        )
        video = np.ones((8, 2, 2, 3), dtype="float32")
        tok = self.tokenizer
        prompt = (
            tok.vision_start_token
            + tok.video_token
            + tok.vision_end_token
            + " air"
            + tok.vision_start_token
            + tok.video_token
            + tok.vision_end_token
        )
        sgt = [0.5, 2.0]
        out = preprocessor.generate_preprocess(
            {
                "prompts": prompt,
                "videos": [video, video],
                "second_per_grid_ts": sgt,
            }
        )
        pos_ids = ops.convert_to_numpy(out["position_ids"])
        # Video 1 (interval = 4*0.5 = 2): pads at 1..16, t steps by 2.
        self.assertEqual(
            list(pos_ids[0, 1:17]), [1] * 4 + [3] * 4 + [5] * 4 + [7] * 4
        )
        # Interleaved text/vision_start continue from the spatial max.
        self.assertEqual(list(pos_ids[:, 17]), [3, 3, 3])
        self.assertEqual(list(pos_ids[:, 18]), [4, 4, 4])
        self.assertEqual(list(pos_ids[:, 19]), [5, 5, 5])
        # Video 2 (interval = 4*2 = 8): pads at 20..35, base offset 6.
        self.assertEqual(
            list(pos_ids[0, 20:36]),
            [6] * 4 + [14] * 4 + [22] * 4 + [30] * 4,
        )
        # Trailing vision_end keeps the spatial continuation; pad
        # positions rebase to llm max + 1 = 31.
        self.assertEqual(list(pos_ids[:, 36]), [8, 8, 8])
        self.assertEqual(list(pos_ids[:, 37]), [31, 31, 31])

        # float64 arrays are accepted and produce identical positions.
        out64 = preprocessor.generate_preprocess(
            {
                "prompts": prompt,
                "videos": [video, video],
                "second_per_grid_ts": np.asarray(sgt, dtype=np.float64),
            }
        )
        self.assertAllEqual(out64["position_ids"], out["position_ids"])

        # The training path (`__call__`) computes the same positions for
        # the real tokens. It additionally appends the end token at
        # index 37, which takes the spatial continuation before the
        # llm-max pad rebase.
        x, _, _ = preprocessor(
            {
                "prompts": [prompt],
                "videos": [video, video],
                "second_per_grid_ts": sgt,
            }
        )
        train_pos = ops.convert_to_numpy(x["position_ids"])[0]
        self.assertAllEqual(train_pos[:, :37], pos_ids[:, :37])
        self.assertAllEqual(train_pos[:, 37], [9, 9, 9])
        self.assertAllEqual(train_pos[:, 38], [31, 31, 31])

    def test_second_per_grid_ts_count_mismatch(self):
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=self._video_converter(),
            sequence_length=64,
        )
        video = np.ones((8, 2, 2, 3), dtype="float32")
        tok = self.tokenizer
        prompt = (
            tok.vision_start_token
            + tok.video_token
            + tok.vision_end_token
            + " air"
            + tok.vision_start_token
            + tok.video_token
            + tok.vision_end_token
        )
        with self.assertRaisesRegex(ValueError, "second_per_grid_ts"):
            preprocessor.generate_preprocess(
                {
                    "prompts": prompt,
                    "videos": [video, video],
                    "second_per_grid_ts": [0.5],
                }
            )

    def test_batched_call_with_images(self):
        """Two prompts with two images: each row expands its own span."""
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self._image_converter(),
            sequence_length=32,
        )
        images = [
            np.ones((4, 4, 3), dtype="float32"),
            np.ones((8, 8, 3), dtype="float32"),
        ]
        pad = "<|image_pad|>"
        prompts = [pad + " air", pad + " airport"]
        x, y, sw = preprocessor({"prompts": prompts, "images": images})

        token_ids = ops.convert_to_numpy(x["token_ids"])
        image_id = self.tokenizer.image_token_id
        # img1 4x4 → grid [1,2,2] → 4 tokens; img2 8x8 → [1,4,4] → 16.
        self.assertEqual(int(np.sum(token_ids[0] == image_id)), 4)
        self.assertEqual(int(np.sum(token_ids[1] == image_id)), 16)
        self.assertEqual(tuple(x["image_grid_thw"].shape), (2, 3))
        # vision_indices covers both rows' image token positions (4+16).
        self.assertEqual(tuple(x["vision_indices"].shape), (4 + 16,))

    def test_prompt_image_count_mismatch(self):
        """Placeholder/media count mismatch raises a descriptive error."""
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self._image_converter(),
            sequence_length=64,
        )
        image = np.ones((4, 4, 3), dtype="float32")
        pad = "<|image_pad|>"
        # More images than placeholders.
        with self.assertRaisesRegex(ValueError, "Found 1.*but 2 image"):
            preprocessor({"prompts": [pad + " air"], "images": [image, image]})
        # More placeholders than images.
        with self.assertRaisesRegex(ValueError, "Found 2.*but 1 image"):
            preprocessor({"prompts": [pad + " air " + pad], "images": [image]})
        # Placeholder with no images at all.
        with self.assertRaisesRegex(ValueError, "Found 1.*but 0 image"):
            preprocessor({"prompts": [pad + " air"], "images": []})

    def test_video_falls_back_to_image_converter(self):
        """`video_converter=None` stays None: videos defer to the shared
        `image_converter` (HF's image-processor fallback for videos)."""
        image_converter = self._image_converter()
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=image_converter,
            sequence_length=32,
        )
        self.assertIsNone(preprocessor.video_converter)

        video = np.ones((8, 2, 2, 3), dtype="float32")
        tok = self.tokenizer
        prompt = (
            tok.vision_start_token
            + tok.video_token
            + tok.vision_end_token
            + " air"
        )
        out = preprocessor.generate_preprocess(
            {"prompts": prompt, "videos": [video]}
        )
        self.assertEqual(
            list(ops.convert_to_numpy(out["image_grid_thw"])[0]), [4, 2, 2]
        )

    def test_video_converter_identity_after_from_config(self):
        """A serialized explicit video converter shares the deserialized
        `image_converter` — bounds mutations propagate to videos."""
        image_converter = self._image_converter()
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=image_converter,
            # `None` bounds delegate to the shared image converter, so a
            # mutation on it propagates to video processing.
            video_converter=Qwen2_5VLVideoConverter(
                image_converter=image_converter,
            ),
            sequence_length=32,
        )
        revived = Qwen2_5VLCausalLMPreprocessor.from_config(
            preprocessor.get_config()
        )
        self.assertIsNotNone(revived.video_converter)
        self.assertIs(
            revived.video_converter.image_converter,
            revived.image_converter,
        )
        revived.image_converter.max_pixels = 42
        self.assertEqual(revived.video_converter.max_pixels, 42)

    def test_serialization(self):
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self._image_converter(),
            tokens_per_second=2,
            sequence_length=32,
        )
        self.run_serialization_test(preprocessor)
        revived = Qwen2_5VLCausalLMPreprocessor.from_config(
            preprocessor.get_config()
        )
        self.assertEqual(revived.tokens_per_second, 2)

    def test_missing_image_converter_error(self):
        preprocessor = Qwen2_5VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=16,
        )
        with self.assertRaises(ValueError):
            preprocessor.generate_preprocess(
                {
                    "prompts": "<|image_pad|> air",
                    "images": [np.ones((4, 4, 3), dtype="float32")],
                }
            )

    def test_tf_data_map_text_only_dict(self):
        """Media-free dict inputs trace through `tf.data.Dataset.map`."""
        import tensorflow as tf

        preprocessor = Qwen2_5VLCausalLMPreprocessor(**self.init_kwargs)
        ds = tf.data.Dataset.from_tensor_slices(
            {"prompts": ["air", "airport"]}
        ).batch(2)
        ds = ds.map(preprocessor)
        for x, y, sw in ds:
            self.assertEqual(tuple(x["token_ids"].shape), (2, 8))
            self.assertEqual(tuple(x["padding_mask"].shape), (2, 8))
