import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen2_vl.qwen2_vl_causal_lm_preprocessor import (
    Qwen2VLCausalLMPreprocessor,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (
    Qwen2VLImageConverter,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_tokenizer import Qwen2VLTokenizer
from keras_hub.src.models.qwen2_vl.qwen2_vl_video_converter import (
    Qwen2VLVideoConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen2VLCausalLMPreprocessorTest(TestCase):
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
        self.tokenizer = Qwen2VLTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = ["airplane at airport"]

    def _image_converter(self):
        return Qwen2VLImageConverter(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=1,
            min_pixels=16,
            max_pixels=100,
        )

    def _video_converter(self):
        """Video converter with small bounds so test grids stay tiny."""
        return Qwen2VLVideoConverter(
            image_converter=self._image_converter(),
            min_pixels=16,
            max_pixels=100,
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=Qwen2VLCausalLMPreprocessor,
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

        preprocessor = Qwen2VLCausalLMPreprocessor(
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
        preprocessor = Qwen2VLCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [10, 22, 32, 31, 24, 1, 1, 1])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [10, 22, 32, 31, 24, 1, 1, 1],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = Qwen2VLCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "airplane at airport")

    def test_generate_preprocess_with_image(self):
        """Image input: marker expansion, vision_indices, position_ids."""
        image_converter = self._image_converter()
        preprocessor = Qwen2VLCausalLMPreprocessor(
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

    def test_generate_preprocess_pixel_values_match_hf(self):
        """Pin `pixel_values` against HF-equivalent preprocessing output.

        Expected values were computed offline with
        `torch.nn.functional.interpolate(mode="bicubic", antialias=True)` —
        the op HF's Qwen2VLImageProcessor resizes through — followed by
        1/255 rescale and CLIP normalization.
        """
        preprocessor = Qwen2VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self._image_converter(),
            sequence_length=32,
        )

        # Structured non-uniform image (smooth gradient + dark edge), so
        # the bicubic+antialias kernel exercises non-trivial weights.
        xx, yy = np.meshgrid(np.linspace(0, 255, 9), np.linspace(0, 255, 12))
        image = np.stack([xx + yy * 0.3, yy, 255 - xx], -1).astype("float32")
        image[:, :3] *= 0.4
        image = np.clip(image, 0, 255)

        out = preprocessor.generate_preprocess(
            {
                "prompts": "<|vision_start|><|image_pad|><|vision_end|> air",
                "images": [image],
            }
        )

        # First patch of the resized grid (top-left 2x2), first temporal
        # frame — HF-equivalent values, atol 1e-3.
        patch0 = ops.convert_to_numpy(out["pixel_values"])[0, 0]
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

    def test_generate_preprocess_with_multiple_images(self):
        """Two image markers each expand independently."""
        image_converter = self._image_converter()
        preprocessor = Qwen2VLCausalLMPreprocessor(
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
        """Video input: temporal grid produces t*h*w tokens."""
        preprocessor = Qwen2VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=self._video_converter(),
            sequence_length=32,
        )

        video = np.ones((8, 2, 2, 3), dtype="float32")
        prompt = "<|vision_start|><|video_pad|><|vision_end|> air"

        out = preprocessor.generate_preprocess(
            {"prompts": prompt, "videos": [video]}
        )

        token_ids = ops.convert_to_numpy(out["token_ids"])
        video_id = self.tokenizer.video_token_id
        self.assertEqual(int(np.sum(token_ids == video_id)), 16)
        self.assertEqual(
            list(ops.convert_to_numpy(out["image_grid_thw"])[0]), [4, 2, 2]
        )

        # Temporal channel increments per merged frame (4 tokens each).
        pos_ids = ops.convert_to_numpy(out["position_ids"])
        self.assertEqual(
            list(pos_ids[0, 1:17]),
            [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
        )
        self.assertEqual(list(pos_ids[:, 17]), [3, 3, 3])
        self.assertEqual(list(pos_ids[:, 18]), [4, 4, 4])

    def test_generate_preprocess_with_video_converter_only(self):
        """Videos work with `image_converter=None`; merge size comes from
        the video converter's inner image converter."""
        video_converter = self._video_converter()
        preprocessor = Qwen2VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=video_converter,
            sequence_length=32,
        )

        video = np.ones((8, 2, 2, 3), dtype="float32")
        prompt = "<|vision_start|><|video_pad|><|vision_end|> air"

        out = preprocessor.generate_preprocess(
            {"prompts": prompt, "videos": [video]}
        )

        token_ids = ops.convert_to_numpy(out["token_ids"])
        video_id = self.tokenizer.video_token_id
        self.assertEqual(int(np.sum(token_ids == video_id)), 16)
        self.assertEqual(
            list(ops.convert_to_numpy(out["image_grid_thw"])[0]), [4, 2, 2]
        )

        pos_ids = ops.convert_to_numpy(out["position_ids"])
        self.assertEqual(
            list(pos_ids[0, 1:17]),
            [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
        )

    def test_generate_preprocess_video_decode_position_base(self):
        """Pad/decode positions rebase to `llm_positions.max() + 1`.

        HF's `mrope_position_deltas` bases generated-token positions on
        the max position across all three channels — including temporal.
        A video with more merged frames than the spatial continuation
        leaves a temporal max above the last text position, so pad
        positions must start at llm max + 1.
        """
        preprocessor = Qwen2VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=self._video_converter(),
            sequence_length=32,
        )

        # 10 frames → grid [5, 2, 2] → 20 video tokens, t positions 1-5.
        video = np.ones((10, 2, 2, 3), dtype="float32")
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
        # Real tokens keep the spatial continuation (positions 21, 22).
        self.assertEqual(list(pos_ids[:, 21]), [3, 3, 3])
        self.assertEqual(list(pos_ids[:, 22]), [4, 4, 4])
        # Pad region starts at llm max + 1 = 6, then runs sequentially.
        self.assertEqual(list(pos_ids[:, 23]), [6, 6, 6])
        self.assertEqual(list(pos_ids[:, 31]), [14, 14, 14])

    def test_batched_call_with_images(self):
        """Two prompts with two images: each row expands its own span."""
        preprocessor = Qwen2VLCausalLMPreprocessor(
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
        preprocessor = Qwen2VLCausalLMPreprocessor(
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

    def test_video_converter_autowrap(self):
        """An image-converter-only preprocessor auto-wraps a video
        converter so videos get video bounds, not image bounds."""
        image_converter = self._image_converter()
        preprocessor = Qwen2VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=image_converter,
            sequence_length=32,
        )
        video_converter = preprocessor.video_converter
        self.assertIsNotNone(video_converter)
        self.assertIsInstance(video_converter, Qwen2VLVideoConverter)
        self.assertIs(video_converter.image_converter, image_converter)
        # Per-frame video bounds inherit the image converter's bounds,
        # matching HF's `from_pretrained` image-config fallback.
        self.assertEqual(video_converter.min_pixels, image_converter.min_pixels)
        self.assertEqual(video_converter.max_pixels, image_converter.max_pixels)

    def test_missing_image_converter_error(self):
        preprocessor = Qwen2VLCausalLMPreprocessor(
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

        preprocessor = Qwen2VLCausalLMPreprocessor(**self.init_kwargs)
        ds = tf.data.Dataset.from_tensor_slices(
            {"prompts": ["air", "airport"]}
        ).batch(2)
        ds = ds.map(preprocessor)
        for x, y, sw in ds:
            self.assertEqual(tuple(x["token_ids"].shape), (2, 8))
            self.assertEqual(tuple(x["padding_mask"].shape), (2, 8))

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen2VLCausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=Qwen2VLCausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
