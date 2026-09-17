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
        image = np.ones((4, 4, 3))
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
        images = [np.ones((4, 4, 3)), np.ones((8, 8, 3))]
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
        image_converter = self._image_converter()
        preprocessor = Qwen2VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=image_converter,
            sequence_length=32,
        )

        # 4 frames of 4x4 → grid [2, 2, 2] → 8 merged tokens.
        video = np.ones((4, 4, 4, 3))
        prompt = "<|vision_start|><|video_pad|><|vision_end|> air"

        out = preprocessor.generate_preprocess(
            {"prompts": prompt, "videos": [video]}
        )

        token_ids = ops.convert_to_numpy(out["token_ids"])
        video_id = self.tokenizer.video_token_id
        self.assertEqual(int(np.sum(token_ids == video_id)), 8)
        self.assertEqual(
            list(ops.convert_to_numpy(out["image_grid_thw"])[0]), [2, 2, 2]
        )

        # Temporal channel increments per frame (2 frames of 4 tokens).
        pos_ids = ops.convert_to_numpy(out["position_ids"])
        self.assertEqual(list(pos_ids[0, 1:9]), [1, 1, 1, 1, 2, 2, 2, 2])

    def test_missing_image_converter_error(self):
        preprocessor = Qwen2VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=16,
        )
        with self.assertRaises(ValueError):
            preprocessor.generate_preprocess(
                {
                    "prompts": "<|image_pad|> air",
                    "images": [np.ones((4, 4, 3))],
                }
            )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen2VLCausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=Qwen2VLCausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
