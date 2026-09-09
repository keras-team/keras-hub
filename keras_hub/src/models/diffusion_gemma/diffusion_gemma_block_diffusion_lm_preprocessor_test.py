import numpy as np
import pytest

from keras_hub.src.models.diffusion_gemma.diffusion_gemma_block_diffusion_lm_preprocessor import (  # noqa: E501
    DiffusionGemmaBlockDiffusionLMPreprocessor,
)
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4ImageConverter,
)
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class DiffusionGemmaBlockDiffusionLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = MockGemma4Tokenizer()

        # Text-only preprocessor (no media converters).
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
            "canvas_length": 4,
        }
        self.preprocessor = DiffusionGemmaBlockDiffusionLMPreprocessor(
            **self.init_kwargs
        )

        # Vision-enabled preprocessor (image_converter set).
        self.image_converter = Gemma4ImageConverter(
            image_size=(16, 16),
            patch_size=4,
        )
        self.vision_init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 24,
            "canvas_length": 4,
            "max_images_per_prompt": 2,
            "num_vision_tokens_per_image": 4,
        }
        self.vision_preprocessor = DiffusionGemmaBlockDiffusionLMPreprocessor(
            **self.vision_init_kwargs
        )

    def test_preprocessor_basics(self):
        # Vocab: 1=<bos>, 9=the, 14=quick, 10=brown, 12=fox, 2=<eos>, 0=<pad>
        expected_x = {
            "token_ids": [[0, 0, 0, 1, 9, 14, 10, 12]] * 2,
            "padding_mask": [[0, 0, 0, 1, 1, 1, 1, 1]] * 2,
            "position_ids": [[0, 1, 2, 3, 4, 5, 6, 7]] * 2,
        }
        # y is token_ids shifted left by one; sw is 1 for non-pad labels.
        expected_y = [[0, 0, 1, 9, 14, 10, 12, 2]] * 2
        expected_sw = [[0, 0, 1, 1, 1, 1, 1, 1]] * 2
        self.run_preprocessor_test(
            cls=DiffusionGemmaBlockDiffusionLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=["the quick brown fox", "the quick brown fox"],
            expected_output=(expected_x, expected_y, expected_sw),
        )

    def test_no_start_end_token(self):
        preprocessor = DiffusionGemmaBlockDiffusionLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        input_data = ["the quick brown fox"] * 2
        x, y, sw = preprocessor(input_data)
        # Without BOS the first non-padding token should be "the" (id=9).
        self.assertAllEqual(x["token_ids"][0, -3], 9)

    def test_generate_preprocess_prompt_only(self):
        # token_ids should contain only the packed prompt, not the canvas.
        output = self.preprocessor.generate_preprocess("the quick brown fox")
        self.assertAllEqual(output["token_ids"], [0, 0, 0, 1, 9, 14, 10, 12])
        self.assertAllEqual(output["padding_mask"], [0, 0, 0, 1, 1, 1, 1, 1])

    def test_generate_preprocess_batched(self):
        output = self.preprocessor.generate_preprocess(
            {"prompts": ["the", "the quick brown fox"]}
        )
        self.assertAllEqual(
            output["token_ids"],
            [[0, 0, 0, 0, 0, 0, 1, 9], [0, 0, 0, 1, 9, 14, 10, 12]],
        )
        self.assertAllEqual(
            output["padding_mask"],
            [[0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]],
        )

    def test_generate_postprocess(self):
        # canvas_length=4; vocab: 9=the, 14=quick, 10=brown, 12=fox
        canvas = np.array([9, 14, 10, 12], dtype="int32")
        result = self.preprocessor.generate_postprocess(canvas)
        self.assertAllEqual(result, "the quick brown fox")

    def test_generate_postprocess_batched(self):
        # canvas_length=4; each row is one generated canvas.
        canvas = np.array(
            [[9, 14, 10, 12], [9, 14, 10, 12]],
            dtype="int32",
        )
        results = self.preprocessor.generate_postprocess(canvas)
        self.assertEqual(len(results), 2)

    def test_vision_call_shape_and_indices(self):
        # One image per sample: image expands to 4 placeholder tokens
        # (num_vision_tokens_per_image=4), bracketed by start/end tokens.
        pixel_values = np.ones([2, 1, 16, 3 * 4 * 4], dtype="float32")
        pixel_position_ids = np.ones([2, 1, 16, 2], dtype="int32")
        x, y, sw = self.vision_preprocessor(
            {
                "prompts": [
                    "the <|image|> fox",
                    "the <|image|> fox",
                ],
                "responses": ["round", "round"],
                "pixel_values": pixel_values,
                "pixel_position_ids": pixel_position_ids,
            }
        )
        self.assertEqual(x["token_ids"].shape[-1], 24)
        self.assertEqual(x["padding_mask"].shape[-1], 24)
        self.assertEqual(x["position_ids"].shape[-1], 24)
        # pixel_values / pixel_position_ids pass through unchanged.
        self.assertAllEqual(x["pixel_values"].shape, [2, 1, 16, 48])
        self.assertAllEqual(x["pixel_position_ids"].shape, [2, 1, 16, 2])
        # vision_indices is padded to max_images_per_prompt (2) *
        # num_vision_tokens_per_image (4) = 8, regardless of how many
        # image tokens this particular sample actually has.
        self.assertEqual(x["vision_indices"].shape[-1], 8)
        self.assertNotIn("vision_mask", x)

    def test_vision_raw_images_input(self):
        # Passing raw `images` (rather than precomputed pixel_values)
        # exercises `_preprocess_images`, which patchifies via the
        # image_converter instead of taking already-patchified input.
        images = np.ones((2, 16, 16, 3), dtype="float32")
        x, y, sw = self.vision_preprocessor(
            {
                "prompts": [
                    "the <|image|> fox",
                    "the <|image|> fox",
                ],
                "responses": ["round", "round"],
                "images": images,
            }
        )
        # Actual patch count depends on Gemma4ImageConverter's aspect-ratio
        # resizing policy, not simply image_size / patch_size, so only rank
        # and patch_dim (= patch_size**2 * 3 = 48) are checked here.
        self.assertEqual(len(x["pixel_values"].shape), 4)
        self.assertEqual(x["pixel_values"].shape[-1], 48)
        self.assertEqual(x["vision_indices"].shape[-1], 8)

    def test_vision_text_only_prompt_dummy_pixel_values(self):
        # A vision-enabled preprocessor with no image in the prompt should
        # still produce well-formed (empty) pixel_values, not None/error.
        x, y, sw = self.vision_preprocessor(
            {
                "prompts": ["the quick brown fox"],
                "responses": ["round"],
            }
        )
        self.assertEqual(x["pixel_values"].shape[0], 1)
        self.assertEqual(x["pixel_values"].shape[1], 0)
        self.assertEqual(x["vision_indices"].shape[0], 1)

    def test_vision_generate_preprocess(self):
        pixel_values = np.ones([1, 16, 3 * 4 * 4], dtype="float32")
        pixel_position_ids = np.ones([1, 16, 2], dtype="int32")
        output = self.vision_preprocessor.generate_preprocess(
            {
                "prompts": "the <|image|> fox",
                "pixel_values": pixel_values,
                "pixel_position_ids": pixel_position_ids,
            }
        )
        self.assertEqual(output["token_ids"].shape[-1], 24)
        # Unbatched input: the leading batch dim is squeezed back out.
        self.assertAllEqual(output["pixel_values"].shape, [1, 16, 48])
        self.assertEqual(output["vision_indices"].shape[-1], 8)

    def test_vision_serialization(self):
        self.run_serialization_test(self.vision_preprocessor)

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        input_data = {
            "prompts": ["the quick brown fox"],
            "responses": ["round"],
        }
        for preset in DiffusionGemmaBlockDiffusionLMPreprocessor.presets:
            self.run_preset_test(
                cls=DiffusionGemmaBlockDiffusionLMPreprocessor,
                preset=preset,
                input_data=input_data,
            )
