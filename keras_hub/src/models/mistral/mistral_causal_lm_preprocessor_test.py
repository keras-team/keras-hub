import os

import numpy as np
import pytest

from keras_hub.src.models.mistral.mistral_causal_lm_preprocessor import (
    MistralCausalLMPreprocessor,
)
from keras_hub.src.models.mistral.mistral_image_converter import (
    Mistral3ImageConverter,
)
from keras_hub.src.models.mistral.mistral_tokenizer import MistralTokenizer
from keras_hub.src.tests.test_case import TestCase


class MistralCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = MistralTokenizer(
            # Generated using create_mistral_test_proto.py
            proto=os.path.join(
                self.get_test_data_dir(), "mistral_test_vocab.spm"
            )
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (["the quick brown fox"],)

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=MistralCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 3, 8, 4, 6, 2, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [[3, 8, 4, 6, 2, 0, 0, 0]],  # Pass through labels.
                [[1, 1, 1, 1, 1, 0, 0, 0]],  # Pass through sample_weights.
            ),
        )

    def test_no_start_end_token(self):
        input_data = ["the quick brown fox"] * 4

        preprocessor = MistralCausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[3, 8, 4, 6, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(y, [[8, 4, 6, 0, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 0, 0, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "the quick brown fox"
        preprocessor = MistralCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 3, 8, 4, 6, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_preprocess_dict_input(self):
        # A text-only preprocessor also accepts the multimodal-style dict
        # form, so callers don't need to special-case text-only presets.
        preprocessor = MistralCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess({"prompts": "the quick brown fox"})
        self.assertAllEqual(x["token_ids"], [1, 3, 8, 4, 6, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_preprocess_dict_input_with_images_raises(self):
        # `@preprocessing_function` eagerly converts every dict value to a
        # tensor, so the placeholder image must be an actual array (`None`
        # can't be staged by every backend, e.g. JAX) even though the
        # `ValueError` fires before its content is ever used.
        preprocessor = MistralCausalLMPreprocessor(**self.init_kwargs)
        image = np.zeros((4, 4, 3), dtype="float32")
        with self.assertRaises(ValueError):
            preprocessor.generate_preprocess(
                {"prompts": "the quick brown fox", "images": [image]}
            )

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 3, 8, 4, 6, 0, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = MistralCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "the quick brown fox")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MistralCausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=MistralCausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )

    # === Multimodal tests ===

    def _multimodal_preprocessor(self, **kwargs):
        kwargs.setdefault(
            "tokenizer",
            MistralTokenizer(
                # Generated using create_mistral_vision_test_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(),
                    "mistral_vision_test_vocab.spm",
                ),
                has_vision_tokens=True,
            ),
        )
        kwargs.setdefault(
            "image_converter",
            Mistral3ImageConverter(
                longest_edge=16, patch_size=4, spatial_merge_size=1
            ),
        )
        kwargs.setdefault("sequence_length", 32)
        kwargs.setdefault("spatial_merge_size", 1)
        return MistralCausalLMPreprocessor(**kwargs)

    def test_multimodal_serialization(self):
        self.run_serialization_test(self._multimodal_preprocessor())

    def test_compute_image_block_ids(self):
        # An 8x8 image with `patch_size=4`, `spatial_merge_size=1`: a 2x2
        # grid of placeholder tokens, i.e. 2 rows of 2 tokens each, each row
        # terminated by the break token, except the final row, whose
        # trailing break becomes the end token.
        preprocessor = self._multimodal_preprocessor()
        placeholder_id = preprocessor.tokenizer.image_placeholder_token_id
        break_id = preprocessor.tokenizer.image_break_token_id
        end_id = preprocessor.tokenizer.image_end_token_id

        block = preprocessor._compute_image_block_ids(height=8, width=8)
        self.assertEqual(
            block,
            [
                placeholder_id,
                placeholder_id,
                break_id,
                placeholder_id,
                placeholder_id,
                end_id,
            ],
        )

        # 8x12 -> 2 rows of 3. 12x8 -> 3 rows of 2.
        block = preprocessor._compute_image_block_ids(height=8, width=12)
        self.assertEqual(
            block,
            [placeholder_id] * 3 + [break_id] + [placeholder_id] * 3 + [end_id],
        )
        block = preprocessor._compute_image_block_ids(height=12, width=8)
        self.assertEqual(
            block,
            [placeholder_id, placeholder_id, break_id] * 2
            + [placeholder_id, placeholder_id, end_id],
        )

    def test_tokenize_with_image_blocks_count_mismatch_raises(self):
        preprocessor = self._multimodal_preprocessor()
        with self.assertRaises(ValueError):
            preprocessor._tokenize_with_image_blocks(
                "one [IMG] two [IMG]", image_sizes=[(8, 8)]
            )

    def _tokenize_to_list(self, preprocessor, text):
        ids = preprocessor.tokenizer(text)
        return ids.numpy().tolist() if hasattr(ids, "numpy") else list(ids)

    def test_tokenize_with_image_blocks_no_images(self):
        preprocessor = self._multimodal_preprocessor()
        token_ids = preprocessor._tokenize_with_image_blocks(
            "just text, no images", image_sizes=[]
        )
        self.assertEqual(
            token_ids,
            self._tokenize_to_list(preprocessor, "just text, no images"),
        )

    def test_tokenize_with_image_blocks_matches_whole_string_boundaries(self):
        # `_tokenize_with_image_blocks` must tokenize the raw prompt as one
        # whole string, then splice in the precomputed image block -- not
        # tokenize text fragments split around the placeholder
        # independently, which would not reproduce SentencePiece's
        # whole-string boundary/whitespace handling.
        preprocessor = self._multimodal_preprocessor()
        prompt = "look [IMG] here"
        token_ids = preprocessor._tokenize_with_image_blocks(
            prompt, image_sizes=[(8, 8)]
        )

        placeholder_id = preprocessor.tokenizer.image_placeholder_token_id
        base_ids = self._tokenize_to_list(preprocessor, prompt)
        expected_block = preprocessor._compute_image_block_ids(8, 8)
        idx = base_ids.index(placeholder_id)
        expected = base_ids[:idx] + expected_block + base_ids[idx + 1 :]
        self.assertEqual(token_ids, expected)

    def test_build_multimodal_inputs_ordering_and_expansion(self):
        preprocessor = self._multimodal_preprocessor()

        # Image 0 for prompt A; images 1 and 2 (in that order) for prompt B.
        # Fill each image with a distinct constant value so we can verify,
        # after conversion, that `pixel_values` preserves the flattening
        # order: batch-row-major, then per-prompt left-to-right.
        image_0 = np.full((8, 8, 3), 0.0, dtype="float32")
        image_1 = np.full((8, 12, 3), 50.0, dtype="float32")
        image_2 = np.full((12, 8, 3), 100.0, dtype="float32")

        prompts = ["look [IMG] here", "two [IMG] and [IMG] here"]
        images_per_prompt = [[image_0], [image_1, image_2]]

        tokenized, pixel_values, image_sizes = (
            preprocessor._build_multimodal_inputs(prompts, images_per_prompt)
        )
        pixel_values = np.array(pixel_values)
        image_sizes = np.array(image_sizes)

        # === Ordering ===
        self.assertEqual(pixel_values.shape[0], 3)
        self.assertAllEqual(
            image_sizes, np.array([[8, 8], [8, 12], [12, 8]], dtype="int32")
        )

        def normalize(value):
            mean = np.array(
                [0.48145466, 0.4578275, 0.40821073], dtype="float32"
            )
            std = np.array(
                [0.26862954, 0.26130258, 0.27577711], dtype="float32"
            )
            return (value / 255.0 - mean) / std

        self.assertAllClose(pixel_values[0, :, 0, 0], normalize(0.0), atol=1e-4)
        self.assertAllClose(
            pixel_values[1, :, 0, 0], normalize(50.0), atol=1e-4
        )
        self.assertAllClose(
            pixel_values[2, :, 0, 0], normalize(100.0), atol=1e-4
        )

        # === Expansion ===
        # Boundary-correctness of `_tokenize_with_image_blocks` itself is
        # covered by `test_tokenize_with_image_blocks_matches_whole_string_
        # boundaries`; this only needs to verify each prompt got tokenized
        # with the right per-image block, in order.
        self.assertEqual(
            tokenized[0],
            preprocessor._tokenize_with_image_blocks(prompts[0], [(8, 8)]),
        )
        self.assertEqual(
            tokenized[1],
            preprocessor._tokenize_with_image_blocks(
                prompts[1], [(8, 12), (12, 8)]
            ),
        )

    def test_build_multimodal_inputs_zero_images_returns_none(self):
        preprocessor = self._multimodal_preprocessor()
        prompts, pixel_values, image_sizes = (
            preprocessor._build_multimodal_inputs(
                ["just text, no images"], [[]]
            )
        )
        self.assertEqual(prompts, ["just text, no images"])
        self.assertIsNone(pixel_values)
        self.assertIsNone(image_sizes)

    def test_multimodal_generate_preprocess_text_only(self):
        preprocessor = self._multimodal_preprocessor()
        x = preprocessor.generate_preprocess("the quick")
        self.assertNotIn("pixel_values", x)
        self.assertNotIn("image_sizes", x)
        self.assertNotIn("placeholder_indices", x)

    def test_multimodal_call_zero_images_raises(self):
        preprocessor = self._multimodal_preprocessor()
        with self.assertRaises(ValueError):
            preprocessor({"prompts": ["just text, no images"]})

    def test_text_only_model_flag_and_delegation(self):
        preprocessor = MistralCausalLMPreprocessor(**self.init_kwargs)
        self.assertTrue(preprocessor.text_only_model)

        input_data = "the quick brown fox"
        x = preprocessor.generate_preprocess(input_data)
        # No vision keys should be present for a text-only preprocessor.
        self.assertNotIn("pixel_values", x)
        self.assertNotIn("placeholder_indices", x)

        # Output must match the base (unmodified) text-only behavior
        # exactly.
        self.assertAllEqual(x["token_ids"], [1, 3, 8, 4, 6, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

        x, y, sw = preprocessor(["the quick brown fox"])
        self.assertNotIn("pixel_values", x)
        self.assertAllEqual(x["token_ids"], [[1, 3, 8, 4, 6, 2, 0, 0]])
        self.assertAllEqual(y, [[3, 8, 4, 6, 2, 0, 0, 0]])
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 0, 0, 0]])

    def test_multimodal_generate_preprocess_output_keys(self):
        preprocessor = self._multimodal_preprocessor()
        image = np.zeros((8, 8, 3), dtype="float32")
        # "the"/"quick" are real vocabulary words in the checked-in test
        # vocab's training data, so this exercises the real tokenizer's
        # actual BPE/SentencePiece tokenization end to end, not a fake
        # stand-in.
        x = preprocessor.generate_preprocess(
            {"prompts": "the [IMG] quick", "images": [image]}
        )
        for key in (
            "token_ids",
            "padding_mask",
            "pixel_values",
            "image_sizes",
            "placeholder_indices",
        ):
            self.assertIn(key, x)
        # 4 placeholder tokens (2x2 grid) should be present in `token_ids`.
        token_ids = np.array(x["token_ids"])
        num_placeholders = int(
            np.sum(
                token_ids == preprocessor.tokenizer.image_placeholder_token_id
            )
        )
        self.assertEqual(num_placeholders, 4)
