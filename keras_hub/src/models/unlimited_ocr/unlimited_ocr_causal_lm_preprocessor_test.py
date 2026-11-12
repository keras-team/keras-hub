import grain
import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.unlimited_ocr.unlimited_ocr_causal_lm_preprocessor import (  # noqa: E501
    UnlimitedOCRCausalLMPreprocessor,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_image_converter import (
    UnlimitedOCRImageConverter,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_tokenizer import (
    OCR_MARKER_TOKENS,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_tokenizer import (
    UnlimitedOCRTokenizer,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tests.test_case import assert_grain_safe_types
from keras_hub.src.utils.tensor_utils import tf


class UnlimitedOCRCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        merges += ["Ġai r", "Ġa i", "pla ne"]
        vocab = []
        for merge in merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab += ["!", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"]
        vocab += ["<｜▁pad▁｜>", "<image>"]
        vocab += list(OCR_MARKER_TOKENS)
        vocab = sorted(set(vocab))
        vocab = dict([(token, i) for i, token in enumerate(vocab)])
        self.tokenizer = UnlimitedOCRTokenizer(vocabulary=vocab, merges=merges)
        # A 64 pixel view gives 2x2 global tokens. A 40 pixel tile gives 2x2
        # local tokens.
        self.image_converter = UnlimitedOCRImageConverter(
            global_size=64,
            tile_size=40,
            min_tiles=2,
            max_tiles=6,
            patch_size=8,
            downsample_ratio=4,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 64,
        }
        self.image_token_id = self.tokenizer.image_token_id
        # A 50x100 image has two tile columns and one row. It has 10 local
        # tokens and 7 global tokens: 17 image tokens.
        rng = np.random.RandomState(0)
        self.wide_image = rng.randint(0, 256, (50, 100, 3)).astype("float32")
        # A 30x30 image has no tile: 7 image tokens.
        self.small_image = rng.randint(0, 256, (30, 30, 3)).astype("float32")

    def test_preprocessor_basics(self):
        # The base mode has no tiles. Then every output dimension is static
        # for one prompt, so the helper can batch unbatched `tf.data`
        # elements. `run_preprocessor_test` builds a second layer from
        # `self.init_kwargs`.
        self.init_kwargs = {
            **self.init_kwargs,
            "image_converter": UnlimitedOCRImageConverter(
                global_size=64,
                tile_size=40,
                min_tiles=2,
                max_tiles=6,
                crop_to_patches=False,
                patch_size=8,
                downsample_ratio=4,
            ),
        }
        self.run_preprocessor_test(
            cls=UnlimitedOCRCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data={
                "prompts": ["<image>airport", "<image>airplane"],
                "images": np.stack([self.wide_image, self.wide_image]),
            },
        )

    def test_samples_with_fewer_images_are_padded(self):
        preprocessor = UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs)
        # Sample 0 has two images and four tiles. Sample 1 has one image and
        # two tiles.
        x = preprocessor.generate_preprocess(
            {
                "prompts": ["<image>airport<image>", "<image>airport"],
                "images": np.stack([self.wide_image] * 3),
            }
        )
        pixel_values = ops.convert_to_numpy(x["pixel_values"])
        pixel_values_local = ops.convert_to_numpy(x["pixel_values_local"])
        self.assertEqual(pixel_values.shape, (2, 2, 64, 64, 3))
        self.assertEqual(pixel_values_local.shape, (2, 4, 40, 40, 3))
        self.assertAllEqual(
            x["patches_grid"], [[[2, 1], [2, 1]], [[2, 1], [0, 0]]]
        )
        # The padding image and the padding tiles have zero pixels.
        self.assertAllEqual(pixel_values[1, 1], np.zeros((64, 64, 3)))
        self.assertAllEqual(pixel_values_local[1, 2:], np.zeros((2, 40, 40, 3)))
        # The real images are the same wide image.
        self.assertAllClose(pixel_values[0, 0], pixel_values[1, 0])
        self.assertAllClose(
            pixel_values_local[0, :2], pixel_values_local[1, :2]
        )

    def assert_outputs_match(self, expected, actual):
        """Compare the outputs of two preprocessing paths."""
        expected = keras.utils.unpack_x_y_sample_weight(expected)
        actual = keras.utils.unpack_x_y_sample_weight(actual)
        for key in expected[0]:
            self.assertAllClose(expected[0][key], np.array(actual[0][key]))
        for expected_part, actual_part in zip(expected[1:], actual[1:]):
            if expected_part is not None:
                self.assertAllClose(expected_part, np.array(actual_part))

    def test_sequence_length_is_respected(self):
        preprocessor = UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs)
        x = {
            "prompts": ["<image>airplane at airport", "<image>airport"],
            "images": np.stack([self.wide_image, self.wide_image]),
        }
        preprocessor.sequence_length = 40
        output = keras.utils.unpack_x_y_sample_weight(preprocessor(x))[0]
        self.assertEqual(output["token_ids"].shape[-1], 40)
        generated = preprocessor.generate_preprocess(x)
        self.assertEqual(generated["token_ids"].shape[-1], 40)
        if tf is not None:
            ds = tf.data.Dataset.from_tensor_slices(x).batch(2)
            mapped = next(iter(ds.map(preprocessor)))
            mapped_x = keras.utils.unpack_x_y_sample_weight(mapped)[0]
            self.assertEqual(mapped_x["token_ids"].shape[-1], 40)

    def test_image_tokens_expand_to_the_feature_count(self):
        preprocessor = UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(
            {
                "prompts": ["<image>airplane at airport", "<image>airport"],
                "images": [self.wide_image, self.small_image],
            }
        )
        token_ids = ops.convert_to_numpy(x["token_ids"])
        self.assertEqual(token_ids.shape, (2, 64))
        counts = (token_ids == self.image_token_id).sum(axis=1)
        self.assertAllEqual(counts, [17, 7])
        # The start token comes first.
        start_ids = [self.tokenizer.start_token_id] * 2
        self.assertAllEqual(token_ids[:, 0], start_ids)
        self.assertEqual(tuple(x["pixel_values"].shape), (2, 1, 64, 64, 3))
        shape = tuple(x["pixel_values_local"].shape)
        self.assertEqual(shape, (2, 2, 40, 40, 3))
        self.assertAllEqual(x["patches_grid"], [[[2, 1]], [[0, 0]]])

    def test_images_are_matched_to_placeholders_in_order(self):
        preprocessor = UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(
            {
                "prompts": ["<image>airport<image>"],
                "images": [self.small_image, self.wide_image],
            }
        )
        token_ids = ops.convert_to_numpy(x["token_ids"])
        self.assertEqual(int((token_ids == self.image_token_id).sum()), 24)
        self.assertAllEqual(x["patches_grid"], [[[0, 0], [2, 1]]])

    def test_text_only_adds_empty_image_inputs(self):
        preprocessor = UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(["airplane at airport"])
        self.assertEqual(tuple(x["pixel_values"].shape), (1, 0, 64, 64, 3))
        shape = tuple(x["pixel_values_local"].shape)
        self.assertEqual(shape, (1, 0, 40, 40, 3))
        self.assertEqual(tuple(x["patches_grid"].shape), (1, 0, 2))
        self.assertEqual(tuple(x["token_ids"].shape), (1, 64))

    def test_call_returns_shifted_labels(self):
        preprocessor = UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs)
        x, y, sample_weight = preprocessor(
            {
                "prompts": ["<image>airplane at airport"],
                "images": [self.small_image],
            }
        )
        token_ids = ops.convert_to_numpy(x["token_ids"])
        labels = ops.convert_to_numpy(y)
        self.assertEqual(token_ids.shape, (1, 64))
        # `length` counts the real tokens of the inputs. The inputs end with
        # the end token. The labels are the inputs shifted by one position.
        length = int(ops.convert_to_numpy(x["padding_mask"]).sum())
        self.assertEqual(token_ids[0, length - 1], self.tokenizer.end_token_id)
        self.assertAllEqual(labels[0, : length - 1], token_ids[0, 1:length])
        weight_total = int(ops.convert_to_numpy(sample_weight).sum())
        self.assertEqual(weight_total, length - 1)

    def test_unbatched_input_squeezes_the_token_axis(self):
        preprocessor = UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(
            {"prompts": "<image>airport", "images": [self.small_image]}
        )
        self.assertEqual(tuple(x["token_ids"].shape), (64,))
        # The image inputs have no batch axis either.
        self.assertEqual(tuple(x["pixel_values"].shape), (1, 64, 64, 3))

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_tf_data_matches_direct_call(self):
        preprocessor = UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs)
        generate = preprocessor.generate_preprocess
        with_images = {
            "prompts": ["<image>airplane at airport", "<image>airport"],
            "images": np.stack([self.wide_image, self.wide_image]),
        }
        # Samples with fewer images than placeholders get padding. One
        # element holds the whole batch.
        padded = {
            "prompts": ["<image>airport<image>", "<image>airport", "airport"],
            "images": np.stack([self.wide_image] * 3),
        }

        def batched(x):
            return tf.data.Dataset.from_tensor_slices(x).batch(2)

        def one_element(x):
            return tf.data.Dataset.from_tensors(x)

        text_only = ["airplane at airport", "airport"]
        cases = [
            ("call", preprocessor, with_images, batched),
            ("call, text only", preprocessor, text_only, batched),
            ("generate_preprocess", generate, with_images, batched),
            ("generate_preprocess, padded", generate, padded, one_element),
        ]
        for name, function, x, make_dataset in cases:
            with self.subTest(name):
                direct = function(x)
                mapped = next(iter(make_dataset(x).map(function)))
                self.assert_outputs_match(direct, mapped)

    def test_grain_matches_direct_call(self):
        # The Python path must return NumPy arrays inside Grain.
        preprocessor = UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs)
        x = {
            "prompts": ["<image>airplane at airport", "<image>airport"],
            "images": np.stack([self.wide_image, self.wide_image]),
        }
        direct = preprocessor(x)
        dataset = grain.MapDataset.source([x])
        (output,) = list(dataset.map(preprocessor))
        assert_grain_safe_types(output)
        self.assert_outputs_match(direct, output)

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_tf_data_two_images_in_one_prompt(self):
        # An unbatched element holds one prompt with two placeholders.
        preprocessor = UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs)
        x = {
            "prompts": "<image>airport<image>",
            "images": np.stack([self.wide_image, self.wide_image]),
        }
        direct = preprocessor(x)
        ds = tf.data.Dataset.from_tensors(x)
        mapped = next(iter(ds.map(preprocessor)))
        self.assert_outputs_match(direct, mapped)
        mapped_x = keras.utils.unpack_x_y_sample_weight(mapped)[0]
        counts = (mapped_x["token_ids"].numpy() == self.image_token_id).sum()
        self.assertEqual(int(counts), 34)

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_tf_data_invalid_images_raise(self):
        short = UnlimitedOCRCausalLMPreprocessor(
            **{**self.init_kwargs, "sequence_length": 10}
        )
        cases = [
            (
                "number of images",
                UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs),
                {
                    "prompts": ["<image>airport"],
                    "images": np.zeros((1, 0, 50, 100, 3), "float32"),
                },
            ),
            (
                "cuts the image tokens",
                short,
                {
                    "prompts": ["<image>airport"],
                    "images": self.wide_image[None, None],
                },
            ),
        ]
        for message, preprocessor, x in cases:
            with self.subTest(message):
                ds = tf.data.Dataset.from_tensor_slices(x).batch(1)
                with self.assertRaisesRegex(
                    tf.errors.InvalidArgumentError, message
                ):
                    next(iter(ds.map(preprocessor.generate_preprocess)))

    def test_mismatched_image_count_raises(self):
        preprocessor = UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "number of images"):
            preprocessor.generate_preprocess(
                {"prompts": ["<image>airport"], "images": []}
            )

    def test_truncated_image_raises(self):
        preprocessor = UnlimitedOCRCausalLMPreprocessor(
            **{**self.init_kwargs, "sequence_length": 10}
        )
        with self.assertRaisesRegex(ValueError, "cuts the image tokens"):
            preprocessor.generate_preprocess(
                {"prompts": ["<image>airport"], "images": [self.wide_image]}
            )

    def test_generate_postprocess_keeps_ocr_markers(self):
        preprocessor = UnlimitedOCRCausalLMPreprocessor(**self.init_kwargs)
        vocab = self.tokenizer.vocabulary
        ids = [
            self.tokenizer.start_token_id,
            vocab["<|det|>"],
            self.tokenizer.token_to_id("ai"),
            vocab["<|/det|>"],
            self.tokenizer.end_token_id,
        ]
        text = preprocessor.generate_postprocess(
            {
                "token_ids": np.array([ids], dtype="int32"),
                "padding_mask": np.ones((1, len(ids)), dtype="bool"),
            }
        )
        self.assertEqual(text[0], "<|det|>ai<|/det|>")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in UnlimitedOCRCausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=UnlimitedOCRCausalLMPreprocessor,
                preset=preset,
                input_data=["document parsing."],
            )
