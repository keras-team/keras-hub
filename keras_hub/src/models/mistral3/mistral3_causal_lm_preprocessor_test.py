from unittest import mock

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.mistral3 import (
    mistral3_causal_lm_preprocessor as preprocessor_module,
)
from keras_hub.src.models.mistral3.mistral3_causal_lm_preprocessor import (
    Mistral3CausalLMPreprocessor,
)
from keras_hub.src.models.mistral3.mistral3_image_converter import (
    Mistral3ImageConverter,
)
from keras_hub.src.models.mistral3.mistral3_tokenizer import Mistral3Tokenizer
from keras_hub.src.models.mistral3.mistral3_tokenizer_test import (
    _tekken_vision_init_kwargs,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils import tensor_utils
from keras_hub.src.utils.tensor_utils import tf


class Mistral3CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = Mistral3Tokenizer(**_tekken_vision_init_kwargs())
        self.image_converter = Mistral3ImageConverter(
            longest_edge=16, patch_size=4, spatial_merge_size=1
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 32,
            "spatial_merge_size": 1,
        }

    def test_preprocessor_basics(self):
        # Zero-image inputs do not batch like per-example data, so the
        # shared helper runs without an image converter.
        init_kwargs = {**self.init_kwargs, "image_converter": None}
        input_data = {"prompts": ["the tin", "in the"]}
        self.run_preprocessor_test(
            cls=Mistral3CausalLMPreprocessor,
            init_kwargs=init_kwargs,
            input_data=input_data,
        )

    def test_generate_preprocess_with_images(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        image = np.zeros((8, 8, 3), dtype="float32")
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
        token_ids = ops.convert_to_numpy(x["token_ids"])
        num_placeholders = int(
            np.sum(
                token_ids == preprocessor.tokenizer.image_placeholder_token_id
            )
        )
        self.assertEqual(num_placeholders, 4)

    def test_unbatched_generate_preprocess_squeezes_placeholder_indices(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        image = np.zeros((8, 8, 3), dtype="float32")
        x = preprocessor.generate_preprocess(
            {"prompts": "the [IMG] quick", "images": [image]}
        )
        self.assertEqual(len(np.shape(x["token_ids"])), 1)
        self.assertEqual(len(np.shape(x["placeholder_indices"])), 1)

    def test_unbatched_call_squeezes_placeholder_indices(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        image = np.zeros((8, 8, 3), dtype="float32")
        x, y, sample_weight = preprocessor(
            {"prompts": "the [IMG] quick", "images": [image]}
        )
        self.assertEqual(len(np.shape(x["token_ids"])), 1)
        self.assertEqual(len(np.shape(x["placeholder_indices"])), 1)
        self.assertEqual(len(np.shape(y)), 1)
        self.assertEqual(len(np.shape(sample_weight)), 1)

    def test_generate_preprocess_rejects_truncated_image_block(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        image = np.zeros((8, 8, 3), dtype="float32")
        # The image block is 6 tokens. The sequence only fits part of it.
        with self.assertRaisesRegex(ValueError, "cuts an image block"):
            preprocessor.generate_preprocess(
                {"prompts": "the [IMG] quick", "images": [image]},
                sequence_length=5,
            )

    def test_call_rejects_truncated_image_block(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        image = np.zeros((8, 8, 3), dtype="float32")
        with self.assertRaisesRegex(ValueError, "cuts an image block"):
            preprocessor(
                {"prompts": ["the [IMG] quick"], "images": [image]},
                sequence_length=5,
            )

    def test_generate_preprocess_with_images_when_tf_handle_is_none(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        image = np.zeros((8, 8, 3), dtype="float32")
        x = {"prompts": "the [IMG] quick", "images": [image]}

        with (
            mock.patch.object(preprocessor_module, "tf", None),
            mock.patch.object(tensor_utils, "tf", None),
        ):
            outputs = preprocessor.generate_preprocess(x)

        self.assertIn("pixel_values", outputs)
        self.assertIn("image_sizes", outputs)
        self.assertIn("placeholder_indices", outputs)
        self.assertAllEqual(outputs["image_sizes"], [[8, 8]])

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_generate_preprocess_with_images_in_tf_function(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)

        @tf.function
        def preprocess(prompts, images):
            return preprocessor.generate_preprocess(
                {"prompts": prompts, "images": images}
            )

        outputs = preprocess(
            tf.constant(["the [IMG] quick"]),
            tf.constant(np.zeros((1, 8, 8, 3), dtype="float32")),
        )
        token_ids = np.array(outputs["token_ids"])
        num_placeholders = int(
            np.sum(
                token_ids == preprocessor.tokenizer.image_placeholder_token_id
            )
        )

        self.assertEqual(num_placeholders, 4)
        self.assertAllEqual(outputs["image_sizes"], [[8, 8]])

    def test_call_with_images_when_tf_handle_is_none(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        image = np.zeros((8, 8, 3), dtype="float32")
        x = {"prompts": ["the [IMG] quick"], "images": [image]}

        with (
            mock.patch.object(preprocessor_module, "tf", None),
            mock.patch.object(tensor_utils, "tf", None),
        ):
            x_out, y, sample_weight = preprocessor(x)

        self.assertEqual(ops.shape(x_out["token_ids"]), (1, 32))
        self.assertEqual(ops.shape(y), (1, 32))
        self.assertEqual(ops.shape(sample_weight), (1, 32))
        self.assertAllEqual(x_out["image_sizes"], [[8, 8]])
        token_ids = ops.convert_to_numpy(x_out["token_ids"])[0]
        expected_indices = np.nonzero(
            token_ids == preprocessor.tokenizer.image_placeholder_token_id
        )[0]
        self.assertEqual(len(expected_indices), 4)
        self.assertAllEqual(x_out["placeholder_indices"], [expected_indices])

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_call_with_images_in_tf_data_matches_python(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        prompts = ["the [IMG] quick"]
        images = np.zeros((1, 8, 8, 3), dtype="float32")
        expected = preprocessor({"prompts": prompts, "images": images})

        ds = tf.data.Dataset.from_tensor_slices(
            {"prompts": prompts, "images": images}
        ).batch(1)
        outputs = next(iter(ds.map(preprocessor)))

        for output, expected_output in zip(
            tf.nest.flatten(outputs), tf.nest.flatten(expected)
        ):
            self.assertAllClose(output, expected_output)

    def test_generate_preprocess_text_only(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess("the tin")
        self.assertEqual(set(x.keys()), {"token_ids", "padding_mask"})

    def test_call_text_only_adds_empty_image_inputs(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        x, _, _ = preprocessor(["the tin", "in the"])
        self.assertEqual(ops.shape(x["pixel_values"])[0], 0)
        self.assertEqual(ops.shape(x["image_sizes"]), (0, 2))
        self.assertAllEqual(x["placeholder_indices"], [[-1], [-1]])

    def test_generate_postprocess(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        input_data = {
            "token_ids": ops.array([1, 265, 40, 124, 266, 0, 0, 0]),
            "padding_mask": ops.array(
                [True, True, True, True, True, False, False, False]
            ),
        }
        x = preprocessor.generate_postprocess(input_data)
        self.assertEqual(x, "the tin")

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        input_data = {
            "prompts": ["Describe the image. [IMG]"],
            "images": [[self.load_test_image()]],
        }
        for preset in Mistral3CausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=Mistral3CausalLMPreprocessor,
                preset=preset,
                input_data=input_data,
            )

    def test_call_mixed_batch_pads_text_only_row(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        image = np.zeros((8, 8, 3), dtype="float32")
        x, _, _ = preprocessor(
            {"prompts": ["the [IMG] quick", "the tin"], "images": [image]}
        )
        placeholder_indices = ops.convert_to_numpy(x["placeholder_indices"])
        token_ids = ops.convert_to_numpy(x["token_ids"])
        expected_indices = np.nonzero(
            token_ids[0] == preprocessor.tokenizer.image_placeholder_token_id
        )[0]
        self.assertEqual(len(expected_indices), 4)
        self.assertEqual(placeholder_indices.shape, (2, 4))
        self.assertAllEqual(placeholder_indices[0], expected_indices)
        self.assertAllEqual(placeholder_indices[1], [-1, -1, -1, -1])
        self.assertAllEqual(x["image_sizes"], [[8, 8]])
