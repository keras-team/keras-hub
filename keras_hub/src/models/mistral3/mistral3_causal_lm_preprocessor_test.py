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
from keras_hub.src.models.mistral3.mistral3_tokenizer import (
    MISTRAL3_TEKKEN_SPLIT_PATTERN as _TEKKEN_SPLIT_PATTERN,
)
from keras_hub.src.models.mistral3.mistral3_tokenizer import Mistral3Tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils import tensor_utils
from keras_hub.src.utils.tensor_utils import tf


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _tekken_vision_init_kwargs():
    """Build a tiny Tekken (byte-level BPE) vocabulary with image tokens."""
    byte_encoder = _bytes_to_unicode()
    special_tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<pad>",
        "[INST]",
        "[IMG]",
        "[IMG_BREAK]",
        "[IMG_END]",
    ]
    vocabulary = {token: i for i, token in enumerate(special_tokens)}
    offset = len(special_tokens)
    for i in range(256):
        vocabulary[byte_encoder[i]] = offset + i
    merges = []
    next_id = offset + 256
    for a, b in [("t", "h"), ("th", "e"), ("i", "n")]:
        vocabulary[a + b] = next_id
        merges.append(f"{a} {b}")
        next_id += 1
    return {
        "vocabulary": vocabulary,
        "merges": merges,
        "split_pattern": _TEKKEN_SPLIT_PATTERN,
    }


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
        input_data = {"prompts": ["the tin", "in the"]}
        self.run_preprocessor_test(
            cls=Mistral3CausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
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
        token_ids = np.array(x["token_ids"])
        num_placeholders = int(
            np.sum(
                token_ids == preprocessor.tokenizer.image_placeholder_token_id
            )
        )
        self.assertEqual(num_placeholders, 4)

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
        token_ids = np.array(x_out["token_ids"])[0]
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
