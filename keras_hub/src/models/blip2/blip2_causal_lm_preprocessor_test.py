"""Tests for BLIP-2 Causal LM preprocessor."""

import numpy as np
import pytest

from keras_hub.src.models.blip2.blip2_causal_lm_preprocessor import (
    Blip2CausalLMPreprocessor,
)
from keras_hub.src.models.blip2.blip2_image_converter import Blip2ImageConverter
from keras_hub.src.models.blip2.blip2_tokenizer import Blip2Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Blip2CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        vocab = {
            "<pad>": 1,
            "</s>": 2,
            "<image>": 3,
            "\u010a": 4,
            "Ġ": 5,
            "t": 6,
            "h": 7,
            "e": 8,
            "q": 9,
            "u": 10,
            "i": 11,
            "k": 12,
            "he": 13,
            "Ġt": 14,
        }
        merges = ["h e", "Ġ t"]
        self.tokenizer = Blip2Tokenizer(vocabulary=vocab, merges=merges)
        self.image_converter = Blip2ImageConverter(image_size=(4, 4))

        self.init_text_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": None,
            "sequence_length": 10,
        }
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 10,
        }
        self.input_data = {
            "images": np.ones((2, 32, 32, 3), dtype="float32"),
            "text": ["the quick", "the"],
        }

    def test_text_preprocessor_basics(self):
        self.run_preprocessing_layer_test(
            cls=Blip2CausalLMPreprocessor,
            init_kwargs=self.init_text_kwargs,
            input_data={"text": ["the quick", "the"]},
        )

    def test_preprocessor_basics(self):
        self.run_preprocessing_layer_test(
            cls=Blip2CausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_text_output_shapes(self):
        preprocessor = Blip2CausalLMPreprocessor(**self.init_text_kwargs)
        x, y, sw = preprocessor({"text": ["the quick", "the"]})
        self.assertEqual(x["token_ids"].shape, (2, 10))
        self.assertEqual(x["padding_mask"].shape, (2, 10))
        self.assertEqual(y.shape, (2, 10))
        self.assertEqual(sw.shape, (2, 10))

    def test_vision_output_shapes(self):
        preprocessor = Blip2CausalLMPreprocessor(**self.init_kwargs)
        x, y, sw = preprocessor(self.input_data)
        self.assertEqual(x["token_ids"].shape, (2, 10))
        self.assertEqual(x["padding_mask"].shape, (2, 10))
        self.assertAllEqual(x["images"].shape, (2, 4, 4, 3))

    def test_text_no_start_end_token(self):
        preprocessor = Blip2CausalLMPreprocessor(
            **self.init_text_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor({"text": ["the"]})
        self.assertEqual(x["token_ids"].shape[-1], 10)
        self.assertEqual(x["padding_mask"].shape[-1], 10)

    def test_no_start_end_token(self):
        preprocessor = Blip2CausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(self.input_data)
        self.assertEqual(x["token_ids"].shape[-1], 10)
        self.assertEqual(x["padding_mask"].shape[-1], 10)

    def test_text_generate_preprocess(self):
        preprocessor = Blip2CausalLMPreprocessor(**self.init_text_kwargs)
        x = preprocessor.generate_preprocess({"text": "the"})
        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)
        self.assertEqual(len(x["token_ids"]), 10)

    def test_generate_preprocess(self):
        preprocessor = Blip2CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess({
            "images": np.ones((32, 32, 3), dtype="float32"),
            "text": "the",
        })
        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)
        self.assertIn("images", x)
        self.assertEqual(len(x["token_ids"]), 10)
        self.assertAllEqual(x["images"].shape[-3:], (4, 4, 3))

    def test_text_generate_postprocess(self):
        preprocessor = Blip2CausalLMPreprocessor(**self.init_text_kwargs)
        preprocessed = preprocessor.generate_preprocess({"text": "the"})
        result = preprocessor.generate_postprocess(preprocessed)
        self.assertIsInstance(result, (str, list))

    def test_vision_generate_postprocess(self):
        preprocessor = Blip2CausalLMPreprocessor(**self.init_kwargs)
        preprocessed = preprocessor.generate_preprocess({
            "images": np.ones((32, 32, 3), dtype="float32"),
            "text": "the",
        })
        result = preprocessor.generate_postprocess(preprocessed)
        self.assertIsInstance(result, (str, list))

    def test_text_only_input_to_vision_preprocessor(self):
        preprocessor = Blip2CausalLMPreprocessor(**self.init_kwargs)
        x, y, sw = preprocessor({"text": ["the quick", "the"]})
        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)
        self.assertNotIn("images", x)

    def test_ragged_images(self):
        preprocessor = Blip2CausalLMPreprocessor(**self.init_kwargs)
        input_data = {
            "images": [
                np.ones((32, 32, 3), dtype="float32"),
                np.ones((32, 32, 3), dtype="float32"),
            ],
            "text": ["the quick", "the"],
        }
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["images"].shape, (2, 4, 4, 3))

    def test_images_in_text_only_mode_raises(self):
        preprocessor = Blip2CausalLMPreprocessor(**self.init_text_kwargs)
        with self.assertRaises(ValueError):
            preprocessor({
                "images": np.ones((2, 32, 32, 3), dtype="float32"),
                "text": ["the quick", "the"],
            })


    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Blip2CausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=Blip2CausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )