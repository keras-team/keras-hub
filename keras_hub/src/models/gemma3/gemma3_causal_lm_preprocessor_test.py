import os

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.gemma3.gemma3_causal_lm_preprocessor import (
    Gemma3CausalLMPreprocessor,
)
from keras_hub.src.models.gemma3.gemma3_tokenizer import Gemma3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Gemma3CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = Gemma3Tokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "gemma3_test_vocab.spm"
            )
        )
        # TODO: Uncomment when we release vision.
        # self.image_converter = Gemma3ImageConverter(
        #     image_size=(4, 4),
        # )
        # self.preprocessor = Gemma3CausalLMPreprocessor(
        #     tokenizer=self.tokenizer,
        #     image_converter=self.image_converter,
        #     sequence_length=100,
        #     max_images_per_prompt=5,
        #     num_vision_tokens_per_image=20,
        # )

        self.text_only_preprocessor = Gemma3CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=None,
            sequence_length=100,
        )

    @pytest.mark.skipif(
        True,
        reason="disabled until the vision release.",
    )
    def test_call_with_vision(self):
        images = np.ones((3, 5, 10, 10, 3), dtype=np.float32)
        num_valid_images = np.array([1, 2, 1])
        prompts = [
            "who is this cricketer <start_of_image>",
            "different flowers 1) <start_of_image> 2) <start_of_image>",
            "hey <start_of_image>",
        ]
        responses = ["bumrah", "hibiscus, sunflower", "you"]
        x = {
            "images": images,
            "num_valid_images": num_valid_images,
            "prompts": prompts,
            "responses": responses,
        }
        x, y, sw = self.preprocessor(x)

        self.assertEqual(ops.shape(x["images"]), (3, 5, 4, 4, 3))
        self.assertEqual(ops.shape(x["token_ids"]), (3, 200))
        self.assertEqual(ops.shape(x["text_mask"]), (3, 200))
        self.assertEqual(ops.shape(x["padding_mask"]), (3, 200))
        self.assertEqual(ops.shape(y), (3, 200))
        self.assertEqual(ops.shape(sw), (3, 200))

    @pytest.mark.skipif(
        True,
        reason="disabled until the vision release.",
    )
    def test_call_with_vision_bsz_1(self):
        images = np.ones((1, 5, 10, 10, 3), dtype=np.float32)
        num_valid_images = np.array(
            [
                1,
            ],
            dtype=np.int32,
        )
        prompts = ["who is this cricketer <img>"]
        responses = ["bumrah"]
        x = {
            "images": images,
            "num_valid_images": num_valid_images,
            "prompts": prompts,
            "responses": responses,
        }
        x, y, sw = self.preprocessor(x)

        self.assertEqual(ops.shape(x["images"]), (1, 5, 4, 4, 3))
        self.assertEqual(ops.shape(x["token_ids"]), (1, 200))
        self.assertEqual(ops.shape(x["text_mask"]), (1, 200))
        self.assertEqual(ops.shape(x["padding_mask"]), (1, 200))
        self.assertEqual(ops.shape(y), (1, 200))
        self.assertEqual(ops.shape(sw), (1, 200))

    @pytest.mark.skipif(
        True,
        reason="disabled until the vision release.",
    )
    def test_call_without_vision(self):
        images = None
        prompts = [
            "virat kohli",
            "sachin tendulkar",
            "too many cricket references",
        ]
        responses = ["steve smith", "brian lara", "yes"]
        x = {"images": images, "prompts": prompts, "responses": responses}
        x, y, sw = self.preprocessor(x)

        self.assertEqual(ops.shape(x["images"]), (3, 0, 4, 4, 3))
        self.assertEqual(ops.shape(x["token_ids"]), (3, 100))
        self.assertEqual(ops.shape(x["text_mask"]), (3, 100))
        self.assertEqual(ops.shape(x["padding_mask"]), (3, 100))
        self.assertEqual(ops.shape(y), (3, 100))
        self.assertEqual(ops.shape(sw), (3, 100))

    def test_call_text_only_preprocessor(self):
        images = None
        prompts = [
            "virat kohli",
            "sachin tendulkar",
            "too many cricket references",
        ]
        responses = ["steve smith", "brian lara", "yes"]
        x = {"images": images, "prompts": prompts, "responses": responses}
        x, y, sw = self.preprocessor(x)

        self.assertEqual(ops.shape(x["images"]), (3, 0, 4, 4, 3))
        self.assertEqual(ops.shape(x["token_ids"]), (3, 100))
        self.assertEqual(ops.shape(x["text_mask"]), (3, 100))
        self.assertEqual(ops.shape(x["padding_mask"]), (3, 100))
        self.assertEqual(ops.shape(y), (3, 100))
        self.assertEqual(ops.shape(sw), (3, 100))
