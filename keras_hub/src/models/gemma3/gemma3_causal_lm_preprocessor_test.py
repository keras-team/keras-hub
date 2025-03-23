import os

import numpy as np
from keras import ops

from keras_hub.src.models.gemma3.gemma3_causal_lm_preprocessor import (
    Gemma3CausalLMPreprocessor,
)
from keras_hub.src.models.gemma3.gemma3_image_converter import (
    Gemma3ImageConverter,
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
        self.image_converter = Gemma3ImageConverter(
            image_size=(4, 4),
        )
        self.preprocessor = Gemma3CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            sequence_length=10,
            image_max_length=5,
            num_vision_tokens_per_image=20,
        )

    def test_call_with_vision(self):
        images = [
            np.ones((1, 10, 10, 3), dtype=np.float32),
            np.ones((2, 10, 10, 3), dtype=np.float32),
            np.ones((1, 10, 10, 3), dtype=np.float32),
        ]
        prompts = [
            "who is this cricketer <img>",
            "different flowers 1) <img> 2) <img>",
            "hey <img>",
        ]
        responses = ["bumrah", "hibiscus, sunflower", "you"]
        x = {"images": images, "prompts": prompts, "responses": responses}
        x, y, sw = self.preprocessor(x)

        self.assertEqual(ops.shape(x["images"]), (3, 1, 5, 4, 4, 3))
        self.assertEqual(ops.shape(x["token_ids"]), (3, 110))
        self.assertEqual(ops.shape(x["text_mask"]), (3, 110))
        self.assertEqual(ops.shape(x["response_mask"]), (3, 110))
        self.assertEqual(ops.shape(x["padding_mask"]), (3, 110))
        self.assertEqual(ops.shape(y), (3, 110))
        self.assertEqual(ops.shape(sw), (3, 110))

    def test_call_with_vision_bsz_1(self):
        images = [
            np.ones((1, 10, 10, 3), dtype=np.float32),
        ]
        prompts = ["who is this cricketer <img>"]
        responses = ["bumrah"]
        x = {"images": images, "prompts": prompts, "responses": responses}
        x, y, sw = self.preprocessor(x)

        self.assertEqual(ops.shape(x["images"]), (1, 1, 5, 4, 4, 3))
        self.assertEqual(ops.shape(x["token_ids"]), (1, 110))
        self.assertEqual(ops.shape(x["text_mask"]), (1, 110))
        self.assertEqual(ops.shape(x["response_mask"]), (1, 110))
        self.assertEqual(ops.shape(x["padding_mask"]), (1, 110))
        self.assertEqual(ops.shape(y), (1, 110))
        self.assertEqual(ops.shape(sw), (1, 110))

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

        self.assertEqual(ops.shape(x["images"]), (3, 0, 5, 4, 4, 3))
        self.assertEqual(ops.shape(x["token_ids"]), (3, 10))
        self.assertEqual(ops.shape(x["text_mask"]), (3, 10))
        self.assertEqual(ops.shape(x["response_mask"]), (3, 10))
        self.assertEqual(ops.shape(x["padding_mask"]), (3, 10))
        self.assertEqual(ops.shape(y), (3, 10))
        self.assertEqual(ops.shape(sw), (3, 10))
