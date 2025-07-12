import numpy as np
import pytest

from keras_hub.src.models.clip.clip_image_converter import CLIPImageConverter
from keras_hub.src.models.clip.clip_preprocessor import CLIPPreprocessor
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.tests.test_case import TestCase


class CLIPPreprocessorTest(TestCase):
    def setUp(self):
        vocab = ["air", "plane</w>", "port</w>"]
        vocab += ["<|endoftext|>", "<|startoftext|>"]
        vocab = dict([(token, i + 1) for i, token in enumerate(vocab)])
        merges = ["a i", "p l", "n e</w>", "p o", "r t</w>", "ai r", "pl a"]
        merges += ["po rt</w>", "pla ne</w>"]
        self.tokenizer = CLIPTokenizer(vocabulary=vocab, merges=merges)
        self.image_converter = CLIPImageConverter(
            (224, 224),
            [2.0 / 255.0] * 3,
            [-1.0] * 3,
            interpolation="bicubic",
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 8,
        }
        self.input_data = {
            "prompts": [" airplane airport"],
            "images": [np.zeros([512, 512, 3])],
        }

    def test_preprocessor_basics(self):
        self.run_preprocessing_layer_test(
            cls=CLIPPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output={
                "token_ids": [[5, 1, 2, 1, 3, 4, 0, 0]],
                "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                "images": np.ones([1, 224, 224, 3]) * -1.0,
            },
        )

    def test_without_images(self):
        input_data = {
            "prompts": [" airplane airport"] * 4,
            "images": None,
        }
        preprocessor = CLIPPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            sequence_length=8,
            add_start_token=False,
            add_end_token=False,
        )
        x = preprocessor(input_data)
        self.assertIsNone(x["images"])

    def test_no_start_end_token(self):
        input_data = {
            "prompts": [" airplane airport"] * 4,
            "images": [np.zeros([512, 512, 3])],
        }
        preprocessor = CLIPPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.image_converter,
            sequence_length=8,
            add_start_token=False,
            add_end_token=False,
        )
        x = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[1, 2, 1, 3, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)

    def test_sequence_length_override(self):
        input_data = {
            "prompts": " airplane airport",
            "images": [np.zeros([512, 512, 3])],
        }
        preprocessor = CLIPPreprocessor(**self.init_kwargs)
        x = preprocessor(input_data, sequence_length=5)
        self.assertAllEqual(x["token_ids"], [5, 1, 2, 1, 4])

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        self.skipTest("TODO")
        for preset in CLIPPreprocessor.presets:
            self.run_preset_test(
                cls=CLIPPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
