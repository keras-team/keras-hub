import os
import string

import numpy as np
import pytest

from keras_hub.src.models.siglip.siglip_image_converter import (
    SigLIPImageConverter,
)
from keras_hub.src.models.siglip.siglip_preprocessor import SigLIPPreprocessor
from keras_hub.src.models.siglip.siglip_tokenizer import SigLIPTokenizer
from keras_hub.src.tests.test_case import TestCase


class SigLIPPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = SigLIPTokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "siglip_test_vocab.spm"
            ),
        )
        self.image_converter = SigLIPImageConverter(
            (224, 224),
            [2.0 / 255.0] * 3,
            [-1.0] * 3,
            crop_to_aspect_ratio=False,
            interpolation="bicubic",
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 8,
        }
        self.input_data = {
            "prompts": ["the quick brown fox"],
            "images": [np.zeros([512, 512, 3])],
        }

    def test_preprocessor_basics(self):
        self.run_preprocessing_layer_test(
            cls=SigLIPPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output={
                "token_ids": [[4, 9, 5, 7, 2, 2, 2, 2]],
                "padding_mask": [[1, 1, 1, 1, 1, 0, 0, 0]],
                "images": np.ones([1, 224, 224, 3]) * -1.0,
            },
        )

    def test_canonicalize_text(self):
        # Do upper case and add punctuations in the inputs.
        # The outputs must be identical to the inputs without them.
        input_data = self.input_data.copy()
        input_data["prompts"] = [
            f"{x.upper()}{string.punctuation}" for x in input_data["prompts"]
        ]
        self.run_preprocessing_layer_test(
            cls=SigLIPPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
            expected_output={
                "token_ids": [[4, 9, 5, 7, 2, 2, 2, 2]],
                "padding_mask": [[1, 1, 1, 1, 1, 0, 0, 0]],
                "images": np.ones([1, 224, 224, 3]) * -1.0,
            },
        )

    def test_no_start_end_token(self):
        input_data = {
            "prompts": ["the quick brown fox"] * 4,
            "images": [np.zeros([512, 512, 3])] * 4,
        }
        preprocessor = SigLIPPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[4, 9, 5, 7, 2, 2, 2, 2]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(x["images"], np.ones([4, 224, 224, 3]) * -1.0)

    def test_generate_preprocess(self):
        input_data = {
            "prompts": "the quick brown fox",
            "images": np.zeros([1, 512, 512, 3]),
        }
        preprocessor = SigLIPPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [4, 9, 5, 7, 2, 2, 2, 2])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertAllEqual(x["images"], np.ones([1, 224, 224, 3]) * -1.0)

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 4, 9, 5, 7, 2, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 1, 0, 0],
        }
        preprocessor = SigLIPPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "the quick brown fox")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in SigLIPPreprocessor.presets:
            self.run_preset_test(
                cls=SigLIPPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
