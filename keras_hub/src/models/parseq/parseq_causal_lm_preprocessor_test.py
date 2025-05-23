import numpy as np
import pytest

from keras_hub.src.models.parseq.parseq_causal_lm_preprocessor import (
    PARSeqCausalLMPreprocessor,
)
from keras_hub.src.models.parseq.parseq_image_converter import (
    PARSeqImageConverter,
)
from keras_hub.src.models.parseq.parseq_tokenizer import PARSeqTokenizer
from keras_hub.src.tests.test_case import TestCase


class PARSeqCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = PARSeqTokenizer()
        self.image_converter = PARSeqImageConverter(image_size=(32, 128))

        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 9,
        }
        self.input_data = {
            "images": [np.zeros([32, 128, 3])],
            "responses": ["Google"],
        }

    def test_preprocessor_basics(self):
        self.run_preprocessing_layer_test(
            cls=PARSeqCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[95, 43, 25, 25, 17, 22, 15, 0, 96]],
                    "padding_mask": [
                        [True, True, True, True, True, True, True, True, False]
                    ],
                    "images": np.zeros([1, 32, 128, 3]),
                },
                [[43, 25, 25, 17, 22, 15, 0, 96, 96]],  # Labels shifted.
                [[True, True, True, True, True, True, True, False, False]],
            ),
        )

    def test_no_start_end_token(self):
        input_data = {
            "responses": ["Google"] * 4,
            "images": [np.zeros([512, 512, 3])] * 4,
        }
        preprocessor = PARSeqCausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(
            x["token_ids"], [[43, 25, 25, 17, 22, 15, 96, 96, 96]] * 4
        )
        self.assertAllEqual(
            x["padding_mask"],
            [[True, True, True, True, True, True, False, False, False]] * 4,
        )
        self.assertAllEqual(x["images"], np.zeros([4, 32, 128, 3]))
        self.assertAllEqual(y, [[25, 25, 17, 22, 15, 96, 96, 96, 96]] * 4)
        self.assertAllEqual(
            sw, [[True, True, True, True, True, False, False, False, False]] * 4
        )

    def test_generate_preprocess(self):
        input_data = np.zeros([1, 32, 128, 3])
        preprocessor = PARSeqCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(
            x["token_ids"], [[95, 96, 96, 96, 96, 96, 96, 96, 96]]
        )
        self.assertAllEqual(
            x["padding_mask"],
            [[True, False, False, False, False, False, False, False, False]],
        )
        self.assertAllEqual(x["images"], np.zeros([1, 32, 128, 3]))

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [43, 25, 25, 17, 22, 15, 0, 96, 96],
            "padding_mask": [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
            ],
        }
        preprocessor = PARSeqCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, ["G", "o", "o", "g", "l", "e"])

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in PARSeqCausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=PARSeqCausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
