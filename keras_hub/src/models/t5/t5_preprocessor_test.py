import os

import pytest

from keras_hub.src.models.t5.t5_preprocessor import T5Preprocessor
from keras_hub.src.models.t5.t5_tokenizer import T5Tokenizer
from keras_hub.src.tests.test_case import TestCase


class T5PreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = T5Tokenizer(
            proto=os.path.join(self.get_test_data_dir(), "t5_test_vocab.spm")
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 10,
        }
        self.input_data = ["the quick brown fox"]

    def test_preprocessor_basics(self):
        self.run_preprocessing_layer_test(
            cls=T5Preprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output={
                "token_ids": [[4, 9, 5, 7, 1, 0, 0, 0, 0, 0]],
                "padding_mask": [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]],
            },
        )

    def test_no_start_end_token(self):
        input_data = ["the quick brown fox"] * 4
        preprocessor = T5Preprocessor(
            tokenizer=self.tokenizer,
            sequence_length=8,
            add_start_token=False,
            add_end_token=False,
        )
        x = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[4, 9, 5, 7, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)

    def test_sequence_length_override(self):
        input_data = "the quick brown fox"
        preprocessor = T5Preprocessor(**self.init_kwargs)
        x = preprocessor(input_data, sequence_length=4)
        self.assertAllEqual(x["token_ids"], [4, 9, 5, 1])

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        self.skipTest("TODO")
        for preset in T5Preprocessor.presets:
            self.run_preset_test(
                cls=T5Preprocessor,
                preset=preset,
                input_data=self.input_data,
            )
