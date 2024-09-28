import os

import pytest

from keras_hub.src.models.t5.t5_tokenizer import T5Tokenizer
from keras_hub.src.tests.test_case import TestCase


class T5TokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_t5_test_proto.py
            "proto": os.path.join(self.get_test_data_dir(), "t5_test_vocab.spm")
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=T5Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[4, 9, 5, 7], [4, 6, 8, 10]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            T5Tokenizer(
                # Generated using create_no_special_token_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    @pytest.mark.large
    def test_smallest_preset(self):
        for preset in T5Tokenizer.presets:
            self.run_preset_test(
                cls=T5Tokenizer,
                preset=preset,
                input_data=["The quick brown fox."],
                expected_output=[[37, 1704, 4216, 3, 20400, 5]],
            )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in T5Tokenizer.presets:
            self.run_preset_test(
                cls=T5Tokenizer,
                preset=preset,
                input_data=self.input_data,
            )
