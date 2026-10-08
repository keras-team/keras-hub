import os

import pytest

from keras_hub.src.models.gemma4.gemma4_tokenizer import Gemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Gemma4TokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "proto": os.path.join(
                self.get_test_data_dir(), "gemma4_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Gemma4Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[9, 14, 10, 12], [9, 11, 13, 15]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            Gemma4Tokenizer(
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Gemma4Tokenizer.presets:
            self.run_preset_test(
                cls=Gemma4Tokenizer,
                preset=preset,
                input_data=self.input_data,
            )
