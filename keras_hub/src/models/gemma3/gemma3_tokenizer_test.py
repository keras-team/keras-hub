import os

import pytest

from keras_hub.src.models.gemma3.gemma3_tokenizer import Gemma3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Gemma3TokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_gemma3_test_proto.py
            # TODO: Uploaded the actual Gemma3 spm file, which is the wrong
            # thing to do. Need to figure out how to add `<start_of_image>`,
            # etc., tokens to the tokeniser during SPM training.
            "proto": os.path.join(
                self.get_test_data_dir(), "gemma3_test_vocab.spm"
            )
        }
        # TODO: Figure out why <start_of_image>, etc. tokens are not getting
        # detokenized properly. I guess special tokens are not meant to be
        # decoded.
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Gemma3Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[9, 14, 10, 12], [9, 11, 13, 15]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            Gemma3Tokenizer(
                # Generated using create_no_special_token_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    @pytest.mark.kaggle_key_required
    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=Gemma3Tokenizer,
            preset="gemma3_instruct_1b",
            input_data=["The quick brown fox"],
            expected_output=[[818, 3823, 8864, 37423]],
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Gemma3Tokenizer.presets:
            self.run_preset_test(
                cls=Gemma3Tokenizer,
                preset=preset,
                input_data=self.input_data,
            )
