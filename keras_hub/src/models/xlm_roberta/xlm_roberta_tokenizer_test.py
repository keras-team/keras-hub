import os

import pytest

from keras_hub.src.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class XLMRobertaTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_xlm_roberta_test_proto.py
            "proto": os.path.join(
                self.get_test_data_dir(), "xlm_roberta_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=XLMRobertaTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[6, 11, 7, 9], [6, 8, 10, 12]],
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=XLMRobertaTokenizer,
            preset="xlm_roberta_base_multi",
            input_data=["The quick brown fox."],
            expected_output=[[581, 63773, 119455, 6, 147797, 5]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in XLMRobertaTokenizer.presets:
            self.run_preset_test(
                cls=XLMRobertaTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
