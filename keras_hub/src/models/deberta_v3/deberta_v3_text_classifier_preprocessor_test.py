import os

import pytest

from keras_hub.src.models.deberta_v3.deberta_v3_text_classifier_preprocessor import (
    DebertaV3TextClassifierPreprocessor,
)
from keras_hub.src.models.deberta_v3.deberta_v3_tokenizer import (
    DebertaV3Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class DebertaV3TextClassifierPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = DebertaV3Tokenizer(
            # Generated using create_deberta_v3_test_proto.py
            proto=os.path.join(
                self.get_test_data_dir(), "deberta_v3_test_vocab.spm"
            )
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (
            ["the quick brown fox"],
            [1],  # Pass through labels.
            [1.0],  # Pass through sample_weights.
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=DebertaV3TextClassifierPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 5, 10, 6, 8, 2, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [1],  # Pass through labels.
                [1.0],  # Pass through sample_weights.
            ),
        )

    def test_errors_for_2d_list_input(self):
        preprocessor = DebertaV3TextClassifierPreprocessor(**self.init_kwargs)
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            preprocessor(ambiguous_input)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DebertaV3TextClassifierPreprocessor.presets:
            self.run_preset_test(
                cls=DebertaV3TextClassifierPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
