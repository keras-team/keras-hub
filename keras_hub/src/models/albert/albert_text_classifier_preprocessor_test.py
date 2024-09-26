import os

import pytest

from keras_hub.src.models.albert.albert_text_classifier_preprocessor import (
    AlbertTextClassifierPreprocessor,
)
from keras_hub.src.models.albert.albert_tokenizer import AlbertTokenizer
from keras_hub.src.tests.test_case import TestCase


class AlbertTextClassifierPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = AlbertTokenizer(
            # Generated using create_albert_test_proto.py
            proto=os.path.join(
                self.get_test_data_dir(), "albert_test_vocab.spm"
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
            cls=AlbertTextClassifierPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[2, 5, 10, 6, 8, 3, 0, 0]],
                    "segment_ids": [[0, 0, 0, 0, 0, 0, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [1],  # Pass through labels.
                [1.0],  # Pass through sample_weights.
            ),
        )

    def test_errors_for_2d_list_input(self):
        preprocessor = AlbertTextClassifierPreprocessor(**self.init_kwargs)
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            preprocessor(ambiguous_input)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in AlbertTextClassifierPreprocessor.presets:
            self.run_preset_test(
                cls=AlbertTextClassifierPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
