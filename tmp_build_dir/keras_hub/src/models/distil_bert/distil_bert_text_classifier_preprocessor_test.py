import pytest

from keras_hub.src.models.distil_bert.distil_bert_text_classifier_preprocessor import (  # noqa: E501
    DistilBertTextClassifierPreprocessor,
)
from keras_hub.src.models.distil_bert.distil_bert_tokenizer import (
    DistilBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class DistilBertTextClassifierPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["THE", "QUICK", "BROWN", "FOX"]
        self.vocab += ["the", "quick", "brown", "fox"]
        self.tokenizer = DistilBertTokenizer(vocabulary=self.vocab)
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (
            ["THE QUICK BROWN FOX."],
            [1],  # Pass through labels.
            [1.0],  # Pass through sample_weights.
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=DistilBertTextClassifierPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[2, 5, 6, 7, 8, 1, 3, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 0]],
                },
                [1],  # Pass through labels.
                [1.0],  # Pass through sample_weights.
            ),
        )

    def test_errors_for_2d_list_input(self):
        preprocessor = DistilBertTextClassifierPreprocessor(**self.init_kwargs)
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            preprocessor(ambiguous_input)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DistilBertTextClassifierPreprocessor.presets:
            self.run_preset_test(
                cls=DistilBertTextClassifierPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
