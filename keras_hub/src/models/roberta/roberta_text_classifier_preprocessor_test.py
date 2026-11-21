import pytest

from keras_hub.src.models.roberta.roberta_text_classifier_preprocessor import (
    RobertaTextClassifierPreprocessor,
)
from keras_hub.src.models.roberta.roberta_tokenizer import RobertaTokenizer
from keras_hub.src.tests.test_case import TestCase


class RobertaTextClassifierPreprocessorTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<s>", "<pad>", "</s>", "<mask>"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.tokenizer = RobertaTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (
            [" airplane at airport"],
            [1],  # Pass through labels.
            [1.0],  # Pass through sample_weights.
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=RobertaTextClassifierPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[3, 27, 18, 28, 27, 20, 0, 2]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 0]],
                },
                [1],  # Pass through labels.
                [1.0],  # Pass through sample_weights.
            ),
        )

    def test_errors_for_2d_list_input(self):
        preprocessor = RobertaTextClassifierPreprocessor(**self.init_kwargs)
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            preprocessor(ambiguous_input)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in RobertaTextClassifierPreprocessor.presets:
            self.run_preset_test(
                cls=RobertaTextClassifierPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
