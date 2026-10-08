import pytest

from keras_hub.src.models.roberta.roberta_tokenizer import RobertaTokenizer
from keras_hub.src.tests.test_case import TestCase


class RobertaTokenizerTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab += ["<s>", "<pad>", "</s>", "<mask>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = [
            "<s> airplane at airport</s><pad>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=RobertaTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[
                [29, 23, 14, 24, 23, 16, 31, 30],
                [23, 14, 23, 16],
            ],
            expected_detokenize_output=[
                "<s> airplane at airport</s><pad>",
                " airplane airport",
            ],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            RobertaTokenizer(vocabulary=["a", "b", "c"], merges=[])

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=RobertaTokenizer,
            preset="roberta_base_en",
            input_data=["The quick brown fox."],
            expected_output=[[133, 2119, 6219, 23602, 4]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in RobertaTokenizer.presets:
            self.run_preset_test(
                cls=RobertaTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
