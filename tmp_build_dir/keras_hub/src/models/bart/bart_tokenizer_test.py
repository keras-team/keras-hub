import pytest

from keras_hub.src.models.bart.bart_tokenizer import BartTokenizer
from keras_hub.src.tests.test_case import TestCase


class BartTokenizerTest(TestCase):
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
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = [
            "<s> airplane at airport</s><pad>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=BartTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[
                [3, 27, 18, 28, 27, 20, 0, 2],
                [27, 18, 27, 20],
            ],
            expected_detokenize_output=[
                "<s> airplane at airport</s><pad>",
                " airplane airport",
            ],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            BartTokenizer(vocabulary=["a", "b", "c"], merges=[])

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=BartTokenizer,
            preset="bart_base_en",
            input_data=["The quick brown fox."],
            expected_output=[[133, 2119, 6219, 23602, 4]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BartTokenizer.presets:
            self.run_preset_test(
                cls=BartTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
