import pytest

from keras_hub.src.models.bart.bart_tokenizer import BartTokenizer
from keras_hub.src.tests.test_case import TestCase


class BartTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = ["<s>", "<pad>", "</s>", "air", "Ġair", "plane", "Ġat"]
        self.vocab += ["port", "<mask>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
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
            expected_output=[[0, 4, 5, 6, 4, 7, 2, 1], [4, 5, 4, 7]],
            expected_detokenize_output=[
                "<s> airplane at airport</s><pad>",
                " airplane airport",
            ],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            BartTokenizer(vocabulary=["a", "b", "c"], merges=[])

    @pytest.mark.large
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
