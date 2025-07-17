import pytest

from keras_hub.src.models.phi4.phi4_tokenizer import Phi4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Phi4TokenizerTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += [
            "<s>",
            "</s>",
            "<pad>",
            "<im_start>",
            "<im_sep>",
            "<im_end>",
        ]
        self.vocab += ["<fim_prefix>", "<fim_middle>", "<fim_suffix>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
            "sequence_length": None,
        }
        self.input_data = [
            "<s> airplane at airport</s><pad>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Phi4Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[6, 2, 3, 4, 2, 5, 7, 8], [2, 3, 2, 5]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            Phi4Tokenizer(vocabulary={"foo": 0, "bar": 1}, merges=["fo o"])

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=Phi4Tokenizer,
            preset="phi4_8b_en",
            input_data=["The quick brown fox."],
            expected_output=[[791, 4062, 14198, 39935, 13]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Phi4Tokenizer.presets:
            self.run_preset_test(
                cls=Phi4Tokenizer,
                preset=preset,
                input_data=self.input_data,
            )
