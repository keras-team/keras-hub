import pytest

from keras_hub.src.models.opt.opt_tokenizer import OPTTokenizer
from keras_hub.src.tests.test_case import TestCase


class OPTTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = ["<pad>", "</s>", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = [
            " airplane at airport</s>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=OPTTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[3, 4, 5, 3, 6, 1], [3, 4, 3, 6]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            OPTTokenizer(vocabulary=["a", "b", "c"], merges=[])

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=OPTTokenizer,
            preset="opt_125m_en",
            input_data=["The quick brown fox."],
            expected_output=[[133, 2119, 6219, 23602, 4]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in OPTTokenizer.presets:
            self.run_preset_test(
                cls=OPTTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
