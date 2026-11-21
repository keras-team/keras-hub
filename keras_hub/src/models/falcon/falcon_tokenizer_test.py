import pytest

from keras_hub.src.models.falcon.falcon_tokenizer import FalconTokenizer
from keras_hub.src.tests.test_case import TestCase


class FalconTokenizerTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|endoftext|>"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = [
            " airplane at airport<|endoftext|>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=FalconTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[
                [25, 16, 26, 25, 18, 1],
                [25, 16, 25, 18],
            ],
            expected_detokenize_output=[
                " airplane at airport<|endoftext|>",
                " airplane airport",
            ],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            FalconTokenizer(vocabulary=["a", "b", "c"], merges=[])

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=FalconTokenizer,
            preset="falcon_refinedweb_1b_en",
            input_data=["The quick brown fox."],
            expected_output=[[464, 2068, 7586, 21831, 13]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in FalconTokenizer.presets:
            self.run_preset_test(
                cls=FalconTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
