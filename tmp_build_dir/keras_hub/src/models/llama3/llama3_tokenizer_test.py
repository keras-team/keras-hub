import pytest

from keras_hub.src.models.llama3.llama3_tokenizer import Llama3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Llama3TokenizerTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|end_of_text|>", "<|begin_of_text|>"]
        self.vocab += ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = [
            "<|begin_of_text|>airplane at airport<|end_of_text|>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Llama3Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[
                [1, 8, 20, 30, 29, 22, 3],
                [29, 20, 29, 22],
            ],
            expected_detokenize_output=[
                "<|begin_of_text|>airplane at airport<|end_of_text|>",
                " airplane airport",
            ],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            Llama3Tokenizer(vocabulary={"foo": 0, "bar": 1}, merges=["fo o"])

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=Llama3Tokenizer,
            preset="llama3_8b_en",
            input_data=["The quick brown fox."],
            expected_output=[[791, 4062, 14198, 39935, 13]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Llama3Tokenizer.presets:
            self.run_preset_test(
                cls=Llama3Tokenizer,
                preset=preset,
                input_data=self.input_data,
            )
