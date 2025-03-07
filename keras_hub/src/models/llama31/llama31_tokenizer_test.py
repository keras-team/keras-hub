import pytest

from keras_hub.src.models.llama31.llama31_tokenizer import Llama31Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Llama31TokenizerTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += ["<|end_of_text|>", "<|begin_of_text|>"]
        self.vocab += ["<|start_header_id|>", "<|end_header_id|>"]
        self.vocab += ["<|eot_id|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = [
            "<|begin_of_text|>airplane at airport<|end_of_text|>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Llama31Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[7, 1, 3, 4, 2, 5, 6], [2, 3, 2, 5]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            Llama31Tokenizer(vocabulary={"foo": 0, "bar": 1}, merges=["fo o"])

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=Llama31Tokenizer,
            preset="llama3_8b_en",
            input_data=["The quick brown fox."],
            expected_output=[[791, 4062, 14198, 39935, 13]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Llama31Tokenizer.presets:
            self.run_preset_test(
                cls=Llama31Tokenizer,
                preset=preset,
                input_data=self.input_data,
            )
