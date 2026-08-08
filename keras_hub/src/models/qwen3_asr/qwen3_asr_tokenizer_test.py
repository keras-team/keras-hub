from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import Qwen3ASRTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRTokenizerTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<|im_end|>", "<|endoftext|>", "!", "<|AUDIO|>"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = ["airplane at airport"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Qwen3ASRTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=None,
            expected_detokenize_output=[
                "airplane at airport",
            ],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            Qwen3ASRTokenizer(vocabulary={"foo": 0, "bar": 1}, merges=["fo o"])
