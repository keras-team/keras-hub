import os

from keras_hub.src.models.moonshine.moonshine_tokenizer import (
    MoonshineTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_llama_test_proto.py.
            "proto": os.path.join(
                self.get_test_data_dir(), "llama_test_vocab.spm"
            )
        }
        self.tokenizer = MoonshineTokenizer(**self.init_kwargs)
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=MoonshineTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_detokenization(self):
        for text in self.input_data:
            tokens = self.tokenizer(text)
            decoded = self.tokenizer.detokenize(tokens)
            self.assertIn(text.lower(), decoded.lower().strip())

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            MoonshineTokenizer(
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    def test_serialization(self):
        instance = MoonshineTokenizer(**self.init_kwargs)
        self.run_serialization_test(instance=instance)
