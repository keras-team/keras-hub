import os

from keras_hub.src.models.gemma3n.gemma3n_tokenizer import Gemma3nTokenizer
from keras_hub.src.tests.test_case import TestCase


class Gemma3nTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using `create_gemma3n_test_proto.py`.
            "proto": os.path.join(
                self.get_test_data_dir(), "gemma3n_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Gemma3nTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[14, 19, 15, 17], [14, 16, 18, 20]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            Gemma3nTokenizer(
                # Generated using `create_no_special_token_proto.py`
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )
