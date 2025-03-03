import os

from keras_hub.src.models.moonshine.moonshine_tokenizer import (
    MoonshineTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineTokenizerTest(TestCase):
    def setUp(self):
        self.tokenizer = MoonshineTokenizer(
            # Generated using create_moonshine_test_proto.py.
            os.path.join(self.get_test_data_dir(), "moonshine_test_vocab.spm")
        )
        self.input_data = [
            "Hello world!",
            "Test with <<ST_42>>",
            "Hex test <0x1F>",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=MoonshineTokenizer,
            init_kwargs={
                "proto": os.path.join(
                    self.get_test_data_dir(), "moonshine_test_vocab.spm"
                )
            },
            input_data=self.input_data,
        )

    def test_special_tokens(self):
        self.assertIsNotNone(self.tokenizer.start_token_id)
        self.assertIsNotNone(self.tokenizer.end_token_id)
        self.assertIsNotNone(self.tokenizer.unk_token_id)
        self.assertIsNotNone(self.tokenizer.pad_token_id)
        self.assertEqual(
            self.tokenizer.token_to_id("<s>"), self.tokenizer.start_token_id
        )
        self.assertEqual(
            self.tokenizer.token_to_id("</s>"), self.tokenizer.end_token_id
        )
        self.assertIsNotNone(self.tokenizer.token_to_id("<<ST_0>>"))
        self.assertIsNotNone(self.tokenizer.token_to_id("<0x00>"))
        self.assertIsNotNone(self.tokenizer.token_to_id("<>"))
