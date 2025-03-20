import os

from keras_hub.src.models.moonshine.moonshine_tokenizer import (
    MoonshineTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_moonshine_test_proto.py.
            "proto": os.path.join(
                self.get_test_data_dir(), "moonshine_test_vocab.spm"
            )
        }
        self.tokenizer = MoonshineTokenizer(**self.init_kwargs)
        self.input_data = ["the quick brown fox", "the earth is round"]
        self.special_token_inputs = [
            "Hello world!",
            "Test with <<ST_42>>",
            "Hex test <0x1F>",
            "Empty token test <>",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=MoonshineTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_special_tokens_existence(self):
        self.assertIsNotNone(self.tokenizer.start_token_id)
        self.assertIsNotNone(self.tokenizer.end_token_id)
        self.assertIsNotNone(self.tokenizer.pad_token_id)

        self.assertIsNotNone(self.tokenizer.unk_token_id)
        self.assertIsNotNone(self.tokenizer.token_to_id("<>"))

        self.assertIsNotNone(self.tokenizer.token_to_id("<<ST_0>>"))
        self.assertIsNotNone(self.tokenizer.token_to_id("<<ST_42>>"))
        self.assertIsNotNone(self.tokenizer.token_to_id("<<ST_100>>"))
        self.assertIsNotNone(self.tokenizer.token_to_id("<<ST_767>>"))

        self.assertIsNotNone(self.tokenizer.token_to_id("<0x00>"))
        self.assertIsNotNone(self.tokenizer.token_to_id("<0x1F>"))
        self.assertIsNotNone(self.tokenizer.token_to_id("<0xA0>"))
        self.assertIsNotNone(self.tokenizer.token_to_id("<0xFF>"))

    def test_special_token_ids_mapping(self):
        self.assertEqual(
            self.tokenizer.token_to_id("<s>"), self.tokenizer.start_token_id
        )
        self.assertEqual(
            self.tokenizer.token_to_id("</s>"), self.tokenizer.end_token_id
        )
        self.assertEqual(
            self.tokenizer.token_to_id("<pad>"), self.tokenizer.pad_token_id
        )
        self.assertEqual(
            self.tokenizer.token_to_id("<unk>"), self.tokenizer.unk_token_id
        )

    def test_special_tokens_tokenization(self):
        tokenized_st42 = self.tokenizer("<<ST_42>>")
        self.assertEqual(len(tokenized_st42), 1)

        tokenized_hex = self.tokenizer("<0x1F>")
        self.assertEqual(len(tokenized_hex), 1)

        tokenized_empty = self.tokenizer("<>")
        self.assertEqual(len(tokenized_empty), 1)

    def test_detokenization(self):
        for text in self.input_data + self.special_token_inputs:
            tokens = self.tokenizer(text)
            decoded = self.tokenizer.detokenize(tokens)
            if text in self.input_data:
                self.assertIn(text.lower(), decoded.lower())

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            MoonshineTokenizer(
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    def test_batch_tokenization(self):
        batch_tokens = self.tokenizer(self.input_data)
        self.assertEqual(len(batch_tokens), len(self.input_data))
