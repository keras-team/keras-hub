import os

import tensorflow as tf
from keras.src.saving import serialization_lib

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


class SentencePieceTokenizerTest(TestCase):
    def setUp(self):
        super().setUp()
        self.proto = os.path.join(
            self.get_test_data_dir(), "tokenizer_test_vocab.spm"
        )

    def test_tokenize(self):
        input_data = ["the quick brown fox."]
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertAllEqual(call_output, [[6, 5, 3, 4]])
        self.assertAllEqual(tokenize_output, [[6, 5, 3, 4]])

    def test_scalar_tokenize(self):
        input_data = "the quick brown fox."
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertAllEqual(call_output, [6, 5, 3, 4])
        self.assertAllEqual(tokenize_output, [6, 5, 3, 4])

    def test_dense_output(self):
        input_data = ["the quick brown fox."]
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
            sequence_length=10,
        )
        output_data = tokenizer(input_data)
        self.assertAllEqual(output_data, [[6, 5, 3, 4, 0, 0, 0, 0, 0, 0]])

    def test_string_tokenize(self):
        input_data = ["the quick brown fox."]
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
            dtype="string",
        )
        output_data = tokenizer(input_data)
        self.assertAllEqual(
            output_data,
            [["▁the", "▁quick", "▁brown", "▁fox."]],
        )

    def test_scalar_bos_eos(self):
        input_data = "the quick brown fox."
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
            add_bos=True,
            add_eos=True,
        )
        output_data = tokenizer(input_data)
        self.assertAllEqual(output_data, [1, 6, 5, 3, 4, 2])

    def test_string_bos_eos(self):
        input_data = ["the quick brown fox."]
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
            dtype="string",
            add_bos=True,
            add_eos=True,
        )
        output_data = tokenizer(input_data)
        self.assertAllEqual(
            output_data, [["<s>", "▁the", "▁quick", "▁brown", "▁fox.", "</s>"]]
        )

    def test_detokenize(self):
        tokenizer = SentencePieceTokenizer(proto=self.proto)
        outputs = tokenizer.detokenize([6, 5, 3, 4])
        self.assertAllEqual(outputs, "the quick brown fox.")
        outputs = tokenizer.detokenize([[6, 5, 3, 4], [6, 4]])
        self.assertAllEqual(outputs, ["the quick brown fox.", "the fox."])

    def test_accessors(self):
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        self.assertEqual(
            tokenizer.get_vocabulary(),
            ["<unk>", "<s>", "</s>", "▁brown", "▁fox.", "▁quick", "▁the"],
        )
        self.assertEqual(type(tokenizer.get_vocabulary()), list)
        self.assertEqual(tokenizer.vocabulary_size(), 7)
        self.assertEqual(type(tokenizer.vocabulary_size()), int)
        self.assertEqual(tokenizer.id_to_token(0), "<unk>")
        self.assertEqual(tokenizer.id_to_token(5), "▁quick")
        self.assertEqual(type(tokenizer.id_to_token(0)), str)
        self.assertEqual(tokenizer.token_to_id("<unk>"), 0)
        self.assertEqual(tokenizer.token_to_id("▁quick"), 5)
        self.assertEqual(type(tokenizer.token_to_id("<unk>")), int)

    def test_error_id_out_of_vocabulary(self):
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        with self.assertRaises(ValueError):
            tokenizer.id_to_token(tokenizer.vocabulary_size())
        with self.assertRaises(ValueError):
            tokenizer.id_to_token(-1)

    def test_from_bytes(self):
        with tf.io.gfile.GFile(self.proto, "rb") as file:
            proto = file.read()
        tokenizer = SentencePieceTokenizer(
            proto=proto,
        )
        output_data = tokenizer(["the quick brown fox."])
        self.assertAllEqual(output_data, [[6, 5, 3, 4]])

    def test_tokenize_then_batch(self):
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )

        ds = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox.", "the quick", "the", "quick brown fox."]
        )
        ds = ds.map(tokenizer).apply(
            tf.data.experimental.dense_to_ragged_batch(4)
        )
        output_data = ds.take(1).get_single_element()

        expected = [
            [6, 5, 3, 4],
            [6, 5],
            [6],
            [5, 3, 4],
        ]
        for i in range(4):
            self.assertAllEqual(output_data[i], expected[i])

    def test_batch_then_tokenize(self):
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )

        ds = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox.", "the quick", "the", "quick brown fox."]
        )
        ds = ds.batch(4).map(tokenizer)
        output_data = ds.take(1).get_single_element()

        expected = [
            [6, 5, 3, 4],
            [6, 5],
            [6],
            [5, 3, 4],
        ]
        for i in range(4):
            self.assertAllEqual(output_data[i], expected[i])

    def test_config(self):
        input_data = ["the quick brown whale."]
        original_tokenizer = SentencePieceTokenizer(
            proto=self.proto,
        )
        cloned_tokenizer = SentencePieceTokenizer.from_config(
            original_tokenizer.get_config()
        )
        cloned_tokenizer.set_proto(original_tokenizer.proto)
        self.assertAllEqual(
            original_tokenizer(input_data),
            cloned_tokenizer(input_data),
        )

    def test_safe_mode_proto_file_disallowed(self):
        temp_dir = self.get_temp_dir()
        proto_path = os.path.join(temp_dir, "model.spm")
        with open(proto_path, "wb") as file:
            file.write(b"dummy proto data")

        tokenizer = SentencePieceTokenizer()
        with serialization_lib.SafeModeScope(True):
            with self.assertRaisesRegex(
                ValueError,
                r"Requested the loading of a proto file outside of the "
                r"model archive.*Proto file: .*model\.spm",
            ):
                tokenizer.set_proto(proto_path)
