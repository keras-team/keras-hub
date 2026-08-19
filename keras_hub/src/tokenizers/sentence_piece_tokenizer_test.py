import os

from keras.src.saving import serialization_lib

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


class SentencePieceTokenizerTest(TestCase):
    def setUp(self):
        super().setUp()
        self._allow_python_workflow = True
        self.proto = os.path.join(
            self.get_test_data_dir(), "tokenizer_test_vocab.spm"
        )
        self.tokenizer = SentencePieceTokenizer(
            proto=self.proto,
            _allow_python_workflow=self._allow_python_workflow,
        )

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=SentencePieceTokenizer,
            init_kwargs={
                "proto": self.proto,
                "_allow_python_workflow": self._allow_python_workflow,
            },
            input_data=[
                "the quick brown fox.",
                "the quick",
                "the",
                "quick brown fox.",
            ],
            expected_output=[
                [6, 5, 3, 4],
                [6, 5],
                [6],
                [5, 3, 4],
            ],
        )

    def test_tokenize(self):
        input_data = ["the quick brown fox."]
        call_output = self.tokenizer(input_data)
        tokenize_output = self.tokenizer.tokenize(input_data)
        self.assertAllEqual(call_output, [[6, 5, 3, 4]])
        self.assertAllEqual(tokenize_output, [[6, 5, 3, 4]])

    def test_scalar_tokenize(self):
        input_data = "the quick brown fox."
        call_output = self.tokenizer(input_data)
        tokenize_output = self.tokenizer.tokenize(input_data)
        self.assertAllEqual(call_output, [6, 5, 3, 4])
        self.assertAllEqual(tokenize_output, [6, 5, 3, 4])

    def test_dense_output(self):
        input_data = ["the quick brown fox."]
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
            sequence_length=10,
            _allow_python_workflow=self._allow_python_workflow,
        )
        output_data = tokenizer(input_data)
        self.assertAllEqual(output_data, [[6, 5, 3, 4, 0, 0, 0, 0, 0, 0]])

    def test_string_tokenize(self):
        input_data = ["the quick brown fox."]
        tokenizer = SentencePieceTokenizer(
            proto=self.proto,
            dtype="string",
            _allow_python_workflow=self._allow_python_workflow,
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
            _allow_python_workflow=self._allow_python_workflow,
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
            _allow_python_workflow=self._allow_python_workflow,
        )
        output_data = tokenizer(input_data)
        self.assertAllEqual(
            output_data, [["<s>", "▁the", "▁quick", "▁brown", "▁fox.", "</s>"]]
        )

    def test_detokenize(self):
        outputs = self.tokenizer.detokenize([6, 5, 3, 4])
        self.assertAllEqual(outputs, "the quick brown fox.")
        outputs = self.tokenizer.detokenize([[6, 5, 3, 4], [6, 4]])
        self.assertAllEqual(outputs, ["the quick brown fox.", "the fox."])

    def test_detokenize_with_special_tokens(self):
        # We create a custom test tokenizer to test the special token chunking
        # fix.
        class CustomTokenizer(SentencePieceTokenizer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # "<s>" and "</s>" are in the tokenizer_test_vocab.spm
                # vocabulary.
                self._add_special_token("<s>", "start_token")
                self._add_special_token("</s>", "end_token")

        tokenizer = CustomTokenizer(proto=self.proto)

        # ID 1 = <s>, ID 2 = </s>, 6="the", 5="quick", 3="brown", 4="fox."
        input_data = [[1, 6, 5, 3, 4, 2]]

        output = tokenizer.detokenize(input_data)
        # Verify it doesn't strip spaces between special tokens and normal
        # tokens incorrectly.
        self.assertAllEqual(output, ["<s>the quick brown fox.</s>"])

    def test_accessors(self):
        self.assertEqual(
            self.tokenizer.get_vocabulary(),
            ["<unk>", "<s>", "</s>", "▁brown", "▁fox.", "▁quick", "▁the"],
        )
        self.assertEqual(type(self.tokenizer.get_vocabulary()), list)
        self.assertEqual(self.tokenizer.vocabulary_size(), 7)
        self.assertEqual(type(self.tokenizer.vocabulary_size()), int)
        self.assertEqual(self.tokenizer.id_to_token(0), "<unk>")
        self.assertEqual(self.tokenizer.id_to_token(5), "▁quick")
        self.assertEqual(type(self.tokenizer.id_to_token(0)), str)
        self.assertEqual(self.tokenizer.token_to_id("<unk>"), 0)
        self.assertEqual(self.tokenizer.token_to_id("▁quick"), 5)
        self.assertEqual(type(self.tokenizer.token_to_id("<unk>")), int)

    def test_error_id_out_of_vocabulary(self):
        with self.assertRaises(ValueError):
            self.tokenizer.id_to_token(self.tokenizer.vocabulary_size())
        with self.assertRaises(ValueError):
            self.tokenizer.id_to_token(-1)

    def test_from_bytes(self):
        with open(self.proto, "rb") as file:
            proto = file.read()
        tokenizer = SentencePieceTokenizer(
            proto=proto,
            _allow_python_workflow=self._allow_python_workflow,
        )
        output_data = tokenizer(["the quick brown fox."])
        self.assertAllEqual(output_data, [[6, 5, 3, 4]])

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


class SentencePieceTokenizerTFTest(SentencePieceTokenizerTest):
    """Set `_allow_python_workflow=False` to test TF execution."""

    def setUp(self):
        super().setUp()
        self._allow_python_workflow = False
        self.tokenizer = SentencePieceTokenizer(
            proto=self.proto,
            _allow_python_workflow=self._allow_python_workflow,
        )
