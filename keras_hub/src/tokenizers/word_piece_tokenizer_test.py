import os

import numpy as np
import tensorflow as tf
from keras.src.saving import serialization_lib

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.word_piece_tokenizer import WordPieceTokenizer


class WordPieceTokenizerTest(TestCase):
    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=WordPieceTokenizer,
            init_kwargs={
                "vocabulary": [
                    "[UNK]",
                    "the",
                    "qu",
                    "##ick",
                    "br",
                    "##own",
                    "fox",
                    ".",
                ]
            },
            input_data=["the quick brown fox."],
            expected_output=[[1, 2, 3, 4, 5, 6, 7]],
            expected_detokenize_output=["the quick brown fox ."],
        )

    def test_tokenizer_basics_with_non_default_config(self):
        special_tokens = ["@UNK@", "@MASK@"]
        vocab_data = ["@UNK@", "qu", "@@ick", "br", "@@own", "fox", "@MASK@"]
        self.run_preprocessing_layer_test(
            cls=WordPieceTokenizer,
            init_kwargs={
                "vocabulary": vocab_data,
                "lowercase": True,
                "oov_token": "@UNK@",
                "suffix_indicator": "@@",
                "special_tokens": special_tokens,
                "special_tokens_in_strings": True,
            },
            input_data=["quick brown whale @MASK@"],
            expected_output=[[1, 2, 3, 4, 0, 6]],
            expected_detokenize_output=["quick brown @UNK@ @MASK@"],
        )

    def test_dense_output(self):
        input_data = ["the quick brown fox."]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data, sequence_length=10
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 6, 7, 0, 0, 0]])

    def test_string_tokenize(self):
        input_data = ["the quick brown fox"]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data, dtype="string")
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [["the", "qu", "##ick", "br", "##own", "fox"]],
        )

    def test_detokenize(self):
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)
        outputs = tokenizer.detokenize([1, 2, 3, 4, 5, 6])
        self.assertAllEqual(outputs, "the quick brown fox")
        outputs = tokenizer.detokenize([[1, 2, 3, 4, 5, 6], [1, 6]])
        self.assertAllEqual(outputs, ["the quick brown fox", "the fox"])

    def test_accessors(self):
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)
        self.assertEqual(tokenizer.vocabulary_size(), 7)
        self.assertEqual(
            tokenizer.get_vocabulary(),
            ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"],
        )
        self.assertEqual(tokenizer.id_to_token(0), "[UNK]")
        self.assertEqual(tokenizer.id_to_token(6), "fox")
        self.assertEqual(tokenizer.token_to_id("[UNK]"), 0)
        self.assertEqual(tokenizer.token_to_id("fox"), 6)

    def test_error_id_out_of_vocabulary(self):
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)
        with self.assertRaises(ValueError):
            tokenizer.id_to_token(tokenizer.vocabulary_size())
        with self.assertRaises(ValueError):
            tokenizer.id_to_token(-1)

    def test_special_tokens_string_dtype(self):
        input_data = ["quick brown whale @MASK@"]
        vocab_data = ["@UNK@", "qu", "@@ick", "br", "@@own", "fox", "@MASK@"]
        special_tokens = ["@UNK@", "@MASK@"]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            oov_token="@UNK@",
            suffix_indicator="@@",
            dtype="string",
            special_tokens=special_tokens,
            special_tokens_in_strings=True,
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [["qu", "@@ick", "br", "@@own", "@UNK@", "@MASK@"]],
        )

    def test_special_tokens_int_dtype(self):
        input_data = ["[UNK] [MASK] [SEP] [PAD] [CLS] the quick brown fox."]
        special_tokens = ["[UNK]", "[MASK]", "[SEP]", "[PAD]", "[CLS]"]
        vocab_data = ["the", "qu", "##ick", "br", "##own", "fox", "."]
        vocab_data = [*special_tokens, *vocab_data]

        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            special_tokens=special_tokens,
            special_tokens_in_strings=True,
        )
        output = tokenizer(input_data)
        self.assertAllEqual(output, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

    def test_special_tokens_with_lowecase(self):
        input_data = ["[UNK] [MASK] [SEP] [PAD] [CLS] THE QUICK BROWN FOX."]
        special_tokens = ["[UNK]", "[MASK]", "[SEP]", "[PAD]", "[CLS]"]
        vocab_data = ["the", "qu", "##ick", "br", "##own", "fox", "."]
        vocab_data = [*special_tokens, *vocab_data]

        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            lowercase=True,
            special_tokens=special_tokens,
            special_tokens_in_strings=True,
        )
        output = tokenizer(input_data)
        self.assertAllEqual(output, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

    def test_cjk_tokens(self):
        input_data = ["ah半推zz"]
        vocab_data = ["[UNK]", "推", "敐", "乐", "半", "偷", "匕", "ah", "zz"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data, dtype="string")
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [["ah", "半", "推", "zz"]],
        )

    def test_lowercase(self):
        input_data = ["the QUicK brOWN FOX"]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data, lowercase=True)
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 6]])

    def test_skip_lowercase(self):
        input_data = ["the QUicK brOWN FOX"]
        vocab_data = ["[UNK]", "the", "QU", "##icK", "br", "##OWN", "fox"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data, lowercase=False)
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 0]])

    def test_strip_accents(self):
        input_data = ["á é í ó ú"]
        vocab_data = ["[UNK]", "a", "e", "i", "o", "u"]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data, strip_accents=True
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5]])

    def test_skip_strip_accents(self):
        input_data = ["á é í ó ú"]
        vocab_data = ["[UNK]", "á", "é", "í", "ó", "ú"]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data, strip_accents=False
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5]])

    def test_no_splitting(self):
        input_data = ["t o k e n", "m i s s i n g", "t o k e n"]
        vocab_data = ["[UNK]", "t o k e n"]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data, split=False)
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [1, 0, 1])

    def test_word_piece_only(self):
        input_data = ["the", "quíck", "Brówn", "Fóx"]
        vocab_data = ["[UNK]", "the", "qu", "##íck", "Br", "##ówn", "Fóx"]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            lowercase=False,
            strip_accents=False,
            split=False,
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [1, 2, 3, 4, 5, 6])

    def test_from_file(self):
        vocab_path = os.path.join(self.get_temp_dir(), "vocab.txt")
        input_data = ["the quick brown fox."]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox", "."]
        with tf.io.gfile.GFile(vocab_path, "w") as file:
            for piece in vocab_data:
                file.write(piece + "\n")
        tokenizer = WordPieceTokenizer(vocabulary=vocab_path)
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 6, 7]])

    def test_no_oov_token_in_vocabulary(self):
        vocab_data = ["qu", "@@ick", "br", "@@OWN", "fox"]
        with self.assertRaises(ValueError):
            WordPieceTokenizer(
                vocabulary=vocab_data,
            )

        vocab_data = ["@UNK@", "qu", "@@ick", "br", "@@OWN", "fox"]
        with self.assertRaises(ValueError):
            WordPieceTokenizer(
                vocabulary=vocab_data,
            )

        vocab_data = ["UNK", "qu", "@@ick", "br", "@@OWN", "fox"]
        with self.assertRaises(ValueError):
            WordPieceTokenizer(
                vocabulary=vocab_data,
            )

        with self.assertRaises(ValueError):
            WordPieceTokenizer(vocabulary=vocab_data, oov_token=None)

    def test_no_splitting_with_special_tokens(self):
        # When `split` is `False`, no special tokens tokenization will be done.
        input_data = [
            "[MASK] t o k e n",
            "m i s s i n g",
            "[MASK]",
            "t o k e n",
        ]
        vocab_data = ["[UNK]", "[MASK]", "t o k e n"]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data, split=False, special_tokens=["[MASK]"]
        )
        output = tokenizer(input_data)
        self.assertAllEqual(output, [0, 0, 1, 2])

    def test_python_workflow_is_default(self):
        vocab_data = [
            "[UNK]",
            "the",
            "qu",
            "##ick",
        ]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)

        self.assertTrue(tokenizer._allow_python_workflow)

        # Ensure the Python implementation is actually used.
        tokenizer._fast_word_piece = None

        output = tokenizer("the quick")
        self.assertAllEqual(output, [1, 2, 3])

    def test_python_workflow_does_not_require_tf_text(self):
        vocab_data = [
            "[UNK]",
            "the",
            "qu",
            "##ick",
        ]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            _allow_python_workflow=True,
        )
        tokenizer._fast_word_piece = None

        output = tokenizer("the quick")
        self.assertAllEqual(output, [1, 2, 3])

        output = tokenizer.detokenize([1, 2, 3])
        self.assertAllEqual(output, "the quick")

    def test_tf_workflow_can_be_explicitly_requested(self):
        vocab_data = [
            "[UNK]",
            "the",
            "qu",
            "##ick",
        ]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            _allow_python_workflow=False,
        )

        output = tokenizer("the quick")
        self.assertAllEqual(output, [1, 2, 3])

        output = tokenizer.detokenize([1, 2, 3])
        self.assertAllEqual(output, "the quick")

    def test_python_and_tf_workflows_match(self):
        vocab_data = [
            "[UNK]",
            "the",
            "qu",
            "##ick",
            "br",
            "##own",
            "fox",
            ".",
        ]
        input_data = ["The quick brown fox."]

        python_tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            lowercase=True,
            _allow_python_workflow=True,
        )
        tf_tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            lowercase=True,
            _allow_python_workflow=False,
        )

        python_output = python_tokenizer(input_data)
        tf_output = tf_tokenizer(input_data)

        self.assertAllEqual(python_output, tf_output)

        python_detokenized = python_tokenizer.detokenize(python_output)
        tf_detokenized = tf_tokenizer.detokenize(tf_output)

        self.assertAllEqual(python_detokenized, tf_detokenized)

    def test_python_workflow_with_numpy_inputs(self):
        vocab_data = [
            "[UNK]",
            "the",
            "qu",
            "##ick",
        ]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)

        # 0-D NumPy string.
        output = tokenizer(np.array("the quick"))
        self.assertAllEqual(output, [1, 2, 3])

        # 1-D NumPy array.
        output = tokenizer(np.array(["the quick"]))
        self.assertAllEqual(output, [[1, 2, 3]])

        # 1-D NumPy array with multiple strings.
        output = tokenizer(np.array(["the quick", "the"]))
        self.assertAllEqual(output, [[1, 2, 3], [1]])

    def test_python_workflow_rejects_2d_list_input(self):
        vocab_data = [
            "[UNK]",
            "the",
            "qu",
            "##ick",
        ]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)

        # 2-D Python lists are not supported when split=True.
        with self.assertRaisesRegex(
            ValueError,
            r"When `split=True`, `inputs` must be a string or a "
            r"1D list/tuple of strings.",
        ):
            tokenizer([["the", "quick"], ["the", "quick"]])

    def test_python_workflow_rejects_2d_numpy_input(self):
        vocab_data = [
            "[UNK]",
            "the",
            "qu",
            "##ick",
        ]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)

        # A 2-D NumPy array is also invalid when split=True.
        with self.assertRaisesRegex(
            ValueError,
            r"When `split=True`, `inputs` must be a string or a "
            r"1D list/tuple of strings.",
        ):
            tokenizer(
                np.array(
                    [
                        ["the", "quick"],
                        ["the", "quick"],
                    ]
                )
            )

    def test_python_workflow_with_numpy_scalar_inputs(self):
        vocab_data = [
            "[UNK]",
            "the",
            "qu",
            "##ick",
        ]
        tokenizer = WordPieceTokenizer(vocabulary=vocab_data)

        # NumPy scalar string.
        output = tokenizer(np.str_("the quick"))
        self.assertAllEqual(output, [1, 2, 3])

        # NumPy scalar bytes.
        output = tokenizer(np.bytes_("the quick"))
        self.assertAllEqual(output, [1, 2, 3])

    def test_python_workflow_with_seq_len_returns_dense_output(self):
        vocab_data = [
            "[UNK]",
            "the",
            "qu",
            "##ick",
            "br",
            "##own",
            "fox",
            ".",
        ]
        tokenizer = WordPieceTokenizer(
            vocabulary=vocab_data,
            sequence_length=10,
        )

        output = tokenizer(["The quick brown fox.", "The fox"])

        self.assertEqual(output.shape, (2, 10))
        self.assertAllEqual(
            output,
            [
                [0, 2, 3, 4, 5, 6, 7, 0, 0, 0],
                [0, 6, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        )

    def test_safe_mode_vocabulary_file_disallowed(self):
        temp_dir = self.get_temp_dir()
        vocab_path = os.path.join(temp_dir, "vocab.txt")
        with open(vocab_path, "w") as file:
            file.write("[UNK]\nthe\nquick\nbrown\nfox\n")

        tokenizer = WordPieceTokenizer()
        with serialization_lib.SafeModeScope(True):
            with self.assertRaisesRegex(
                ValueError,
                r"Requested the loading of a vocabulary file outside of the "
                r"model archive.*Vocabulary file: .*vocab\.txt",
            ):
                tokenizer.set_vocabulary(vocab_path)
