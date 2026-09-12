import os

import numpy as np
import pytest
import tensorflow as tf
from keras.src.saving import serialization_lib

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.word_piece_tokenizer import WordPieceTokenizer

try:
    import grain
except ImportError:
    grain = None
try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None


class WordPieceTokenizerTest(TestCase):
    def setUp(self):
        super().setUp()
        self._allow_python_workflow = True

    def make_tokenizer(self, **kwargs):
        return WordPieceTokenizer(
            _allow_python_workflow=self._allow_python_workflow, **kwargs
        )

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=WordPieceTokenizer,
            init_kwargs={
                "_allow_python_workflow": self._allow_python_workflow,
                "vocabulary": [
                    "[UNK]",
                    "the",
                    "qu",
                    "##ick",
                    "br",
                    "##own",
                    "fox",
                    ".",
                ],
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
                "_allow_python_workflow": self._allow_python_workflow,
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
        tokenizer = self.make_tokenizer(
            vocabulary=vocab_data, sequence_length=10
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 6, 7, 0, 0, 0]])

    def test_string_tokenize(self):
        input_data = ["the quick brown fox"]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = self.make_tokenizer(vocabulary=vocab_data, dtype="string")
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [["the", "qu", "##ick", "br", "##own", "fox"]],
        )

    def test_detokenize(self):
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = self.make_tokenizer(vocabulary=vocab_data)
        outputs = tokenizer.detokenize([1, 2, 3, 4, 5, 6])
        self.assertAllEqual(outputs, "the quick brown fox")
        outputs = tokenizer.detokenize([[1, 2, 3, 4, 5, 6], [1, 6]])
        self.assertAllEqual(outputs, ["the quick brown fox", "the fox"])

    def test_accessors(self):
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = self.make_tokenizer(vocabulary=vocab_data)
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
        tokenizer = self.make_tokenizer(vocabulary=vocab_data)
        with self.assertRaises(ValueError):
            tokenizer.id_to_token(tokenizer.vocabulary_size())
        with self.assertRaises(ValueError):
            tokenizer.id_to_token(-1)

    def test_special_tokens_string_dtype(self):
        input_data = ["quick brown whale @MASK@"]
        vocab_data = ["@UNK@", "qu", "@@ick", "br", "@@own", "fox", "@MASK@"]
        special_tokens = ["@UNK@", "@MASK@"]
        tokenizer = self.make_tokenizer(
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

        tokenizer = self.make_tokenizer(
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

        tokenizer = self.make_tokenizer(
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
        tokenizer = self.make_tokenizer(vocabulary=vocab_data, dtype="string")
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [["ah", "半", "推", "zz"]],
        )

    def test_lowercase(self):
        input_data = ["the QUicK brOWN FOX"]
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = self.make_tokenizer(vocabulary=vocab_data, lowercase=True)
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 6]])

    def test_skip_lowercase(self):
        input_data = ["the QUicK brOWN FOX"]
        vocab_data = ["[UNK]", "the", "QU", "##icK", "br", "##OWN", "fox"]
        tokenizer = self.make_tokenizer(vocabulary=vocab_data, lowercase=False)
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 0]])

    def test_strip_accents(self):
        input_data = ["á é í ó ú"]
        vocab_data = ["[UNK]", "a", "e", "i", "o", "u"]
        tokenizer = self.make_tokenizer(
            vocabulary=vocab_data, strip_accents=True
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5]])

    def test_skip_strip_accents(self):
        input_data = ["á é í ó ú"]
        vocab_data = ["[UNK]", "á", "é", "í", "ó", "ú"]
        tokenizer = self.make_tokenizer(
            vocabulary=vocab_data, strip_accents=False
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5]])

    def test_no_splitting(self):
        input_data = ["t o k e n", "m i s s i n g", "t o k e n"]
        vocab_data = ["[UNK]", "t o k e n"]
        tokenizer = self.make_tokenizer(vocabulary=vocab_data, split=False)
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [1, 0, 1])

    def test_word_piece_only(self):
        input_data = ["the", "quíck", "Brówn", "Fóx"]
        vocab_data = ["[UNK]", "the", "qu", "##íck", "Br", "##ówn", "Fóx"]
        tokenizer = self.make_tokenizer(
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
        tokenizer = self.make_tokenizer(vocabulary=vocab_path)
        call_output = tokenizer(input_data)
        self.assertAllEqual(call_output, [[1, 2, 3, 4, 5, 6, 7]])

    def test_no_oov_token_in_vocabulary(self):
        vocab_data = ["qu", "@@ick", "br", "@@OWN", "fox"]
        with self.assertRaises(ValueError):
            self.make_tokenizer(
                vocabulary=vocab_data,
            )

        vocab_data = ["@UNK@", "qu", "@@ick", "br", "@@OWN", "fox"]
        with self.assertRaises(ValueError):
            self.make_tokenizer(
                vocabulary=vocab_data,
            )

        vocab_data = ["UNK", "qu", "@@ick", "br", "@@OWN", "fox"]
        with self.assertRaises(ValueError):
            self.make_tokenizer(
                vocabulary=vocab_data,
            )

        with self.assertRaises(ValueError):
            self.make_tokenizer(vocabulary=vocab_data, oov_token=None)

    def test_no_splitting_with_special_tokens(self):
        # When `split` is `False`, no special tokens tokenization will be done.
        input_data = [
            "[MASK] t o k e n",
            "m i s s i n g",
            "[MASK]",
            "t o k e n",
        ]
        vocab_data = ["[UNK]", "[MASK]", "t o k e n"]
        tokenizer = self.make_tokenizer(
            vocabulary=vocab_data, split=False, special_tokens=["[MASK]"]
        )
        output = tokenizer(input_data)
        self.assertAllEqual(output, [0, 0, 1, 2])

    def test_safe_mode_vocabulary_file_disallowed(self):
        temp_dir = self.get_temp_dir()
        vocab_path = os.path.join(temp_dir, "vocab.txt")
        with open(vocab_path, "w") as file:
            file.write("[UNK]\nthe\nquick\nbrown\nfox\n")

        tokenizer = self.make_tokenizer()
        with serialization_lib.SafeModeScope(True):
            with self.assertRaisesRegex(
                ValueError,
                r"Requested the loading of a vocabulary file outside of the "
                r"model archive.*Vocabulary file: .*vocab\.txt",
            ):
                tokenizer.set_vocabulary(vocab_path)

    def test_split_false_ragged_inputs(self):
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = self.make_tokenizer(vocabulary=vocab_data, split=False)
        # A batch of pre-split words.
        input_data = tf.ragged.constant([["the", "quick"], ["Brown", "fox"]])
        self.assertAllEqual(tokenizer(input_data), [[1, 2, 3], [0, 6]])
        self.assertAllEqual(
            tokenizer([["the", "quick"], ["Brown", "fox"]]), [[1, 2, 3], [0, 6]]
        )
        if self._allow_python_workflow:
            # A single pre-split word. The TF path only supports batches of
            # words when `split=False`.
            self.assertAllEqual(tokenizer("quick"), [2, 3])

    def test_bytes_and_tensor_inputs(self):
        if not self._allow_python_workflow:
            self.skipTest("The TF path does not accept numpy string arrays.")
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = self.make_tokenizer(vocabulary=vocab_data)
        expected = [[1, 2, 3], [4, 5, 6]]
        self.assertAllEqual(tokenizer([b"the quick", b"brown fox"]), expected)
        self.assertAllEqual(
            tokenizer(np.array(["the quick", "brown fox"])), expected
        )
        self.assertAllEqual(
            tokenizer(tf.constant(["the quick", "brown fox"])), expected
        )
        self.assertAllEqual(tokenizer(b"the quick"), expected[0])
        self.assertAllEqual(tokenizer(tf.constant("the quick")), expected[0])

    def test_detokenize_inputs(self):
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        tokenizer = self.make_tokenizer(vocabulary=vocab_data)
        ids = [[1, 2, 3], [4, 5, 6]]
        expected = ["the quick", "brown fox"]
        self.assertAllEqual(tokenizer.detokenize(ids), expected)
        self.assertAllEqual(tokenizer.detokenize(np.array(ids)), expected)
        self.assertAllEqual(
            tokenizer.detokenize(tf.constant(ids, "int32")), expected
        )
        self.assertAllEqual(tokenizer.detokenize(ids[0]), expected[0])
        self.assertAllEqual(tokenizer.detokenize(np.array(ids[0])), expected[0])
        # Leading suffix tokens and bare suffix indicators are kept as is.
        vocab_data = ["[UNK]", "the", "##", "##ick", "a##b"]
        tokenizer = self.make_tokenizer(vocabulary=vocab_data)
        self.assertAllEqual(tokenizer.detokenize([3]), "##ick")
        self.assertAllEqual(tokenizer.detokenize([3, 3]), "##ickick")
        self.assertAllEqual(tokenizer.detokenize([1, 2]), "the ##")
        self.assertAllEqual(tokenizer.detokenize([1, 4, 3]), "the a##bick")
        self.assertAllEqual(tokenizer.detokenize([]), "")

    def test_long_words_map_to_oov(self):
        vocab_data = ["[UNK]", "a", "##a"]
        tokenizer = self.make_tokenizer(vocabulary=vocab_data)
        self.assertAllEqual(tokenizer("a" * 100), [1] + [2] * 99)
        self.assertAllEqual(tokenizer("a" * 101), [0])

    @pytest.mark.skipif(grain is None, reason="grain is not installed")
    def test_grain_outputs_numpy(self):
        vocab_data = ["[UNK]", "the", "qu", "##ick", "br", "##own", "fox"]
        input_data = ["the quick", "brown fox"]
        # Ragged outputs are python lists.
        tokenizer = self.make_tokenizer(vocabulary=vocab_data)
        (outputs,) = list(grain.MapDataset.source([input_data]).map(tokenizer))
        self.assertIsInstance(outputs, list)
        self.assertEqual(outputs, [[1, 2, 3], [4, 5, 6]])
        # Dense outputs are numpy arrays.
        tokenizer = self.make_tokenizer(
            vocabulary=vocab_data, sequence_length=4
        )
        outputs = list(grain.MapDataset.source(input_data).map(tokenizer))
        for output in outputs:
            self.assertIsInstance(output, np.ndarray)
        (batch,) = list(
            grain.MapDataset.source(input_data).map(tokenizer).batch(2)
        )
        self.assertAllEqual(batch, [[1, 2, 3, 0], [4, 5, 6, 0]])
        self.assertAllEqual(batch, tokenizer(input_data))

    @pytest.mark.skipif(tf_text is None, reason="tensorflow-text not installed")
    def test_python_path_matches_tf_path(self):
        if not self._allow_python_workflow:
            self.skipTest("Parity test only runs from the python path.")
        # NOTE: `tf_text.regex_split` truncates strings at "\x00" bytes, so
        # the paths are only compared on inputs without them.
        corpus = [
            "The quick brown fox.",
            "hi[MASK]there, you! [CLS]x",
            "Hello,World.  多个 test\u200bab",
            "á é í ó ú Ç",
            "don't stop-me now...",
            "a$b^c`d{e}~f",
            "  leading and trailing  ",
            "ﬁsh ǅ ß İ Σ",
            "x\xa0y\u3000z\ty\nq",
            "“quoted” — dash…",
            "[MASK][MASK]x[MASK]",
            "quickquick theq the##ick",
            "emoji 😀 test 한국어",
            "a" * 101,
            "",
        ]
        vocab_data = [
            "[PAD]", "[UNK]", "[CLS]", "[MASK]", "the", "qu", "##ick", "br",
            "##own", "fox", ".", ",", "!", "'", "-", "a", "b", "c", "d", "e",
            "x", "y", "z", "##h", "##ere", "t", "hi", "半", "推", "多", "个",
            "test", "you", "now", "don", "##t", "stop", "me", "quick",
            "##quick", "fi", "##sh", "ss", "i", "σ", "emoji", "한국어",
            "leading", "and", "trailing", "“", "”", "—", "…", "dash",
            "quoted", "$", "^", "`", "{", "}", "~", "##a",
        ]  # fmt: skip
        for kwargs in [
            {},
            {"lowercase": True},
            {"strip_accents": True},
            {"lowercase": True, "strip_accents": True, "split_on_cjk": False},
            {
                "lowercase": True,
                "special_tokens": ["[MASK]", "[CLS]"],
                "special_tokens_in_strings": True,
            },
            {"dtype": "string"},
            {"dtype": "string", "lowercase": True, "split_on_cjk": False},
            {"sequence_length": 8},
        ]:
            python_tokenizer = WordPieceTokenizer(
                vocabulary=vocab_data, **kwargs
            )
            tf_tokenizer = WordPieceTokenizer(
                vocabulary=vocab_data, _allow_python_workflow=False, **kwargs
            )
            python_output = python_tokenizer(corpus)
            tf_output = tf_tokenizer(corpus)
            self.assertAllEqual(python_output, tf_output)
            for text in corpus:
                self.assertAllEqual(python_tokenizer(text), tf_tokenizer(text))
            if kwargs.get("dtype", "int32") != "string":
                self.assertAllEqual(
                    python_tokenizer.detokenize(python_output),
                    tf_tokenizer.detokenize(tf_output),
                )
        # Pre-split inputs.
        words = ["the", "quick", "Fox", "ǅ", "theq", "", "hi"]
        ragged = tf.ragged.constant([["the", "quick"], ["Fox"], []])
        for kwargs in [{}, {"lowercase": True, "strip_accents": True}]:
            python_tokenizer = WordPieceTokenizer(
                vocabulary=vocab_data, split=False, **kwargs
            )
            tf_tokenizer = WordPieceTokenizer(
                vocabulary=vocab_data,
                split=False,
                _allow_python_workflow=False,
                **kwargs,
            )
            self.assertAllEqual(python_tokenizer(words), tf_tokenizer(words))
            self.assertAllEqual(python_tokenizer(ragged), tf_tokenizer(ragged))


class WordPieceTokenizerTFTest(WordPieceTokenizerTest):
    """Set `_allow_python_workflow=False` to test TF execution."""

    def setUp(self):
        super().setUp()
        self._allow_python_workflow = False
