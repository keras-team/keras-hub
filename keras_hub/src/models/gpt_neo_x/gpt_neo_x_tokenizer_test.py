from keras_hub.src.models.gpt_neo_x.gpt_neo_x_tokenizer import GPTNeoXTokenizer
from keras_hub.src.tests.test_case import TestCase


class GPTNeoXTokenizerTest(TestCase):
    def setUp(self):
        self._allow_python_workflow = True
        self._set_up()

    def _set_up(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|endoftext|>"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
            "_allow_python_workflow": self._allow_python_workflow,
        }
        self.input_data = [
            " airplane at airport<|endoftext|>",
            " airplane airport",
        ]
        # A vocabulary shaped like the Pythia one: a padding token, tokens
        # for runs of spaces, and newline and other whitespace with merges.
        extra = ["<|padding|>", "  ", "   ", "Ċ", "ĊĊ", "Ã", "©"]
        extra += ["ċ", "Ġċ", "Â", "ħ", "ĠÂ", "î", "Ģ", "ģ", "Ĥ", "c", "ĥ"]
        extra += ["<", "|", "d", "g", ">"]
        self.pythia_vocab = {
            **self.vocab,
            **{token: len(self.vocab) + i for i, token in enumerate(extra)},
        }
        self.pythia_kwargs = {
            "vocabulary": self.pythia_vocab,
            "merges": self.merges + ["Ċ Ċ", "Ġ ċ", "Ġ Â"],
            "_allow_python_workflow": self._allow_python_workflow,
        }
        # The same vocabulary without the run-of-spaces tokens.
        self.no_runs_kwargs = {
            **self.pythia_kwargs,
            "vocabulary": {
                token: id
                for token, id in self.pythia_vocab.items()
                if token.strip(" ") or len(token) < 2
            },
        }
        # The strings of `test_ue000_and_ue001` and their byte-level tokens.
        self.ue_texts = ["a\ue000b", "a\ue000b\ue001c", "a\ue001\ue002b"]
        self.ue_tokens = [
            ["a", "î", "Ģ", "Ģ", "b"],
            ["a", "î", "Ģ", "Ģ", "b", "î", "Ģ", "ģ", "c"],
            ["a", "î", "Ģ", "ģ", "î", "Ģ", "Ĥ", "b"],
        ]

    def assert_tokens(self, tokenizer, text, tokens):
        ids = [self.pythia_vocab[token] for token in tokens]
        self.assertAllEqual(tokenizer(text), ids)

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=GPTNeoXTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[
                [25, 16, 26, 25, 18, 1],
                [25, 16, 25, 18],
            ],
            expected_detokenize_output=[
                " airplane at airport<|endoftext|>",
                " airplane airport",
            ],
        )
        tokens = [
            [
                "Ġair",
                "plane",
                "   ",
                "a",
                "t",
                "  ",
                "air",
                "port",
                "<|padding|>",
            ],
            ["Ġat", "Ċ", "Ċ", "a", "t"],
        ]
        ids = [[self.pythia_vocab[t] for t in row] for row in tokens]
        self.run_preprocessing_layer_test(
            cls=GPTNeoXTokenizer,
            init_kwargs=self.pythia_kwargs,
            input_data=[" airplane   at  airport<|padding|>", " at\n\nat"],
            expected_output=ids,
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            GPTNeoXTokenizer(vocabulary=["a", "b", "c"], merges=[])

    def test_pad_token_id(self):
        tokenizer = GPTNeoXTokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertIsNone(tokenizer.pad_token)
        # Pythia vocabularies have a dedicated padding token.
        tokenizer = GPTNeoXTokenizer(**self.pythia_kwargs)
        self.assertEqual(
            tokenizer.pad_token_id, self.pythia_vocab["<|padding|>"]
        )
        self.assertEqual(tokenizer.pad_token, "<|padding|>")
        self.assertIn(tokenizer.pad_token_id, tokenizer.special_token_ids)
        tokenizer = GPTNeoXTokenizer(sequence_length=3, **self.pythia_kwargs)
        self.assert_tokens(
            tokenizer, " at", ["Ġat", "<|padding|>", "<|padding|>"]
        )

    def test_vocabulary_swap(self):
        # The padding token follows the current vocabulary.
        tokenizer = GPTNeoXTokenizer(**self.pythia_kwargs)
        tokenizer.set_vocabulary_and_merges(self.vocab, self.merges)
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertIsNone(tokenizer.pad_token)
        self.assertNotIn("<|padding|>", tokenizer.special_tokens)
        # So does the derived unsplittable token: the text now splits the
        # way it does for a tokenizer built on this vocabulary.
        self.assertNotIn("<|padding|>", tokenizer.unsplittable_tokens)
        fresh = GPTNeoXTokenizer(**self.init_kwargs)
        self.assertAllEqual(tokenizer("a<|padding|>b"), fresh("a<|padding|>b"))
        tokenizer.set_vocabulary_and_merges(
            self.pythia_vocab, self.pythia_kwargs["merges"]
        )
        self.assertEqual(
            tokenizer.pad_token_id, self.pythia_vocab["<|padding|>"]
        )
        self.assertEqual(tokenizer._special_token_attrs.count("pad_token"), 1)
        self.assertIn("<|padding|>", tokenizer.unsplittable_tokens)
        self.assert_tokens(
            tokenizer, " at<|padding|>!", ["Ġat", "<|padding|>", "!"]
        )
        tokenizer.set_vocabulary_and_merges(
            self.pythia_vocab, self.pythia_kwargs["merges"]
        )
        self.assertEqual(tokenizer._special_token_attrs.count("pad_token"), 1)

    def test_explicit_unsplittable_tokens(self):
        tokenizer = GPTNeoXTokenizer(
            unsplittable_tokens=[], **self.pythia_kwargs
        )
        # "<|padding|>" is ordinary text: "<|", "padding" and "|>!" are the
        # GPT-2 words, and none of their byte pairs is merged.
        self.assert_tokens(
            tokenizer,
            " at<|padding|>!",
            ["Ġat", "<", "|", "p", "a", "d", "d", "i", "n", "g", "|", ">", "!"],
        )
        self.assertEqual(tokenizer.get_config()["unsplittable_tokens"], [])
        explicit = ["<|endoftext|>", "<|padding|>"]
        tokenizer = GPTNeoXTokenizer(
            unsplittable_tokens=explicit, **self.pythia_kwargs
        )
        self.assert_tokens(
            tokenizer, " at<|padding|>!", ["Ġat", "<|padding|>", "!"]
        )
        self.assertEqual(
            tokenizer.get_config()["unsplittable_tokens"], explicit
        )

    def test_config_round_trip_without_vocabulary(self):
        tokenizer = GPTNeoXTokenizer()
        config = tokenizer.get_config()
        self.assertIsNone(config["unsplittable_tokens"])
        pythia_config = GPTNeoXTokenizer(**self.pythia_kwargs).get_config()
        self.assertIsNone(pythia_config["unsplittable_tokens"])
        # A clone made before the vocabulary is known keeps "<|padding|>"
        # whole once it gets a vocabulary that has it.
        clone = GPTNeoXTokenizer.from_config(
            {**config, "_allow_python_workflow": self._allow_python_workflow}
        )
        clone.set_vocabulary_and_merges(
            self.pythia_vocab, self.pythia_kwargs["merges"]
        )
        self.assert_tokens(
            clone, " at<|padding|>!", ["Ġat", "<|padding|>", "!"]
        )

    def test_whitespace_runs(self):
        tokenizer = GPTNeoXTokenizer(**self.pythia_kwargs)
        # A run of spaces is one token, and the text after it starts fresh.
        self.assert_tokens(
            tokenizer,
            " airplane   at  airport",
            ["Ġair", "plane", "   ", "a", "t", "  ", "air", "port"],
        )
        # The longest run token is taken first.
        self.assert_tokens(tokenizer, "     at", ["   ", "  ", "a", "t"])
        # A space left over stays with the word that follows.
        self.assert_tokens(tokenizer, "    at", ["   ", "Ġat"])
        self.assert_tokens(tokenizer, " at  ", ["Ġat", "  "])
        text = " airplane   at  airport"
        output = tokenizer.detokenize(tokenizer(text))
        if hasattr(output, "numpy"):
            output = output.numpy().decode("utf-8")
        self.assertEqual(output, text)

    def test_padding_token_in_text(self):
        tokenizer = GPTNeoXTokenizer(**self.pythia_kwargs)
        self.assert_tokens(
            tokenizer, " at<|padding|>!", ["Ġat", "<|padding|>", "!"]
        )

    def test_newlines_before_text(self):
        tokenizer = GPTNeoXTokenizer(**self.pythia_kwargs)
        # GPT-2 splits the last newline off a run that is followed by text.
        self.assert_tokens(tokenizer, " at\n\nat", ["Ġat", "Ċ", "Ċ", "a", "t"])
        self.assert_tokens(
            tokenizer, " at\n\n\nat", ["Ġat", "ĊĊ", "Ċ", "a", "t"]
        )
        self.assert_tokens(tokenizer, " at\n\n", ["Ġat", "ĊĊ"])
        self.assert_tokens(
            tokenizer, " at\n\n<|endoftext|>", ["Ġat", "ĊĊ", "<|endoftext|>"]
        )

    def test_unicode_whitespace(self):
        tokenizer = GPTNeoXTokenizer(**self.pythia_kwargs)
        # Whitespace other than ASCII also stands apart from a space before
        # it. "ċ" is the vertical tab, and "Â", "ħ" the bytes of U+0085.
        self.assert_tokens(tokenizer, " at \x0bat", ["Ġat", "Ġ", "ċ", "a", "t"])
        self.assert_tokens(
            tokenizer, " at \x85at", ["Ġat", "Ġ", "Â", "ħ", "a", "t"]
        )

    def test_nfc_normalization(self):
        tokenizer = GPTNeoXTokenizer(**self.pythia_kwargs)
        # "e" and a combining acute accent become "é", bytes "Ã" and "©".
        self.assert_tokens(tokenizer, "e\u0301", ["Ã", "©"])

    def test_add_prefix_space(self):
        tokenizer = GPTNeoXTokenizer(
            add_prefix_space=True, **self.pythia_kwargs
        )
        # Each piece of text gets the prefix, also after a whitespace run.
        self.assert_tokens(tokenizer, "at  at", ["Ġat", "  ", "Ġat"])
        self.assert_tokens(tokenizer, " at", ["Ġat"])

    def test_ue000_and_ue001(self):
        tokenizer = GPTNeoXTokenizer(**self.pythia_kwargs)
        self.assert_tokens(tokenizer, "a\ue000b", ["a", "î", "Ģ", "Ģ", "b"])
        text = "a\ue000b\ue001c"
        self.assert_tokens(
            tokenizer, text, ["a", "î", "Ģ", "Ģ", "b", "î", "Ģ", "ģ", "c"]
        )
        output = tokenizer.detokenize(tokenizer(text))
        if hasattr(output, "numpy"):
            output = output.numpy().decode("utf-8")
        self.assertEqual(output, text)
        # An escaped U+E001 directly followed by U+E002 must not be read as
        # U+E001 then an escaped U+E000.
        self.assert_tokens(
            tokenizer,
            "a\ue001\ue002b",
            ["a", "î", "Ģ", "ģ", "î", "Ģ", "Ĥ", "b"],
        )
        # U+E002 and U+E003 are the other escape characters; they and the
        # characters they stand next to also come through unchanged.
        self.assert_tokens(tokenizer, "a\ue002b", ["a", "î", "Ģ", "Ĥ", "b"])
        self.assert_tokens(tokenizer, "a\ue003b", ["a", "î", "Ģ", "ĥ", "b"])
        self.assert_tokens(
            tokenizer,
            "a\ue001\ue003b",
            ["a", "î", "Ģ", "ģ", "î", "Ģ", "ĥ", "b"],
        )
        self.assert_tokens(
            tokenizer,
            "a\ue000\ue003b",
            ["a", "î", "Ģ", "Ģ", "î", "Ģ", "ĥ", "b"],
        )

    def test_ue000_and_ue001_batch_and_string(self):
        ids = [[self.pythia_vocab[t] for t in row] for row in self.ue_tokens]
        tokenizer = GPTNeoXTokenizer(**self.pythia_kwargs)
        self.assertAllEqual(tokenizer(self.ue_texts), ids)
        tokenizer = GPTNeoXTokenizer(dtype="string", **self.pythia_kwargs)
        self.assertAllEqual(tokenizer(self.ue_texts), self.ue_tokens)

    def test_ue000_and_ue001_without_run_tokens(self):
        ids = [[self.pythia_vocab[t] for t in row] for row in self.ue_tokens]
        # With no run tokens and no unsplittable tokens there is nothing to
        # match before the byte-pair split, and U+E000 must still be text.
        no_unsplittable_kwargs = {
            **self.no_runs_kwargs,
            "unsplittable_tokens": [],
        }
        for kwargs in (self.no_runs_kwargs, no_unsplittable_kwargs):
            tokenizer = GPTNeoXTokenizer(**kwargs)
            for text, expected in zip(self.ue_texts, ids):
                self.assertAllEqual(tokenizer(text), expected)
            self.assertAllEqual(tokenizer(self.ue_texts), ids)
            tokenizer = GPTNeoXTokenizer(dtype="string", **kwargs)
            self.assertAllEqual(tokenizer(self.ue_texts), self.ue_tokens)
        self.run_preprocessing_layer_test(
            cls=GPTNeoXTokenizer,
            init_kwargs=no_unsplittable_kwargs,
            input_data=self.ue_texts,
            expected_output=ids,
        )

    def test_rank_2_input_raises(self):
        tokenizer = GPTNeoXTokenizer(**self.pythia_kwargs)
        with self.assertRaises(ValueError):
            tokenizer([[" at"]])


class GPTNeoXTokenizerTFTest(GPTNeoXTokenizerTest):
    """Set `_allow_python_workflow=False` to test TF execution."""

    def setUp(self):
        self._allow_python_workflow = False
        self._set_up()
