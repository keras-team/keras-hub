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
        # for runs of spaces, and a newline with a merge.
        extra = ["<|padding|>", "  ", "   ", "Ċ", "ĊĊ"]
        self.pythia_vocab = {
            **self.vocab,
            **{token: len(self.vocab) + i for i, token in enumerate(extra)},
        }
        self.pythia_kwargs = {
            "vocabulary": self.pythia_vocab,
            "merges": self.merges + ["Ċ Ċ"],
            "_allow_python_workflow": self._allow_python_workflow,
        }

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

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            GPTNeoXTokenizer(vocabulary=["a", "b", "c"], merges=[])

    def test_pad_token_id(self):
        tokenizer = GPTNeoXTokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.pad_token_id, 0)
        # Pythia vocabularies have a dedicated padding token.
        tokenizer = GPTNeoXTokenizer(**self.pythia_kwargs)
        self.assertEqual(
            tokenizer.pad_token_id, self.pythia_vocab["<|padding|>"]
        )
        tokenizer = GPTNeoXTokenizer(sequence_length=3, **self.pythia_kwargs)
        self.assert_tokens(
            tokenizer, " at", ["Ġat", "<|padding|>", "<|padding|>"]
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


class GPTNeoXTokenizerTFTest(GPTNeoXTokenizerTest):
    """Set `_allow_python_workflow=False` to test TF execution."""

    def setUp(self):
        self._allow_python_workflow = False
        self._set_up()
