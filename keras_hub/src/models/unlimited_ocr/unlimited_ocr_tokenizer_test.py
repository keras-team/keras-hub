from keras_hub.src.models.unlimited_ocr.unlimited_ocr_tokenizer import (
    OCR_MARKER_TOKENS,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_tokenizer import (
    UnlimitedOCRTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class UnlimitedOCRTokenizerTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"]
        self.vocab += ["<｜▁pad▁｜>", "<image>"]
        self.vocab += list(OCR_MARKER_TOKENS)
        self.vocab = sorted(set(self.vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = [
            "<｜begin▁of▁sentence｜>airplane at airport<｜end▁of▁sentence｜>",
            "<image><|grounding|> airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=UnlimitedOCRTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_special_token_ids(self):
        tokenizer = UnlimitedOCRTokenizer(**self.init_kwargs)
        self.assertEqual(
            tokenizer.start_token_id, self.vocab["<｜begin▁of▁sentence｜>"]
        )
        self.assertEqual(
            tokenizer.end_token_id, self.vocab["<｜end▁of▁sentence｜>"]
        )
        self.assertEqual(tokenizer.pad_token_id, self.vocab["<｜▁pad▁｜>"])
        self.assertEqual(tokenizer.image_token_id, self.vocab["<image>"])
        self.assertIn(tokenizer.image_token_id, tokenizer.special_token_ids)

    def test_markers_are_single_tokens_but_not_special(self):
        tokenizer = UnlimitedOCRTokenizer(**self.init_kwargs)
        for marker in OCR_MARKER_TOKENS:
            self.assertEqual(
                list(tokenizer(marker)), [self.vocab[marker]], msg=marker
            )
            self.assertNotIn(self.vocab[marker], tokenizer.special_token_ids)

    def test_image_token_is_one_token(self):
        tokenizer = UnlimitedOCRTokenizer(**self.init_kwargs)
        ids = list(tokenizer("<image><image>"))
        self.assertEqual(ids, [self.vocab["<image>"]] * 2)

    def test_missing_special_token_raises(self):
        vocab = {k: v for k, v in self.vocab.items() if k != "<image>"}
        with self.assertRaisesRegex(ValueError, "<image>"):
            UnlimitedOCRTokenizer(vocabulary=vocab, merges=self.merges)

    def test_pre_tokenizer_matches_checkpoint_split(self):
        tokenizer = UnlimitedOCRTokenizer(**self.init_kwargs)
        tokenizer._maybe_initialized_tokenizers()

        def split(text):
            pre = tokenizer._tokenizer.pre_tokenizer
            return [piece for piece, _ in pre.pre_tokenize_str(text)]

        # Digits group in threes.
        self.assertEqual(split("1234567"), ["123", "456", "7"])
        # A run of CJK characters is one piece.
        self.assertEqual(split("ab你好cd"), ["ab", "ä½łå¥½", "cd"])
        # Punctuation splits from words. The space joins the next word.
        self.assertEqual(split("Hi, you!"), ["Hi", ",", "Ġyou", "!"])
        # A punctuation mark glues to a following Latin word.
        self.assertEqual(split("a.b"), ["a", ".b"])
        # A line break is its own piece.
        self.assertEqual(split("a\nb"), ["a", "Ċ", "b"])

    def test_no_prefix_space_is_added(self):
        tokenizer = UnlimitedOCRTokenizer(**self.init_kwargs)
        self.assertFalse(tokenizer.add_prefix_space)
