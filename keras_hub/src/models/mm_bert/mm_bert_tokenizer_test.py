import os

import pytest

from keras_hub.src.models.mm_bert.mm_bert_tokenizer import MMBertTokenizer
from keras_hub.src.tests.test_case import TestCase


class MMBertTokenizerTest(TestCase):
    """Tests for the `MMBertTokenizer`."""

    def setUp(self):
        # A toy vocabulary shaped like the checkpoint one: `▁` prefixed
        # pieces, and a merge whose parts contain a newline.
        self.merges = [
            "▁ a",
            "▁ t",
            "▁ c",
            "a t",
            "▁c a",
            "▁ca t",
            "\n\n \n",
        ]
        vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab = sorted(set(vocab))
        vocab += ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>"]
        self.vocab = {token: index for index, token in enumerate(vocab)}

        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
            # The checkpoint declares runs of whitespace as added tokens.
            # They are matched in the raw text and never receive a `▁`.
            "unsplittable_tokens": ["\n\n"],
        }

        self.input_data = ["a cat", "a\n\ncat"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=MMBertTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[
                [8, 9, 4],
                [8, 1, 9, 4],
            ],
            # `▁` decodes to a space, so the `▁` of the first piece shows up
            # in the decoded text.
            expected_detokenize_output=[" a cat", " a\n\n cat"],
        )

    def test_added_token_gets_no_prefix(self):
        """An added token is one token, and no `▁` precedes it."""
        tokenizer = MMBertTokenizer(**self.init_kwargs)

        self.assertAllEqual(
            [int(i) for i in tokenizer(["\n\n"])[0]],
            [self.vocab["\n\n"]],
        )
        # The text after the added token starts a new piece, so it gets a
        # `▁`, exactly like the reference pipeline.
        self.assertAllEqual(
            [int(i) for i in tokenizer(["a\n\ncat"])[0]],
            [
                self.vocab["▁a"],
                self.vocab["\n\n"],
                self.vocab["▁c"],
                self.vocab["at"],
            ],
        )

    def test_merges_with_newlines_survive_a_save(self):
        """`merges.txt` cannot hold a newline, so mmBERT saves JSON."""
        tokenizer = MMBertTokenizer(**self.init_kwargs)
        tokenizer.save_assets(self.get_temp_dir())

        self.assertIn("merges.json", os.listdir(self.get_temp_dir()))

        # The added tokens live in the config, the merge table in the assets.
        restored = MMBertTokenizer(unsplittable_tokens=["\n\n"])
        restored.load_assets(self.get_temp_dir())

        self.assertEqual(restored.merges, tokenizer.merges)
        self.assertAllEqual(
            [int(i) for i in restored(["a\n\ncat"])[0]],
            [int(i) for i in tokenizer(["a\n\ncat"])[0]],
        )

    def test_errors_missing_special_tokens(self):
        vocabulary = {
            token: index
            for token, index in self.vocab.items()
            if token != "<mask>"
        }

        with self.assertRaisesRegex(ValueError, "<mask>"):
            MMBertTokenizer(vocabulary=vocabulary, merges=self.merges)

    def test_special_token_ids(self):
        tokenizer = MMBertTokenizer(**self.init_kwargs)

        self.assertEqual(tokenizer.start_token_id, tokenizer.bos_token_id)
        self.assertEqual(tokenizer.end_token_id, tokenizer.eos_token_id)
        self.assertEqual(tokenizer.bos_token_id, self.vocab["<bos>"])
        self.assertEqual(tokenizer.eos_token_id, self.vocab["<eos>"])
        self.assertEqual(tokenizer.pad_token_id, self.vocab["<pad>"])
        self.assertEqual(tokenizer.mask_token_id, self.vocab["<mask>"])
        self.assertEqual(tokenizer.unk_token_id, self.vocab["<unk>"])

    @pytest.mark.extra_large
    def test_tokenizer_matches_hf(self):
        """Body-token parity with the checkpoint's tokenizer.

        mmBERT ships no `tokenizer.model`, so `convert_modern_bert.py` builds
        this tokenizer from `tokenizer.json`, and the added-token flags come
        from `tokenizer_config.json`, where `<mask>` is declared with
        `lstrip=True`. The pre-tokenization of an added token is subtle enough
        (`<start_of_turn>`, runs of newlines, byte fallback) that it is worth
        comparing against the reference tokenizer.
        """
        from transformers import AutoTokenizer

        from keras_hub.src.utils.transformers import convert_modern_bert

        reference = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
        # `MMBertTokenizer.backbone_cls` is the mmBERT backbone, so the HF
        # route cannot use `from_preset`; the conversion script and this test
        # both go through the converter.
        tokenizer = convert_modern_bert.convert_tokenizer(
            MMBertTokenizer, "hf://jhu-clsp/mmBERT-base"
        )

        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertEqual(tokenizer.eos_token_id, 1)
        self.assertEqual(tokenizer.bos_token_id, 2)
        self.assertEqual(tokenizer.unk_token_id, 3)
        self.assertEqual(tokenizer.mask_token_id, 4)

        test_strings = [
            "",
            "The quick brown fox jumps over the lazy dog.",
            "  leading and trailing spaces  ",
            "a\nb\tc",
            "\n\n\n",
            "Mixed<mask>inline",
            "The capital of <mask> is Paris.",
            "a <mask> b",
            "a  <mask>  b",
            "\t<mask> x",
            "a\n\n<mask>",
            "x <bos> y",
            "a <mask> <mask> b",
            "<start_of_turn>user\nhi<end_of_turn>",
            "café naïve résumé",
            "你好，世界！",
            " ᐊᖏᔪᖅ",
        ]
        for text in test_strings:
            expected = reference(text, add_special_tokens=False)["input_ids"]
            self.assertAllEqual(
                [int(i) for i in tokenizer([text])[0]],
                expected,
            )
