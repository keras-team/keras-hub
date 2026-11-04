import pytest

from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_tokenizer import (  # noqa: E501
    OpenAIPrivacyFilterTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class OpenAIPrivacyFilterTokenizerTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|endoftext|>"]
        self.vocab = sorted(set(self.vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=OpenAIPrivacyFilterTokenizer,
            init_kwargs={"vocabulary": self.vocab, "merges": self.merges},
            input_data=["airplane at airport"],
            expected_output=[[4, 16, 26, 25, 18]],
            expected_detokenize_output=["airplane at airport"],
        )

    def test_pad_token_comes_from_vocabulary(self):
        tokenizer = OpenAIPrivacyFilterTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.assertEqual(tokenizer.pad_token_id, self.vocab["<|endoftext|>"])
        self.assertIsNone(tokenizer.start_token_id)

    def test_errors_missing_special_tokens(self):
        vocab = dict(self.vocab)
        del vocab["<|endoftext|>"]
        with self.assertRaises(ValueError):
            OpenAIPrivacyFilterTokenizer(vocabulary=vocab, merges=self.merges)

    def test_python_workflow_uses_o200k_split(self):
        # The o200k pattern splits digit runs into groups of at most three
        # and keeps a run of spaces before a word separate from the word.
        # Neither holds for the GPT-2 split, and neither holds when the
        # pattern is not applied at all.
        tokenizer = OpenAIPrivacyFilterTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        pre_tokenizer = tokenizer._tokenizer.pre_tokenizer
        pieces = [p for p, _ in pre_tokenizer.pre_tokenize_str("4111  air")]
        self.assertEqual(pieces, ["411", "1", "Ġ", "Ġair"])

    def test_tf_workflow_uses_o200k_split(self):
        # `\p{N}{1,3}` splits "4111" into "411" + "1" before BPE, so with
        # these merge ranks it encodes as `4 11 | 1`; the GPT-2 split keeps
        # "4111" whole and BPE gives `4 111`. The TensorFlow path must agree
        # with the Python path, not with GPT-2.
        merges = self.merges + ["1 1", "4 1", "11 1"]
        vocab = dict(self.vocab)
        for piece in ["1", "11", "4", "41", "111"]:
            vocab.setdefault(piece, len(vocab))
        tokenizer = OpenAIPrivacyFilterTokenizer(
            vocabulary=vocab, merges=merges
        )
        expected = [vocab[p] for p in ["4", "11", "1", "Ġ", "Ġair"]]
        tokenizer._allow_python_workflow = True
        self.assertAllEqual(tokenizer("4111  air"), expected)
        tokenizer._allow_python_workflow = False
        self.assertAllEqual(tokenizer("4111  air"), expected)

    @pytest.mark.extra_large
    def test_hf_tokenizer_parity(self):
        try:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError:
            self.skipTest("Requires tokenizers and huggingface_hub")

        hf_repo = "yujiepan/openai-privacy-filter-tiny-random"
        tok_path = hf_hub_download(hf_repo, "tokenizer.json")
        hf_tok = Tokenizer.from_file(tok_path)

        kh_tok = OpenAIPrivacyFilterTokenizer.from_preset(f"hf://{hf_repo}")

        corpus = [
            "My name is John Smith and my email is john@example.com",
            "card 4111 1111 1111 1111",
            "Contact me at jane.doe+work@sub.example.co.uk today.",
            "See https://example.com/path?user=bob&id=12345 for details",
            "ftp://files.internal:8080/share",
            "Call +1 (555) 123-4567 or 555.987.6543",
            "SSN 123-45-6789, account 00012345678, routing 021000021",
            "Order #987654321 shipped 2024-03-15 at 14:32:07",
            "pi is 3.14159265358979 and e is 2.718281828",
            "1234567890123456789012345678901234567890",
            "Born 12/25/1990, moved on March 3rd, 2015.",
            "tabs\there\tand  double  spaces   triple",
            "line one\nline two\r\nline three",
            "   leading spaces and trailing   ",
            "a\n\n\nb",
            "I'm sure it's O'Brien's; don't you think?",
            "Mr. J. R. R. Tolkien, Ph.D.",
            "Zoë Müller lives at Straße 12, 80331 München",
            "名前は山田太郎です。電話は090-1234-5678",
            "emoji 😀 test 🚀🚀 end",
            "123 Main St., Apt 4B, Springfield, IL 62704",
            "IP 192.168.1.254 and MAC 00:1A:2B:3C:4D:5E",
            "IBAN DE89 3704 0044 0532 0130 00",
            "Password: hunter2!  Token: ghp_ABCdef123456",
            "HELLO World ÉCOLE École naïve CAFÉ",
            "",
            " ",
            "x",
            "<|endoftext|>",
        ]
        # RE2 cannot express `\s+(?!\S)`; the `६` emulation in
        # `split_strings_for_bpe` covers every case above except a trailing
        # run of spaces, which the GPT-2 TensorFlow path gets wrong the same
        # way. Pinned here so a fix shows up as a failure.
        tf_known_misses = ["   leading spaces and trailing   "]

        import numpy as np

        saved_flag = kh_tok._allow_python_workflow
        try:
            for python_workflow in (True, False):
                kh_tok._allow_python_workflow = python_workflow
                path = "python" if python_workflow else "tf"
                for text in corpus:
                    hf_ids = hf_tok.encode(text, add_special_tokens=False).ids
                    kh_ids = np.asarray(kh_tok(text)).tolist()
                    if not python_workflow and text in tf_known_misses:
                        self.assertNotEqual(
                            hf_ids, kh_ids, f"[{path}] now matches {text!r}"
                        )
                        continue
                    self.assertEqual(
                        hf_ids, kh_ids, f"[{path}] mismatch on {text!r}"
                    )
        finally:
            kh_tok._allow_python_workflow = saved_flag
