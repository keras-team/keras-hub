import pytest

from keras_hub.src.models.modernbert.modern_bert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertTokenizerTest(TestCase):
    """
    Tests for verifying the `ModernBertTokenizer`
    implementation details.
    """

    def setUp(self):
        self.merges = [
            "Ġ a",
            "Ġ t",
            "Ġ i",
            "Ġ b",
            "a i",
            "p l",
            "n e",
        ]
        self.merges += [
            "Ġa t",
            "p o",
            "r t",
            "Ġt h",
            "ai r",
            "pl a",
            "po rt",
        ]
        self.merges += [
            "Ġai r",
            "Ġa i",
            "pla ne",
        ]

        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])

        self.vocab = sorted(set(self.vocab))
        self.vocab += [
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "[UNK]",
        ]
        self.vocab = {token: i for i, token in enumerate(self.vocab)}

        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
        }

        self.input_data = [
            "[CLS] airplane at airport",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=ModernBertTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[
                [29, 23, 14, 24, 23, 16],
                [23, 14, 23, 16],
            ],
            expected_detokenize_output=[
                "[CLS] airplane at airport",
                " airplane airport",
            ],
        )

    def test_errors_missing_special_tokens(self):
        # A `dict` vocabulary that is simply missing `[MASK]`. A `list`
        # vocabulary would trip `BytePairTokenizer`'s type check first and
        # never reach `_update_special_token_ids`, so the error raised would
        # not be the one this test is named for.
        vocabulary = {
            token: index
            for token, index in self.vocab.items()
            if token != "[MASK]"
        }

        with self.assertRaisesRegex(ValueError, r"\[MASK\]"):
            ModernBertTokenizer(
                vocabulary=vocabulary,
                merges=self.merges,
            )

    def test_mask_token_absorbs_preceding_whitespace(self):
        """HF marks `[MASK]` as `AddedToken(..., lstrip=True)`.

        Whitespace immediately before the mask is part of the token, so it
        must not survive as a separate `Ġ` id. Runs on CPU with a toy
        vocabulary so the guard does not depend on network access, unlike
        `test_tokenizer_matches_hf_autotokenizer`.
        """
        tokenizer = ModernBertTokenizer(**self.init_kwargs)

        with_space = [int(i) for i in tokenizer(["airplane [MASK]"])[0]]
        without_space = [int(i) for i in tokenizer(["airplane[MASK]"])[0]]

        self.assertAllEqual(with_space, without_space)
        self.assertIn(tokenizer.mask_token_id, with_space)

        # Multiple spaces and a tab are all absorbed.
        for text in ("airplane  [MASK]", "airplane\t[MASK]"):
            self.assertAllEqual(
                [int(i) for i in tokenizer([text])[0]],
                without_space,
            )

    def test_mask_token_lstrip_leaves_other_tokens_alone(self):
        """Only `[MASK]` is `lstrip=True`; `[CLS]` and `[SEP]` are not."""
        tokenizer = ModernBertTokenizer(**self.init_kwargs)

        with_space = [int(i) for i in tokenizer(["airplane [SEP]"])[0]]
        without_space = [int(i) for i in tokenizer(["airplane[SEP]"])[0]]

        self.assertNotEqual(with_space, without_space)

    def test_special_token_ids(self):
        tokenizer = ModernBertTokenizer(**self.init_kwargs)

        self.assertEqual(
            tokenizer.start_token_id,
            tokenizer.cls_token_id,
        )
        self.assertEqual(
            tokenizer.end_token_id,
            tokenizer.sep_token_id,
        )

        self.assertNotEqual(
            tokenizer.start_token_id,
            tokenizer.end_token_id,
        )
        self.assertEqual(tokenizer.cls_token_id, self.vocab["[CLS]"])
        self.assertEqual(tokenizer.sep_token_id, self.vocab["[SEP]"])
        self.assertEqual(tokenizer.pad_token_id, self.vocab["[PAD]"])
        self.assertEqual(tokenizer.mask_token_id, self.vocab["[MASK]"])

    @pytest.mark.extra_large
    def test_tokenizer_matches_hf_autotokenizer(self):
        """Body-token parity against HF's AutoTokenizer.

        `ModernBertTokenizer` deliberately does not add `[CLS]`/`[SEP]` --
        in KerasHub that is `MultiSegmentPacker`'s job, inside the
        preprocessor. So this compares the body ids and asserts the special
        token ids resolve identically; the full packed sequence is covered
        by `test_preprocessor_matches_hf_autotokenizer` below.

        Neither `convert_modern_bert_checkpoints.py` (which tokenizes with
        AutoTokenizer) nor the converter backbone test (which feeds random
        ids) exercises this class, so these two tests are the only checks
        that would catch a tokenization mismatch.
        """
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-base"
        )
        keras_tokenizer = ModernBertTokenizer.from_preset(
            "hf://answerdotai/ModernBERT-base"
        )

        self.assertEqual(
            keras_tokenizer.cls_token_id,
            hf_tokenizer.cls_token_id,
        )
        self.assertEqual(
            keras_tokenizer.sep_token_id,
            hf_tokenizer.sep_token_id,
        )
        self.assertEqual(
            keras_tokenizer.pad_token_id,
            hf_tokenizer.pad_token_id,
        )
        self.assertEqual(
            keras_tokenizer.mask_token_id,
            hf_tokenizer.mask_token_id,
        )

        test_strings = [
            "The quick brown fox jumps over the lazy dog.",
            "ModernBERT uses local-global alternating attention.",
            "",
            # `[MASK]` is the only added token with `lstrip=True`, so these
            # cover the case the rest of the strings miss entirely.
            "The capital of France is [MASK].",
            "[MASK] is the capital of France.",
            "Two masks: [MASK] and [MASK].",
        ]

        for text in test_strings:
            hf_ids = hf_tokenizer(text)["input_ids"]
            keras_ids = [int(i) for i in keras_tokenizer([text])[0]]

            # Strip by position rather than by value, so a boundary id that
            # legitimately appears inside the body is not discarded too.
            self.assertEqual(hf_ids[0], hf_tokenizer.cls_token_id)
            self.assertEqual(hf_ids[-1], hf_tokenizer.sep_token_id)

            self.assertAllEqual(
                keras_ids,
                hf_ids[1:-1],
                msg=f"Body tokenization diverged for {text!r}",
            )

    @pytest.mark.extra_large
    def test_preprocessor_matches_hf_autotokenizer(self):
        """Full packed-sequence parity, including the `[CLS]`/`[SEP]` ids.

        This is the comparison against HF's complete `input_ids`. It belongs
        on the preprocessor because that is where KerasHub adds the boundary
        tokens.
        """
        from transformers import AutoTokenizer

        from keras_hub.src.models.modernbert.modern_bert_text_classifier_preprocessor import (  # noqa: E501
            ModernBertTextClassifierPreprocessor,
        )

        sequence_length = 32

        hf_tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-base"
        )
        preprocessor = ModernBertTextClassifierPreprocessor.from_preset(
            "hf://answerdotai/ModernBERT-base",
            sequence_length=sequence_length,
        )
        pad_token_id = preprocessor.tokenizer.pad_token_id

        test_strings = [
            "The quick brown fox jumps over the lazy dog.",
            "The capital of France is [MASK].",
            "[MASK] is the capital of France.",
        ]

        for text in test_strings:
            hf_ids = hf_tokenizer(text)["input_ids"]

            x = preprocessor([text])
            keras_ids = [int(i) for i in x["token_ids"][0]]
            unpadded_length = int(sum(int(m) for m in x["padding_mask"][0]))

            self.assertAllEqual(
                keras_ids[:unpadded_length],
                hf_ids,
                msg=f"Packed sequence diverged for {text!r}",
            )
            self.assertAllEqual(
                keras_ids[unpadded_length:],
                [pad_token_id] * (sequence_length - unpadded_length),
            )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=ModernBertTokenizer,
            preset="modernbert_base_en",
            input_data=["The quick brown fox."],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in ModernBertTokenizer.presets:
            self.run_preset_test(
                cls=ModernBertTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
