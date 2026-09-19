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
        with self.assertRaises(ValueError):
            ModernBertTokenizer(
                vocabulary=["a", "b", "c"],
                merges=[],
            )

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
        """End-to-end parity check against HF's AutoTokenizer.

        Verifies ModernBertTokenizer produces identical ids to HF's
        released tokenizer on the same strings, including the [CLS]/[SEP]
        boundary ids , the numerical-verification path in
        convert_modern_bert_checkpoints.py never exercises this tokenizer
        class directly (it tokenizes with AutoTokenizer, and the
        converter test only feeds random ids), so this is the only check
        that would catch a special-token mismatch like this one.
        """
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-base"
        )
        keras_tokenizer = ModernBertTokenizer.from_preset("modernbert_base_en")

        test_strings = [
            "The quick brown fox jumps over the lazy dog.",
            "ModernBERT uses local-global alternating attention.",
            "",
        ]

        for text in test_strings:
            hf_ids = hf_tokenizer(text)["input_ids"]
            keras_ids = [int(i) for i in keras_tokenizer([text])[0]]

            # keras_tokenizer does not add [CLS]/[SEP] itself (that's the
            # preprocessor's job); compare the raw BPE ids, then check the
            # special ids separately.
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

            hf_body_ids = [
                i
                for i in hf_ids
                if i
                not in (
                    hf_tokenizer.cls_token_id,
                    hf_tokenizer.sep_token_id,
                )
            ]
            self.assertAllEqual(keras_ids, hf_body_ids)

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
