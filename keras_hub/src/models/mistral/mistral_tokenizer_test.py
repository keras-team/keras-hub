import os

import pytest

from keras_hub.src.models.mistral.mistral_tokenizer import (
    MistralTekkenTokenizer,
)
from keras_hub.src.models.mistral.mistral_tokenizer import MistralTokenizer
from keras_hub.src.tests.test_case import TestCase

# A tiktoken-style split pattern, matching the Tekken format.
_TEKKEN_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|\p{N}| ?[^\s\p{L}\p{N}]+"
    r"[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _tekken_init_kwargs():
    """Build a tiny Tekken (byte-level BPE) vocabulary for offline tests."""
    byte_encoder = _bytes_to_unicode()
    # Special tokens occupy the first ids, matching the Tekken layout.
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "[INST]"]
    vocabulary = {token: i for i, token in enumerate(special_tokens)}
    offset = len(special_tokens)
    # The 256 single bytes form the base alphabet.
    for i in range(256):
        vocabulary[byte_encoder[i]] = offset + i
    # A few merges on top.
    merges = []
    next_id = offset + 256
    for a, b in [("t", "h"), ("th", "e"), ("i", "n")]:
        vocabulary[a + b] = next_id
        merges.append(f"{a} {b}")
        next_id += 1
    return {
        "vocabulary": vocabulary,
        "merges": merges,
        "split_pattern": _TEKKEN_SPLIT_PATTERN,
    }


def _tekken_vision_init_kwargs():
    """Like `_tekken_init_kwargs`, but with the Mistral image tokens."""
    byte_encoder = _bytes_to_unicode()
    special_tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<pad>",
        "[INST]",
        "[IMG]",
        "[IMG_BREAK]",
        "[IMG_END]",
    ]
    vocabulary = {token: i for i, token in enumerate(special_tokens)}
    offset = len(special_tokens)
    for i in range(256):
        vocabulary[byte_encoder[i]] = offset + i
    merges = []
    next_id = offset + 256
    for a, b in [("t", "h"), ("th", "e"), ("i", "n")]:
        vocabulary[a + b] = next_id
        merges.append(f"{a} {b}")
        next_id += 1
    return {
        "vocabulary": vocabulary,
        "merges": merges,
        "split_pattern": _TEKKEN_SPLIT_PATTERN,
    }


class MistralTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_mistral_test_proto.py
            "proto": os.path.join(
                self.get_test_data_dir(), "mistral_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=MistralTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[3, 8, 4, 6], [3, 5, 7, 9]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            MistralTokenizer(
                # Generated using create_no_special_token_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MistralTokenizer,
            preset="mistral_7b_en",
            input_data=["The quick brown fox."],
            expected_output=[[415, 2936, 9060, 285, 1142, 28723]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MistralTokenizer.presets:
            self.run_preset_test(
                cls=MistralTokenizer,
                preset=preset,
                input_data=self.input_data,
            )

    def test_no_vision_tokens_by_default(self):
        tokenizer = MistralTokenizer(**self.init_kwargs)
        self.assertFalse(tokenizer.has_vision_tokens)
        self.assertEqual(tokenizer.image_placeholder_token_id, -1)
        self.assertEqual(tokenizer.image_break_token_id, -1)
        self.assertEqual(tokenizer.image_end_token_id, -1)

    def test_no_vision_tokens_when_explicitly_disabled(self):
        tokenizer = MistralTokenizer(
            has_vision_tokens=False, **self.init_kwargs
        )
        self.assertFalse(tokenizer.has_vision_tokens)
        self.assertEqual(tokenizer.image_placeholder_token_id, -1)
        self.assertEqual(tokenizer.image_break_token_id, -1)
        self.assertEqual(tokenizer.image_end_token_id, -1)

    def test_vision_tokens_when_enabled(self):
        tokenizer = MistralTokenizer(
            # Generated using create_mistral_vision_test_proto.py
            proto=os.path.join(
                self.get_test_data_dir(), "mistral_vision_test_vocab.spm"
            ),
            has_vision_tokens=True,
        )
        self.assertTrue(tokenizer.has_vision_tokens)
        ids = [
            tokenizer.image_placeholder_token_id,
            tokenizer.image_break_token_id,
            tokenizer.image_end_token_id,
        ]
        for token_id in ids:
            self.assertNotEqual(token_id, -1)
        # All three ids are distinct from each other.
        self.assertEqual(len(set(ids)), 3)
        # And distinct from the other existing special tokens.
        existing_ids = {
            tokenizer.start_token_id,
            tokenizer.end_token_id,
            tokenizer.pad_token_id,
        }
        self.assertTrue(existing_ids.isdisjoint(set(ids)))

    def test_vision_tokens_config_round_trip(self):
        tokenizer = MistralTokenizer(
            # Generated using create_mistral_vision_test_proto.py
            proto=os.path.join(
                self.get_test_data_dir(), "mistral_vision_test_vocab.spm"
            ),
            has_vision_tokens=True,
        )
        config = tokenizer.get_config()
        self.assertTrue(config["has_vision_tokens"])
        # `proto` is persisted as a file asset rather than in `get_config()`
        # (see `SentencePieceTokenizer.get_config()`), so re-supply it
        # explicitly to fully rebuild an equivalent tokenizer.
        config["proto"] = tokenizer.proto
        restored = MistralTokenizer.from_config(config)
        self.assertTrue(restored.has_vision_tokens)
        self.assertEqual(
            restored.image_placeholder_token_id,
            tokenizer.image_placeholder_token_id,
        )
        self.assertEqual(
            restored.image_break_token_id, tokenizer.image_break_token_id
        )
        self.assertEqual(
            restored.image_end_token_id, tokenizer.image_end_token_id
        )


class MistralTekkenTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = _tekken_init_kwargs()
        self.input_data = ["the tin", "in the"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=MistralTekkenTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_special_tokens(self):
        tokenizer = MistralTekkenTokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.start_token_id, 1)
        self.assertEqual(tokenizer.end_token_id, 2)
        self.assertEqual(tokenizer.pad_token_id, 0)
        # Round-trip a simple string.
        output = tokenizer("the tin")
        self.assertEqual(tokenizer.detokenize(output), "the tin")

    def test_no_vision_tokens_by_default(self):
        tokenizer = MistralTekkenTokenizer(**self.init_kwargs)
        self.assertFalse(tokenizer.has_vision_tokens)
        self.assertEqual(tokenizer.image_placeholder_token_id, -1)
        self.assertEqual(tokenizer.image_break_token_id, -1)
        self.assertEqual(tokenizer.image_end_token_id, -1)

    def test_no_vision_tokens_when_explicitly_disabled(self):
        tokenizer = MistralTekkenTokenizer(
            has_vision_tokens=False, **self.init_kwargs
        )
        self.assertFalse(tokenizer.has_vision_tokens)
        self.assertEqual(tokenizer.image_placeholder_token_id, -1)
        self.assertEqual(tokenizer.image_break_token_id, -1)
        self.assertEqual(tokenizer.image_end_token_id, -1)

    def test_vision_tokens_when_enabled(self):
        tokenizer = MistralTekkenTokenizer(
            has_vision_tokens=True, **_tekken_vision_init_kwargs()
        )
        self.assertTrue(tokenizer.has_vision_tokens)
        ids = [
            tokenizer.image_placeholder_token_id,
            tokenizer.image_break_token_id,
            tokenizer.image_end_token_id,
        ]
        for token_id in ids:
            self.assertNotEqual(token_id, -1)
        # All three ids are distinct from each other.
        self.assertEqual(len(set(ids)), 3)
        # And distinct from the other existing special tokens.
        existing_ids = {
            tokenizer.start_token_id,
            tokenizer.end_token_id,
            tokenizer.pad_token_id,
        }
        self.assertTrue(existing_ids.isdisjoint(set(ids)))
        # The image tokens are unsplittable, just like `<s>`/`</s>`.
        self.assertIn(
            tokenizer.image_placeholder_token, tokenizer.unsplittable_tokens
        )
        self.assertIn(
            tokenizer.image_break_token, tokenizer.unsplittable_tokens
        )
        self.assertIn(tokenizer.image_end_token, tokenizer.unsplittable_tokens)

    def test_vision_tokens_config_round_trip(self):
        tokenizer = MistralTekkenTokenizer(
            has_vision_tokens=True, **_tekken_vision_init_kwargs()
        )
        config = tokenizer.get_config()
        self.assertTrue(config["has_vision_tokens"])
        # `vocabulary`/`merges` are persisted as file assets rather than in
        # `get_config()` (see `BytePairTokenizer.get_config()`), so re-supply
        # them explicitly to fully rebuild an equivalent tokenizer.
        config["vocabulary"] = tokenizer.vocabulary
        config["merges"] = tokenizer.merges
        restored = MistralTekkenTokenizer.from_config(config)
        self.assertTrue(restored.has_vision_tokens)
        self.assertEqual(
            restored.image_placeholder_token_id,
            tokenizer.image_placeholder_token_id,
        )
        self.assertEqual(
            restored.image_break_token_id, tokenizer.image_break_token_id
        )
        self.assertEqual(
            restored.image_end_token_id, tokenizer.image_end_token_id
        )
