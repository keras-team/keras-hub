import pytest

from keras_hub.src.models.mistral3.mistral3_tokenizer import (
    MISTRAL3_TEKKEN_SPLIT_PATTERN as _TEKKEN_SPLIT_PATTERN,
)
from keras_hub.src.models.mistral3.mistral3_tokenizer import Mistral3Tokenizer
from keras_hub.src.tests.test_case import TestCase


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


def _tekken_vision_init_kwargs():
    """Build a tiny Tekken (byte-level BPE) vocabulary with image tokens."""
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


class Mistral3TokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = _tekken_vision_init_kwargs()

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Mistral3Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=["the tin", "in the"],
            expected_output=[[265, 40, 124, 266], [266, 40, 265]],
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Mistral3Tokenizer.presets:
            self.run_preset_test(
                cls=Mistral3Tokenizer,
                preset=preset,
                input_data=["The quick brown fox jumped."],
            )
