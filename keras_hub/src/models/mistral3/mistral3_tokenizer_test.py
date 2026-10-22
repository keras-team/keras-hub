import pytest

from keras_hub.src.models.mistral3.mistral3_tokenizer import (
    MISTRAL3_TEKKEN_SPLIT_PATTERN as _TEKKEN_SPLIT_PATTERN,
)
from keras_hub.src.models.mistral3.mistral3_tokenizer import Mistral3Tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import tf


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

    def test_special_tokens(self):
        tokenizer = Mistral3Tokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.start_token_id, 1)
        self.assertEqual(tokenizer.end_token_id, 2)
        self.assertEqual(tokenizer.pad_token_id, 3)

    def test_padding_uses_pad_token_id(self):
        tokenizer = Mistral3Tokenizer(sequence_length=6, **self.init_kwargs)
        output = tokenizer("the tin")
        self.assertEqual(list(output[-2:]), [3, 3])

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Mistral3Tokenizer.presets:
            self.run_preset_test(
                cls=Mistral3Tokenizer,
                preset=preset,
                input_data=["The quick brown fox jumped."],
            )

    def test_detokenize(self):
        tokenizer = Mistral3Tokenizer(**self.init_kwargs)
        token_ids = tokenizer(["the tin", "in the"])
        self.assertAllEqual(
            tokenizer.detokenize(token_ids), ["the tin", "in the"]
        )

    def test_detokenize_in_tf_data(self):
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        tokenizer = Mistral3Tokenizer(**self.init_kwargs)
        dataset = (
            tf.data.Dataset.from_tensor_slices(["the tin", "in the"])
            .batch(2)
            .map(tokenizer)
            .map(tokenizer.detokenize)
        )
        outputs = [x.decode("utf-8") for x in next(iter(dataset)).numpy()]
        self.assertEqual(outputs, ["the tin", "in the"])
