"""CPU unit tests for serving KerasHub tokenizers through vLLM.

Drives `KerasHubTokenizer` with a tiny fake KerasHub tokenizer (no preset
download) whose preset defines bos/eos but no pad token, exercising the
special-token alignment, start-token prepending, `skip_special_tokens`
decoding, and the token<->id<->string surface vLLM's detokenizer uses.
"""

import json
import os
import tempfile
from unittest import mock

import numpy as np
from keras import ops

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm.tokenizer import KerasHubTokenizer


class _FakeTokenizer:
    """Minimal stand-in for a KerasHub tokenizer (bos/eos, but no pad)."""

    name = "fake"
    start_token_id = 1
    end_token_id = 2
    # No pad_token_id attribute: this preset has no pad token.

    _vocab = {0: "a", 1: "<s>", 2: "</s>", 3: "b", 4: "c"}

    def vocabulary_size(self):
        return len(self._vocab)

    def get_vocabulary(self):
        return [self._vocab[i] for i in range(len(self._vocab))]

    def id_to_token(self, token_id):
        return self._vocab[int(token_id)]

    def token_to_id(self, token):
        return {v: k for k, v in self._vocab.items()}[token]

    def detokenize(self, token_ids):
        text = "".join(self._vocab[int(i)] for i in token_ids)
        return np.array(text.encode("utf-8"))

    def __call__(self, text):
        return np.array([3, 4])


class KerasHubTokenizerTest(TestCase):
    def setUp(self):
        super().setUp()
        self.adapter = KerasHubTokenizer(_FakeTokenizer())

    def test_scalar_and_numpy_ids(self):
        # vLLM hands over numpy scalars and 0-d arrays as well as ints.
        self.assertEqual(self.adapter.decode(3), "b")
        self.assertEqual(self.adapter.decode(np.int64(3)), "b")
        self.assertEqual(
            self.adapter.decode(np.array(3), skip_special_tokens=True), "b"
        )
        self.assertEqual(self.adapter.convert_ids_to_tokens(np.int64(4)), "c")
        # A scalar special id filtered out maps to None, per the HF API.
        self.assertIsNone(
            self.adapter.convert_ids_to_tokens(1, skip_special_tokens=True)
        )

    def test_vocab_is_cached(self):
        first = self.adapter.get_vocab()
        self.assertIs(first, self.adapter.get_vocab())
        self.assertEqual(self.adapter.convert_ids_to_tokens([3, 4]), ["b", "c"])
        self.assertEqual(self.adapter.convert_tokens_to_ids(["b", "c"]), [3, 4])
        # Byte tokens normalize the same way the vocab was built.
        self.assertEqual(self.adapter.convert_tokens_to_ids(b"b"), 3)
        self.assertEqual(
            self.adapter.convert_tokens_to_ids([b"b", "c"]), [3, 4]
        )
        with self.assertRaises(KeyError):
            self.adapter.convert_tokens_to_ids("not_a_token")

    def test_none_vocabulary_falls_back_to_id_to_token(self):
        class _NoneVocabTokenizer(_FakeTokenizer):
            def get_vocabulary(self):
                return None

        adapter = KerasHubTokenizer(_NoneVocabTokenizer())
        self.assertEqual(adapter.convert_ids_to_tokens([3, 4]), ["b", "c"])

    def test_unknown_token_raises_in_convert_to_string(self):
        with self.assertRaises(KeyError):
            self.adapter.convert_tokens_to_string(["b", "not_a_token"])

    def test_decode_empty_and_all_special(self):
        self.assertEqual(self.adapter.decode([]), "")
        # bos/eos only, all filtered out.
        self.assertEqual(
            self.adapter.decode([1, 2], skip_special_tokens=True), ""
        )
        self.assertEqual(self.adapter.decode(np.array([], dtype="int32")), "")

    def test_tensor_ids(self):
        # Backend tensors decode like lists (converted via numpy, not
        # iterated element-wise).
        tensor_ids = ops.convert_to_tensor([3, 4], dtype="int32")
        self.assertEqual(self.adapter.decode(tensor_ids), "bc")
        # 0-d backend tensor behaves like a scalar id.
        scalar_tensor = ops.convert_to_tensor(3, dtype="int32")
        self.assertEqual(self.adapter.decode(scalar_tensor), "b")
        self.assertEqual(
            self.adapter.convert_ids_to_tokens(tensor_ids), ["b", "c"]
        )

    def test_truncation_side_honored(self):
        # bos(1) + [3, 4]; right truncation keeps the start, left keeps
        # the end (what vLLM's renderer sets truncation_side for).
        right = self.adapter.encode("xx", truncation=True, max_length=2)
        self.assertEqual(right, [1, 3])
        self.adapter.truncation_side = "left"
        left = self.adapter.encode("xx", truncation=True, max_length=2)
        self.assertEqual(left, [3, 4])
        self.adapter.truncation_side = "right"

    def test_encode_rejects_batches(self):
        with self.assertRaisesRegex(TypeError, "single string"):
            self.adapter.encode(["a", "b"])

    def test_special_ids_drop_missing_token(self):
        # pad is absent, so only bos/eos ids appear (no defaulted id).
        self.assertEqual(self.adapter.all_special_ids, [1, 2])
        self.assertIsNone(self.adapter.pad_token_id)

    def test_special_tokens_derived_and_aligned(self):
        # Strings come from the vocab, not hardcoded, and stay 1:1 with ids.
        self.assertEqual(self.adapter.all_special_tokens, ["<s>", "</s>"])
        self.assertEqual(
            len(self.adapter.all_special_tokens),
            len(self.adapter.all_special_ids),
        )

    def test_encode_prepends_start_token(self):
        # Mirrors KerasHub generation (add_start_token=True).
        self.assertEqual(self.adapter.encode("bc"), [1, 3, 4])
        self.assertEqual(
            self.adapter.encode("bc", add_special_tokens=False), [3, 4]
        )
        self.assertEqual(self.adapter.num_special_tokens_to_add(), 1)

    def test_encode_truncates(self):
        self.assertEqual(
            self.adapter.encode("bc", truncation=True, max_length=2), [1, 3]
        )

    def test_call_returns_batch_encoding(self):
        out = self.adapter("bc")
        self.assertEqual(out["input_ids"], [1, 3, 4])
        self.assertEqual(out["attention_mask"], [1, 1, 1])

    def test_decode_skips_special_tokens(self):
        ids = [1, 3, 2]  # <s> b </s>
        self.assertEqual(self.adapter.decode(ids), "<s>b</s>")
        self.assertEqual(
            self.adapter.decode(ids, skip_special_tokens=True), "b"
        )

    def test_decode_returns_str(self):
        self.assertIsInstance(self.adapter.decode([3, 4]), str)

    def test_token_id_string_round_trip(self):
        # vLLM's incremental detokenizer drives these three methods.
        self.assertEqual(self.adapter.convert_ids_to_tokens([3, 4]), ["b", "c"])
        self.assertEqual(self.adapter.convert_tokens_to_ids(["b", "c"]), [3, 4])
        self.assertEqual(self.adapter.convert_tokens_to_ids("b"), 3)
        self.assertEqual(
            self.adapter.convert_tokens_to_string(["b", "c"]), "bc"
        )

    def test_sizes_and_hash(self):
        self.assertEqual(self.adapter.vocab_size, 5)
        self.assertEqual(len(self.adapter), 5)
        self.assertEqual(self.adapter.max_token_id, 4)
        self.assertEqual(self.adapter.max_chars_per_token, len("</s>"))
        self.assertEqual(hash(self.adapter), hash(self.adapter))

    def test_chat_template_raises(self):
        with self.assertRaises(ValueError):
            self.adapter.apply_chat_template([{"role": "user", "content": "x"}])

    def test_from_pretrained_reads_preset_from_config(self):
        # A dir written by setup_vllm_model carries `keras_hub_preset`; the
        # class loads that preset rather than treating the path as one.
        temp_dir = tempfile.mkdtemp()
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump({"keras_hub_preset": "some_preset"}, f)
        with mock.patch.object(
            KerasHubTokenizer, "__init__", return_value=None
        ) as init:
            KerasHubTokenizer.from_pretrained(temp_dir)
        init.assert_called_once_with("some_preset")

    def test_get_vocab_decodes_byte_tokens(self):
        # Byte-level BPE vocabularies expose some tokens as bytes; vLLM needs
        # str keys, so get_vocab must decode them in both code paths.
        class _ByteVocab:
            name = "bytes"

            def get_vocabulary(self):
                return ["a", b"\xc3\xa9", "c"]  # b"..." -> "é"

            def vocabulary_size(self):
                return 3

        vocab = KerasHubTokenizer(_ByteVocab()).get_vocab()
        self.assertTrue(all(isinstance(k, str) for k in vocab))
        self.assertEqual(vocab["é"], 1)
