import base64

from keras_hub.src.models.mistral.mistral_tokenizer import MistralTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.convert_tekken import (
    convert_tekken_tokenizer,
)

# A small tiktoken-style split pattern, sufficient for the synthetic vocab.
_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?"
    r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _build_tekken_config():
    """Build a tiny synthetic `tekken.json` config for offline tests."""
    vocab = []
    # The 256 single bytes are always the base alphabet.
    for i in range(256):
        vocab.append(
            {
                "rank": i,
                "token_bytes": base64.b64encode(bytes([i])).decode(),
                "token_str": None,
            }
        )
    # A handful of merges, added as higher ranks.
    for rank, piece in [
        (256, b"th"),
        (257, b"the"),
        (258, b"in"),
        (259, b" t"),
        (260, b" th"),
    ]:
        vocab.append(
            {
                "rank": rank,
                "token_bytes": base64.b64encode(piece).decode(),
                "token_str": piece.decode("latin-1"),
            }
        )
    special_tokens = [
        {"rank": 0, "token_str": "<unk>", "is_control": True},
        {"rank": 1, "token_str": "<s>", "is_control": True},
        {"rank": 2, "token_str": "</s>", "is_control": True},
        {"rank": 3, "token_str": "<pad>", "is_control": True},
        {"rank": 4, "token_str": "[INST]", "is_control": True},
    ]
    config = {
        "pattern": _SPLIT_PATTERN,
        "num_vocab_tokens": 261,
        "default_vocab_size": 261,
        "default_num_special_tokens": 5,
        "version": "test",
    }
    # vocab_size = 256 bytes + 5 merges + 5 special tokens.
    return (
        {
            "config": config,
            "vocab": vocab,
            "special_tokens": special_tokens,
        },
        266,
    )


class ConvertTekkenTest(TestCase):
    def test_convert_tekken_tokenizer(self):
        tekken_config, vocab_size = _build_tekken_config()
        vocabulary, merges, special_tokens = convert_tekken_tokenizer(
            tekken_config, vocab_size
        )
        # 256 bytes + 5 merges + 5 special tokens.
        self.assertEqual(len(vocabulary), 266)
        self.assertEqual(len(merges), 5)
        self.assertEqual(
            special_tokens, ["<unk>", "<s>", "</s>", "<pad>", "[INST]"]
        )
        # Special tokens keep their reserved rank as id.
        self.assertEqual(vocabulary["<unk>"], 0)
        self.assertEqual(vocabulary["<s>"], 1)
        self.assertEqual(vocabulary["</s>"], 2)
        self.assertEqual(vocabulary["<pad>"], 3)

    def test_tokenizer_basics(self):
        tekken_config, vocab_size = _build_tekken_config()
        vocabulary, merges, _ = convert_tekken_tokenizer(
            tekken_config, vocab_size
        )
        self.run_preprocessing_layer_test(
            cls=MistralTokenizer,
            init_kwargs={
                "vocabulary": vocabulary,
                "merges": merges,
                "split_pattern": tekken_config["config"]["pattern"],
            },
            input_data=["the tin", "in the"],
        )

    def test_tekken_special_token_ids(self):
        tekken_config, vocab_size = _build_tekken_config()
        vocabulary, merges, _ = convert_tekken_tokenizer(
            tekken_config, vocab_size
        )
        tokenizer = MistralTokenizer(
            vocabulary=vocabulary,
            merges=merges,
            split_pattern=tekken_config["config"]["pattern"],
        )
        self.assertEqual(tokenizer.start_token_id, 1)
        self.assertEqual(tokenizer.end_token_id, 2)
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertEqual(tokenizer.vocabulary_size(), 266)
        # Round-trip a simple string.
        output = tokenizer("the tin")
        self.assertEqual(tokenizer.detokenize(output), "the tin")
