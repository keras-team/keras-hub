from keras_hub.src.models.muse_glimmer.muse_glimmer_tokenizer import (
    MuseGlimmerTokenizer,
)
from keras_hub.src.tests.test_case import TestCase

DELIMITER_TOKENS = {
    "image_start_token": "<|image_start|>",
    "image_end_token": "<|image_end|>",
    "video_start_token": "<|vid_start|>",
    "video_end_token": "<|vid_end|>",
    "video_separator_token": "<|vid_frame_separator|>",
}


class MuseGlimmerTokenizerTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|end_of_text|>", "<|begin_of_text|>"]
        self.vocab += ["<|finetune_right_pad|>", "<|eot|>"]
        self.vocab += list(DELIMITER_TOKENS.values())
        self.vocab = sorted(set(self.vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = [
            "<|begin_of_text|>airplane at airport<|end_of_text|>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=MuseGlimmerTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_default_vision_token_ids(self):
        tokenizer = MuseGlimmerTokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.image_token_id, 200092)
        self.assertEqual(tokenizer.video_token_id, 200091)

    def test_eot_token_is_second_end_token(self):
        tokenizer = MuseGlimmerTokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.end_token2_id, self.vocab["<|eot|>"])

    def test_special_token_ids_include_vision_placeholders(self):
        tokenizer = MuseGlimmerTokenizer(**self.init_kwargs)
        ids = tokenizer.special_token_ids
        self.assertIn(tokenizer.image_token_id, ids)
        self.assertIn(tokenizer.video_token_id, ids)
        self.assertIn(tokenizer.start_token_id, ids)

    def test_delimiter_ids_come_from_the_vocabulary(self):
        tokenizer = MuseGlimmerTokenizer(**self.init_kwargs)
        for name, token in DELIMITER_TOKENS.items():
            self.assertEqual(getattr(tokenizer, name), token)
            token_id = getattr(tokenizer, f"{name}_id")
            self.assertEqual(token_id, self.vocab[token])
            self.assertIn(token_id, tokenizer.special_token_ids)

    def test_missing_delimiter_token_raises(self):
        for name, token in DELIMITER_TOKENS.items():
            vocab = {k: v for k, v in self.vocab.items() if k != token}
            with self.assertRaisesRegex(ValueError, token):
                MuseGlimmerTokenizer(vocabulary=vocab, merges=self.merges)

    def test_non_default_tokens_survive_serialization(self):
        custom = {
            "bos_token": "<|start|>",
            "eos_token": "<|message|>",
            "pad_token": "<|eom|>",
            "image_start_token": "<|a|>",
            "image_end_token": "<|b|>",
            "video_start_token": "<|c|>",
            "video_end_token": "<|d|>",
            "video_separator_token": "<|e|>",
        }
        vocab = dict(self.vocab)
        for token in custom.values():
            vocab[token] = len(vocab)
        tokenizer = MuseGlimmerTokenizer(
            vocabulary=vocab, merges=self.merges, **custom
        )
        # A tokenizer built from a config has no vocabulary, so the ids
        # are checked here.
        names = {
            "bos_token": "start_token",
            "eos_token": "end_token",
            "pad_token": "pad_token",
        }
        for name, token in custom.items():
            attribute = names.get(name, name)
            self.assertEqual(getattr(tokenizer, attribute), token)
            token_id = getattr(tokenizer, f"{attribute}_id")
            self.assertEqual(token_id, vocab[token])
        # The helper fails if `get_config` drops one of these tokens.
        self.run_serialization_test(tokenizer)

    def test_pre_tokenizer_matches_checkpoint_split(self):
        tokenizer = MuseGlimmerTokenizer(**self.init_kwargs)
        tokenizer._maybe_initialized_tokenizers()

        def split(text):
            pre = tokenizer._tokenizer.pre_tokenizer
            return [piece for piece, _ in pre.pre_tokenize_str(text)]

        # Two leading spaces split into one space and one word prefix.
        self.assertEqual(split("  ab"), ["Ġ", "Ġab"])
        # Digits group in threes.
        self.assertEqual(split("1234567"), ["123", "456", "7"])
        # Combining marks stay inside the word.
        self.assertEqual(len(split("नमस्ते")), 1)
        # Mixed-case words split at the case change.
        self.assertEqual(split("camelCase"), ["camel", "Case"])

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            MuseGlimmerTokenizer(
                vocabulary={"foo": 0, "bar": 1}, merges=["fo o"]
            )
