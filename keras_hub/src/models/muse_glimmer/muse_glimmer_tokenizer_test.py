from keras_hub.src.models.muse_glimmer.muse_glimmer_tokenizer import (
    MuseGlimmerTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


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

    def test_get_config_round_trips_non_default_tokens(self):
        vocab = dict(self.vocab)
        for token in ("<|start|>", "<|message|>", "<|eom|>"):
            vocab[token] = len(vocab)
        tokenizer = MuseGlimmerTokenizer(
            vocabulary=vocab,
            merges=self.merges,
            bos_token="<|start|>",
            eos_token="<|message|>",
            pad_token="<|eom|>",
        )
        revived = MuseGlimmerTokenizer.from_config(tokenizer.get_config())
        self.assertEqual(revived.start_token, "<|start|>")
        self.assertEqual(revived.end_token, "<|message|>")
        self.assertEqual(revived.pad_token, "<|eom|>")

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
