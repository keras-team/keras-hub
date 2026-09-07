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
        self.vocab += ["<|finetune_right_pad|>"]
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

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            MuseGlimmerTokenizer(
                vocabulary={"foo": 0, "bar": 1}, merges=["fo o"]
            )
