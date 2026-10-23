import pytest

from keras_hub.src.models.mpnet.mpnet_tokenizer import MPNetTokenizer
from keras_hub.src.tests.test_case import TestCase


class MPNetTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = ["<s>", "<pad>", "</s>", "[UNK]", "<mask>"]
        self.vocab += ["the", "quick", "brown", "fox", ".", "cafe", "naive"]
        self.init_kwargs = {"vocabulary": self.vocab}
        self.input_data = ["the quick brown fox", "the fox"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=MPNetTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[5, 6, 7, 8], [5, 8]],
        )

    def test_lowercase_and_punctuation(self):
        tokenizer = MPNetTokenizer(**self.init_kwargs)
        output = tokenizer(["THE QUICK BROWN FOX.", "THE FOX"])
        self.assertAllEqual(output, [[5, 6, 7, 8, 9], [5, 8]])

    def test_special_token_ids(self):
        tokenizer = MPNetTokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.cls_token_id, 0)
        self.assertEqual(tokenizer.start_token_id, 0)
        self.assertEqual(tokenizer.pad_token_id, 1)
        self.assertEqual(tokenizer.sep_token_id, 2)
        self.assertEqual(tokenizer.end_token_id, 2)
        self.assertEqual(tokenizer.mask_token_id, 4)

    def test_lowercase_strips_accents(self):
        tokenizer = MPNetTokenizer(**self.init_kwargs)
        self.assertTrue(tokenizer.strip_accents)
        self.assertAllEqual(tokenizer(["Café NAÏVE"]), [[10, 11]])

    def test_cased_keeps_accents(self):
        tokenizer = MPNetTokenizer(vocabulary=self.vocab, lowercase=False)
        self.assertFalse(tokenizer.strip_accents)
        self.assertAllEqual(tokenizer(["café"]), [[3]])

    def test_out_of_vocabulary_maps_to_unk(self):
        tokenizer = MPNetTokenizer(**self.init_kwargs)
        self.assertAllEqual(tokenizer(["zebra"]), [[3]])

    def test_tokenizer_special_tokens(self):
        input_data = ["<s> THE <mask> FOX </s> <pad>"]
        tokenizer = MPNetTokenizer(
            **self.init_kwargs, special_tokens_in_strings=True
        )
        self.assertAllEqual(tokenizer(input_data), [[0, 5, 4, 8, 2, 1]])

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            MPNetTokenizer(vocabulary=["a", "b", "c"])

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MPNetTokenizer,
            preset="all_mpnet_base_v2_en",
            input_data=["The quick brown fox."],
            expected_output=[[2000, 4252, 2833, 4423, 1016]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MPNetTokenizer.presets:
            self.run_preset_test(
                cls=MPNetTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
