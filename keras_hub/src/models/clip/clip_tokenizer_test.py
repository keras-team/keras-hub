import pytest

from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.tests.test_case import TestCase


class CLIPTokenizerTest(TestCase):
    def setUp(self):
        vocab = ["air", "plane</w>", "port</w>"]
        vocab += ["<|endoftext|>", "<|startoftext|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(vocab)])
        merges = ["a i", "p l", "n e</w>", "p o", "r t</w>", "ai r", "pl a"]
        merges += ["po rt</w>", "pla ne</w>"]
        self.merges = merges
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = ["airplane ", " airport"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=CLIPTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            # Whitespaces should be removed.
            expected_output=[[0, 1], [0, 2]],
            expected_detokenize_output=["airplane", "airport"],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            CLIPTokenizer(vocabulary={"foo": 0, "bar": 1}, merges=["fo o"])

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=CLIPTokenizer,
            preset="clip_vit_base_patch16",
            input_data=["The quick brown fox."],
            expected_output=[[51, 797, 3712, 2866, 3240, 269]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in CLIPTokenizer.presets:
            self.run_preset_test(
                cls=CLIPTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
