import pytest

from keras_hub.src.models.clip.clip_preprocessor import CLIPPreprocessor
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.tests.test_case import TestCase


class CLIPPreprocessorTest(TestCase):
    def setUp(self):
        vocab = ["air", "plane</w>", "port</w>"]
        vocab += ["<|endoftext|>", "<|startoftext|>"]
        vocab = dict([(token, i + 1) for i, token in enumerate(vocab)])
        merges = ["a i", "p l", "n e</w>", "p o", "r t</w>", "ai r", "pl a"]
        merges += ["po rt</w>", "pla ne</w>"]
        self.tokenizer = CLIPTokenizer(vocabulary=vocab, merges=merges)
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = [" airplane airport"]

    def test_preprocessor_basics(self):
        self.run_preprocessing_layer_test(
            cls=CLIPPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output={
                "token_ids": [[5, 1, 2, 1, 3, 4, 4, 4]],
                "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
            },
        )

    def test_no_start_end_token(self):
        input_data = [" airplane airport"] * 4
        preprocessor = CLIPPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=8,
            add_start_token=False,
            add_end_token=False,
        )
        x = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[1, 2, 1, 3, 4, 4, 4, 4]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)

    def test_sequence_length_override(self):
        input_data = " airplane airport"
        preprocessor = CLIPPreprocessor(**self.init_kwargs)
        x = preprocessor(input_data, sequence_length=5)
        self.assertAllEqual(x["token_ids"], [5, 1, 2, 1, 4])

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        self.skipTest("TODO")
        for preset in CLIPPreprocessor.presets:
            self.run_preset_test(
                cls=CLIPPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
