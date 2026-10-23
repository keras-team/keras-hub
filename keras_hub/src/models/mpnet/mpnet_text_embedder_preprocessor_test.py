import pytest

from keras_hub.src.models.mpnet.mpnet_text_embedder_preprocessor import (
    MPNetTextEmbedderPreprocessor,
)
from keras_hub.src.models.mpnet.mpnet_tokenizer import MPNetTokenizer
from keras_hub.src.tests.test_case import TestCase


class MPNetTextEmbedderPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["<s>", "<pad>", "</s>", "[UNK]", "<mask>"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.tokenizer = MPNetTokenizer(vocabulary=self.vocab)
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (
            ["THE QUICK BROWN FOX."],
            [1],  # Pass through labels.
            [1.0],  # Pass through sample_weights.
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=MPNetTextEmbedderPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[0, 5, 6, 7, 8, 9, 2, 1]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 0]],
                },
                [1],  # Pass through labels.
                [1.0],  # Pass through sample_weights.
            ),
        )

    def test_sentence_pair_uses_double_separator(self):
        preprocessor = MPNetTextEmbedderPreprocessor(
            self.tokenizer, sequence_length=10
        )
        # Pass segments with `x=`, a positional tuple is unpacked as `(x, y)`.
        output = preprocessor(x=(["the quick"], ["brown fox"]))
        # <s> the quick </s> </s> brown fox </s> <pad> <pad>
        self.assertAllEqual(
            output["token_ids"], [[0, 5, 6, 2, 2, 7, 8, 2, 1, 1]]
        )
        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
        )

    def test_no_segment_ids_in_output(self):
        preprocessor = MPNetTextEmbedderPreprocessor(**self.init_kwargs)
        output, _, _ = preprocessor(*self.input_data)
        self.assertEqual(set(output.keys()), {"token_ids", "padding_mask"})

    def test_truncation_keeps_special_tokens(self):
        preprocessor = MPNetTextEmbedderPreprocessor(
            self.tokenizer, sequence_length=4
        )
        output = preprocessor(["the quick brown fox."])
        self.assertAllEqual(output["token_ids"], [[0, 5, 6, 2]])

    def test_sequence_length_override(self):
        preprocessor = MPNetTextEmbedderPreprocessor(**self.init_kwargs)
        preprocessor(*self.input_data)
        preprocessor.sequence_length = 6
        output, _, _ = preprocessor(*self.input_data)
        self.assertAllEqual(output["token_ids"], [[0, 5, 6, 7, 8, 2]])

    def test_errors_for_2d_list_input(self):
        preprocessor = MPNetTextEmbedderPreprocessor(**self.init_kwargs)
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            preprocessor(ambiguous_input)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MPNetTextEmbedderPreprocessor.presets:
            self.run_preset_test(
                cls=MPNetTextEmbedderPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
