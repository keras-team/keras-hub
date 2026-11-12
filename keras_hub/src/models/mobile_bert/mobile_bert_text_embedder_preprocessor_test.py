import numpy as np
import pytest

from keras_hub.src.models.mobile_bert.mobile_bert_text_embedder_preprocessor import (  # noqa: E501
    MobileBertTextEmbedderPreprocessor,
)
from keras_hub.src.models.mobile_bert.mobile_bert_tokenizer import (
    MobileBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MobileBertTextEmbedderPreprocessorTest(TestCase):
    def setUp(self):
        # `[PAD]` = 0, `[UNK]` = 1, `[CLS]` = 2, `[SEP]` = 3, `[MASK]` = 4.
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox"]
        self.tokenizer = MobileBertTokenizer(vocabulary=self.vocab)
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (
            ["the quick brown fox"],
            [1],  # Pass through labels.
            np.array([1.0], dtype="float32"),  # Pass through sample_weights.
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=MobileBertTextEmbedderPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    # [CLS] the quick brown fox [SEP] [PAD] [PAD]
                    "token_ids": [[2, 5, 6, 7, 8, 3, 0, 0]],
                    "segment_ids": [[0, 0, 0, 0, 0, 0, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [1],  # Pass through labels.
                np.array(
                    [1.0], dtype="float32"
                ),  # Pass through sample_weights.
            ),
        )

    def test_sentence_pair(self):
        """Pairs are packed as `[CLS] A [SEP] B [SEP]` with segment ids."""
        preprocessor = MobileBertTextEmbedderPreprocessor(**self.init_kwargs)
        # Pass `y` explicitly so the tuple is read as two segments, rather
        # than being unpacked as `(x, y)`.
        x = preprocessor((["the quick"], ["the fox"]), y=None)
        self.assertAllEqual(x["token_ids"], [[2, 5, 6, 3, 5, 8, 3, 0]])
        self.assertAllEqual(x["segment_ids"], [[0, 0, 0, 0, 1, 1, 1, 0]])
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]])

    def test_truncation_keeps_boundary_tokens(self):
        preprocessor = MobileBertTextEmbedderPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=4,
        )
        x = preprocessor(["the quick brown fox"])
        # [CLS] the quick [SEP]
        self.assertAllEqual(x["token_ids"], [[2, 5, 6, 3]])
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1]])

    def test_errors_for_2d_list_input(self):
        preprocessor = MobileBertTextEmbedderPreprocessor(**self.init_kwargs)
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            preprocessor(ambiguous_input)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MobileBertTextEmbedderPreprocessor.presets:
            self.run_preset_test(
                cls=MobileBertTextEmbedderPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
