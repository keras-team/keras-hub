import pytest

from keras_hub.src.models.bert.bert_text_classifier_preprocessor import (
    BertTextClassifierPreprocessor,
)
from keras_hub.src.models.bert.bert_tokenizer import BertTokenizer
from keras_hub.src.tests.test_case import TestCase

try:
    import grain
except ImportError:
    grain = None


class BertTextClassifierPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["THE", "QUICK", "BROWN", "FOX"]
        self.vocab += ["the", "quick", "brown", "fox"]
        self.tokenizer = BertTokenizer(vocabulary=self.vocab)
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
            cls=BertTextClassifierPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[2, 5, 6, 7, 8, 1, 3, 0]],
                    "segment_ids": [[0, 0, 0, 0, 0, 0, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 0]],
                },
                [1],  # Pass through labels.
                [1.0],  # Pass through sample_weights.
            ),
        )

    @pytest.mark.skipif(grain is None, reason="grain is not installed")
    def test_grain_unpacks_tuple_elements(self):
        preprocessor = BertTextClassifierPreprocessor(**self.init_kwargs)
        expected = preprocessor(*self.input_data)
        # Unbatched `(x, y, sample_weight)` elements.
        elements = [("THE QUICK BROWN FOX.", 1, 1.0)]
        ds = grain.MapDataset.source(elements).map(preprocessor).batch(1)
        (output,) = list(ds)
        self.assertAllClose(output, expected)
        # Batched `(x, y, sample_weight)` elements.
        ds = grain.MapDataset.source([self.input_data]).map(preprocessor)
        (output,) = list(ds)
        self.assertAllClose(output, expected)
        # `(x, y)` elements.
        expected = preprocessor(self.input_data[0], self.input_data[1])
        ds = grain.MapDataset.source([self.input_data[:2]]).map(preprocessor)
        (output,) = list(ds)
        self.assertAllClose(output, expected)

    def test_errors_for_2d_list_input(self):
        preprocessor = BertTextClassifierPreprocessor(**self.init_kwargs)
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            preprocessor(ambiguous_input)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BertTextClassifierPreprocessor.presets:
            self.run_preset_test(
                cls=BertTextClassifierPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
