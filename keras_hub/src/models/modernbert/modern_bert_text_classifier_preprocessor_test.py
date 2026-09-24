from keras_hub.src.models.modernbert.modern_bert_text_classifier_preprocessor import (  # noqa: E501
    ModernBertTextClassifierPreprocessor,
)
from keras_hub.src.models.modernbert.modern_bert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertTextClassifierPreprocessorTest(TestCase):
    """Tests for verifying the `ModernBertTextClassifierPreprocessor`."""

    def setUp(self):
        self.vocab = [
            "[PAD]",
            "[MASK]",
            "[CLS]",
            "[SEP]",
            "t",
            "h",
            "e",
            "q",
            "u",
            "i",
            "c",
            "k",
            "b",
            "r",
            "o",
            "w",
            "n",
            "f",
            "x",
            "th",
            "qu",
            "qui",
            "ck",
            "br",
            "ow",
            "wn",
            "own",
            "the",
            "quick",
            "brown",
            "fox",
        ]

        self.vocab_dict = {token: i for i, token in enumerate(self.vocab)}

        self.merges = [
            "t h",
            "q u",
            "qu i",
            "c k",
            "b r",
            "o w",
            "w n",
            "th e",
            "qui ck",
            "br own",
        ]

        self.tokenizer = ModernBertTokenizer(
            vocabulary=self.vocab_dict,
            merges=self.merges,
        )

        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 12,
        }

        self.input_data = ["the quick brown fox"]

    def test_preprocessor_basics(self):
        preprocessor = ModernBertTextClassifierPreprocessor(**self.init_kwargs)
        x = preprocessor(self.input_data)

        self.assertEqual(x["token_ids"].shape, (1, 12))
        self.assertEqual(x["padding_mask"].shape, (1, 12))

    def test_no_segment_ids(self):
        """ModernBERT has no segment embeddings, so the key is stripped."""
        preprocessor = ModernBertTextClassifierPreprocessor(**self.init_kwargs)
        x = preprocessor(self.input_data)

        self.assertNotIn("segment_ids", x)

    def test_labels_and_sample_weight_pass_through(self):
        preprocessor = ModernBertTextClassifierPreprocessor(**self.init_kwargs)
        x, y, sample_weight = preprocessor(
            self.input_data,
            y=[1],
            sample_weight=[1.0],
        )

        self.assertNotIn("segment_ids", x)
        self.assertAllEqual(y, [1])
        self.assertAllClose(sample_weight, [1.0])

    def test_sequence_boundary_tokens(self):
        """Sequences are wrapped in `[CLS]` / `[SEP]`."""
        preprocessor = ModernBertTextClassifierPreprocessor(**self.init_kwargs)
        x = preprocessor(self.input_data)

        token_ids = [int(i) for i in x["token_ids"][0]]

        self.assertEqual(token_ids[0], self.tokenizer.start_token_id)
        self.assertIn(self.tokenizer.end_token_id, token_ids)

    def test_serialization(self):
        preprocessor = ModernBertTextClassifierPreprocessor(**self.init_kwargs)

        config = preprocessor.get_config()
        restored = ModernBertTextClassifierPreprocessor.from_config(config)

        self.assertEqual(
            restored.sequence_length,
            preprocessor.sequence_length,
        )
        self.assertEqual(restored.truncate, preprocessor.truncate)
        self.assertIsInstance(restored.tokenizer, ModernBertTokenizer)
