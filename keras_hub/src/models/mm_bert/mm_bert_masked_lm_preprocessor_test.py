from keras_hub.src.models.mm_bert import mm_bert_masked_lm_preprocessor
from keras_hub.src.models.mm_bert.mm_bert_tokenizer import MMBertTokenizer
from keras_hub.src.tests.test_case import TestCase


class MMBertMaskedLMPreprocessorTest(TestCase):
    """Tests for verifying the `MMBertMaskedLMPreprocessor`."""

    def setUp(self):
        self.vocab = [
            "<pad>",
            "<mask>",
            "<bos>",
            "<eos>",
            "<unk>",
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

        self.vocab_dict = {w: i for i, w in enumerate(self.vocab)}
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

        self.tokenizer = MMBertTokenizer(
            vocabulary=self.vocab_dict,
            merges=self.merges,
        )

        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 12,
            "mask_selection_length": 4,
        }

        self.input_data = ["the quick brown fox"]

    def test_preprocessor_basics(self):
        """Verify the preprocessor forward pass."""
        test_init_kwargs = self.init_kwargs.copy()
        test_init_kwargs["mask_selection_rate"] = 0.0

        preprocessor = (
            mm_bert_masked_lm_preprocessor.MMBertMaskedLMPreprocessor
        )(**test_init_kwargs)
        x, y, sample_weight = preprocessor(self.input_data)

        self.assertEqual(x["token_ids"].shape, (1, 12))
        self.assertEqual(x["padding_mask"].shape, (1, 12))
        self.assertEqual(x["mask_positions"].shape, (1, 4))
        self.assertEqual(y.shape, (1, 4))
        self.assertEqual(sample_weight.shape, (1, 4))

        # mmBERT, like ModernBERT, does not use segment IDs.
        self.assertNotIn("segment_ids", x)

    def test_no_masking_zero_rate(self):
        """Verify that zero mask selection rate produces no masked tokens."""
        preprocessor = (
            mm_bert_masked_lm_preprocessor.MMBertMaskedLMPreprocessor
        )(
            tokenizer=self.tokenizer,
            mask_selection_rate=0.0,
            mask_selection_length=4,
            sequence_length=12,
        )

        _, _, sample_weight = preprocessor(self.input_data)

        self.assertAllClose(
            sample_weight,
            [[0.0, 0.0, 0.0, 0.0]],
        )

    def test_serialization(self):
        """Verify that the preprocessor can be serialized and restored."""
        preprocessor = (
            mm_bert_masked_lm_preprocessor.MMBertMaskedLMPreprocessor
        )(**self.init_kwargs)

        config = preprocessor.get_config()
        restored = (
            mm_bert_masked_lm_preprocessor.MMBertMaskedLMPreprocessor
        ).from_config(config)

        self.assertEqual(
            restored.sequence_length,
            preprocessor.sequence_length,
        )
        self.assertEqual(
            restored.mask_selection_rate,
            preprocessor.mask_selection_rate,
        )
        self.assertEqual(
            restored.mask_selection_length,
            preprocessor.mask_selection_length,
        )
        self.assertEqual(
            restored.mask_token_rate,
            preprocessor.mask_token_rate,
        )
        self.assertEqual(
            restored.random_token_rate,
            preprocessor.random_token_rate,
        )
        self.assertIsInstance(
            restored.tokenizer,
            MMBertTokenizer,
        )
