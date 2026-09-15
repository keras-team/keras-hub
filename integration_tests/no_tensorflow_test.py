import unittest

import numpy as np

import keras_hub


class NoTensorflow(unittest.TestCase):
    def test_backbone_works(self):
        backbone = keras_hub.models.BertBackbone.from_preset(
            "bert_tiny_en_uncased",
        )
        backbone.predict(
            {
                "token_ids": np.ones((4, 128)),
                "padding_mask": np.ones((4, 128)),
                "segment_ids": np.ones((4, 128)),
            }
        )

    def test_tokenizer_works(self):
        # `WordPieceTokenizer` has a pure Python path, so `BertTokenizer` no
        # longer requires tensorflow-text. This previously asserted that
        # `from_preset` raised "pip install tensorflow-text".
        tokenizer = keras_hub.models.BertTokenizer.from_preset(
            "bert_tiny_en_uncased",
        )
        outputs = np.array(tokenizer("the quick brown fox"))
        self.assertEqual(outputs.ndim, 1)
        self.assertGreater(outputs.shape[0], 0)
        self.assertEqual(outputs.dtype, np.int32)
        # Round trip: detokenize and verify we recover the original text.
        decoded = tokenizer.detokenize(outputs)
        self.assertEqual(decoded, "the quick brown fox")
