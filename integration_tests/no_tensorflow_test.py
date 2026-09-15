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

    def test_preprocessor_works(self):
        # Preprocessing layers default to the pure Python path, so model
        # preprocessors can be built and run without tensorflow-text.
        preprocessor = (
            keras_hub.models.BertTextClassifierPreprocessor.from_preset(
                "bert_tiny_en_uncased",
                sequence_length=8,
            )
        )
        x, y = preprocessor(["the quick brown fox"], [1])
        self.assertEqual(np.shape(x["token_ids"]), (1, 8))
        self.assertEqual(np.shape(x["padding_mask"]), (1, 8))
        self.assertEqual(np.shape(x["segment_ids"]), (1, 8))
        self.assertEqual(np.asarray(y).tolist(), [1])

    def test_tf_workflow_errors(self):
        # Explicitly requesting the TensorFlow path raises an informative
        # error on first use rather than at construction time.
        tokenizer = keras_hub.models.BertTokenizer.from_preset(
            "bert_tiny_en_uncased",
            _allow_python_workflow=False,
        )
        with self.assertRaisesRegex(ImportError, "pip install tensorflow-text"):
            tokenizer("the quick brown fox")
