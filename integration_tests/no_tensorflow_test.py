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

    def test_tokenizer_errors(self):
        with self.assertRaises(Exception) as e:
            keras_hub.models.BertTokenizer.from_preset(
                "bert_tiny_en_uncased",
            )
            self.assertTrue("pip install tensorflow-text" in e.exception)
