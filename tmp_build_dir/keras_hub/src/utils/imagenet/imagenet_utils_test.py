import numpy as np

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.imagenet.imagenet_utils import (
    decode_imagenet_predictions,
)
from keras_hub.src.utils.imagenet.imagenet_utils import imagenet_id_to_name
from keras_hub.src.utils.imagenet.imagenet_utils import imagenet_name_to_id


class ImageNetUtilsTest(TestCase):
    def test_imagenet_id_to_name(self):
        self.assertEqual(imagenet_id_to_name(0), "tench")
        self.assertEqual(imagenet_id_to_name(21), "kite")
        with self.assertRaises(KeyError):
            imagenet_id_to_name(2001)

    def test_imagenet_name_to_id(self):
        self.assertEqual(imagenet_name_to_id("tench"), 0)
        self.assertEqual(imagenet_name_to_id("kite"), 21)
        with self.assertRaises(KeyError):
            imagenet_name_to_id(2001)

    def test_decode_imagenet_predictions(self):
        preds = np.array(
            [
                [0.0] * 997 + [0.5, 0.3, 0.2],
                [0.0] * 997 + [0.2, 0.3, 0.5],
            ]
        )
        labels = decode_imagenet_predictions(preds, top=3)
        self.assertEqual(
            labels,
            [
                [("bolete", 0.5), ("ear", 0.3), ("toilet_tissue", 0.2)],
                [("toilet_tissue", 0.5), ("ear", 0.3), ("bolete", 0.2)],
            ],
        )
        labels = decode_imagenet_predictions(
            preds, top=3, include_synset_ids=True
        )
        self.assertEqual(
            labels,
            [
                [
                    ("n13054560", "bolete", 0.5),
                    ("n13133613", "ear", 0.3),
                    ("n15075141", "toilet_tissue", 0.2),
                ],
                [
                    ("n15075141", "toilet_tissue", 0.5),
                    ("n13133613", "ear", 0.3),
                    ("n13054560", "bolete", 0.2),
                ],
            ],
        )
