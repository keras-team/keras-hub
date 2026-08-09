from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.coco.coco_utils import coco_id_to_name
from keras_hub.src.utils.coco.coco_utils import coco_name_to_id


class CocoUtilsTest(TestCase):
    def test_coco_id_to_name(self):
        self.assertEqual(coco_id_to_name(0), "unlabeled")
        self.assertEqual(coco_id_to_name(24), "zebra")
        with self.assertRaises(KeyError):
            coco_id_to_name(2001)

    def test_coco_name_to_id(self):
        self.assertEqual(coco_name_to_id("unlabeled"), 0)
        self.assertEqual(coco_name_to_id("zebra"), 24)
        with self.assertRaises(KeyError):
            coco_name_to_id("whirligig")
