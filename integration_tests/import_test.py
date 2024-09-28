import unittest

import keras_hub


class ImportTest(unittest.TestCase):
    def test_version(self):
        self.assertIsNotNone(keras_hub.__version__)
