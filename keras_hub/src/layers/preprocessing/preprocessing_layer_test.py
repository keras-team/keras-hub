import numpy as np
import pytest

from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_hub.src.tests.test_case import TestCase

try:
    import grain
except ImportError:
    grain = None


class LabelsLayer(PreprocessingLayer):
    """A layer with the `(x, y, sample_weight)` call signature."""

    def __init__(self, **kwargs):
        super().__init__(_allow_python_workflow=True, **kwargs)

    def call(self, x, y=None, sample_weight=None):
        return {"x": x, "y": y, "sample_weight": sample_weight}


class FeaturesLayer(PreprocessingLayer):
    """A layer that takes a single input and no labels."""

    def __init__(self, **kwargs):
        super().__init__(_allow_python_workflow=True, **kwargs)

    def call(self, inputs):
        return {"inputs": inputs}


class PreprocessingLayerTest(TestCase):
    def setUp(self):
        self.layer = LabelsLayer()

    def test_dict_element(self):
        element = {"text": "the quick brown fox", "images": np.ones((2, 2))}
        output = self.layer(element)
        self.assertIs(output["x"], element)
        self.assertIsNone(output["y"])
        self.assertIsNone(output["sample_weight"])

    def test_string_element(self):
        output = self.layer("the quick brown fox")
        self.assertEqual(output["x"], "the quick brown fox")
        self.assertIsNone(output["y"])
        self.assertIsNone(output["sample_weight"])

    def test_x_y_tuple(self):
        output = self.layer(("the quick brown fox", 1))
        self.assertEqual(output["x"], "the quick brown fox")
        self.assertEqual(output["y"], 1)
        self.assertIsNone(output["sample_weight"])

    def test_x_y_sample_weight_tuple(self):
        output = self.layer(("the quick brown fox", 1, 0.5))
        self.assertEqual(output["x"], "the quick brown fox")
        self.assertEqual(output["y"], 1)
        self.assertEqual(output["sample_weight"], 0.5)

    def test_tuple_of_features_x_with_explicit_labels(self):
        # A tuple `x` (e.g. multiple text segments) must not be unpacked when
        # `y` or `sample_weight` are passed explicitly.
        x = ("first segment", "second segment")
        output = self.layer(x, 1)
        self.assertIs(output["x"], x)
        self.assertEqual(output["y"], 1)
        output = self.layer(x, y=1)
        self.assertIs(output["x"], x)
        self.assertEqual(output["y"], 1)
        output = self.layer(x, sample_weight=0.5)
        self.assertIs(output["x"], x)
        self.assertIsNone(output["y"])
        self.assertEqual(output["sample_weight"], 0.5)
        output = self.layer(x=x)
        self.assertIs(output["x"], x)
        self.assertIsNone(output["y"])

    def test_tuple_of_features_x_without_labels_signature(self):
        # Layers that do not accept labels never unpack tuples.
        layer = FeaturesLayer()
        x = ("first segment", "second segment")
        output = layer(x)
        self.assertIs(output["inputs"], x)
        x = ("first segment", "second segment", "third segment")
        output = layer(x)
        self.assertIs(output["inputs"], x)

    def test_long_tuple_and_list_not_unpacked(self):
        x = ("a", "b", "c", "d")
        output = self.layer(x)
        self.assertIs(output["x"], x)
        self.assertIsNone(output["y"])
        # Lists are data, not `(x, y)` structures.
        x = ["the quick brown fox", "the earth is round"]
        output = self.layer(x)
        self.assertIs(output["x"], x)
        self.assertIsNone(output["y"])

    @pytest.mark.skipif(grain is None, reason="grain is not installed")
    def test_grain_map_over_tuple_elements(self):
        elements = [
            ("the quick brown fox", 1, 0.5),
            ("the earth is round", 0, 1.0),
        ]
        ds = grain.MapDataset.source(elements).map(self.layer)
        outputs = list(ds)
        self.assertEqual(outputs[0]["x"], "the quick brown fox")
        self.assertEqual(outputs[0]["y"], 1)
        self.assertEqual(outputs[0]["sample_weight"], 0.5)
        self.assertEqual(outputs[1]["x"], "the earth is round")
        self.assertEqual(outputs[1]["y"], 0)
        self.assertEqual(outputs[1]["sample_weight"], 1.0)
        # Dict elements pass through as `x`.
        elements = [{"text": "the quick brown fox", "label": 1}]
        (output,) = list(grain.MapDataset.source(elements).map(self.layer))
        self.assertEqual(output["x"], elements[0])
        self.assertIsNone(output["y"])
