from unittest import mock

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils import tensor_utils

try:
    import grain
except ImportError:
    grain = None


class LabelsLayer(PreprocessingLayer):
    """A layer with the `(x, y, sample_weight)` call signature."""

    def call(self, x, y=None, sample_weight=None):
        return {"x": x, "y": y, "sample_weight": sample_weight}


class FeaturesLayer(PreprocessingLayer):
    """A layer that takes a single input and no labels."""

    def call(self, inputs):
        return {"inputs": inputs}


class DeviceLayer(PreprocessingLayer):
    """A layer whose output is a backend tensor created inside `call`."""

    def call(self, inputs):
        return ops.ones((2,)) * inputs


def _device_type(x):
    """The device type a backend tensor lives on: `"cpu"`, `"gpu"`, ..."""
    if keras.backend.backend() == "tensorflow":
        # "/job:localhost/replica:0/task:0/device:CPU:0"
        return x.device.rsplit("device:", 1)[-1].split(":")[0].lower()
    if keras.backend.backend() == "jax":
        return next(iter(x.devices())).platform
    if keras.backend.backend() == "torch":
        return x.device.type
    raise NotImplementedError(keras.backend.backend())


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

    def test_python_workflow_is_the_default(self):
        layer = LabelsLayer()
        self.assertTrue(layer._allow_python_workflow)
        self.assertFalse(layer._use_tf_workflow())
        layer = LabelsLayer(_allow_python_workflow=False)
        self.assertFalse(layer._allow_python_workflow)
        self.assertTrue(layer._use_tf_workflow())

    def test_tf_workflow_inside_tf_function(self):
        layer = LabelsLayer()
        ds = tf.data.Dataset.from_tensors(0)
        ds = ds.map(lambda _: tf.constant(layer._use_tf_workflow()))
        self.assertTrue(ds.get_single_element())

    def test_tf_libs_checked_on_first_tf_workflow_use(self):
        # Without tensorflow-text, layers can still be constructed and run on
        # the Python path. The TensorFlow path raises an informative error on
        # first use rather than at construction time.
        with mock.patch.object(tensor_utils, "tf_text", None):
            layer = LabelsLayer()
            self.assertFalse(layer._use_tf_workflow())
            layer = LabelsLayer(_allow_python_workflow=False)
            with self.assertRaisesRegex(ImportError, "tensorflow-text"):
                layer._use_tf_workflow()

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

    @pytest.mark.skipif(grain is None, reason="grain is not installed")
    def test_grain_runs_on_cpu(self):
        # Grain maps on a pool of threads and pickles NumPy outputs, so the
        # ops must run on the CPU even when an accelerator is the default
        # device. On the TensorFlow backend concurrent eager GPU ops from
        # several threads can also mix up their inputs.
        layer = DeviceLayer()
        (output,) = list(grain.MapDataset.source([1.0]).map(layer))
        self.assertEqual(_device_type(output), "cpu")
