import os

import keras
import numpy as np
import tensorflow as tf

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.pipeline_model import PipelineModel
from keras_hub.src.utils.pipeline_model import _convert_inputs_to_dataset

try:
    import grain
except ImportError:
    grain = None


class NoopPipeline(PipelineModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)


class FeaturePipeline(PipelineModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1)

    def preprocess_samples(self, x, y=None, sample_weight=None):
        x = tf.strings.to_number(x)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def call(self, inputs):
        return self.dense(inputs)


class LabelPipeline(PipelineModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1)

    def preprocess_samples(self, x, y=None, sample_weight=None):
        if y is not None:
            y = tf.strings.to_number(y)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def call(self, inputs):
        return self.dense(inputs)


class DataPipeline(PipelineModel):
    """This model generates labels straight from the input data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1)

    def preprocess_samples(self, x, y=None, sample_weight=None):
        y = x = tf.strings.to_number(x)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def call(self, inputs):
        return self.dense(inputs)


class FunctionalPipeline(PipelineModel):
    def __init__(self, **kwargs):
        inputs = keras.Input(shape=(5,))
        outputs = keras.layers.Dense(1)(inputs)
        super().__init__(inputs, outputs, **kwargs)

    def preprocess_samples(self, x, y=None, sample_weight=None):
        x = tf.strings.to_number(x)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TestNoopPipelineModel(TestCase):
    def test_fit(self):
        x = np.random.uniform(size=(8, 5))
        y = np.random.uniform(size=(8, 1))
        sw = np.random.uniform(size=(8, 1))
        model = NoopPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.fit(x=x, y=y, sample_weight=sw, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.fit(x=x, y=y, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_evaluate(self):
        x = np.random.uniform(size=(8, 5))
        y = np.random.uniform(size=(8, 1))
        sw = np.random.uniform(size=(8, 1))
        model = NoopPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.evaluate(x=x, y=y, sample_weight=sw, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.evaluate(x=x, y=y, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_predict(self):
        x = np.random.uniform(size=(8, 5))
        model = NoopPipeline()
        model.compile(loss="mse")
        model.predict(x=x, batch_size=8)
        model.predict(tf.data.Dataset.from_tensor_slices(x).batch(8))

    def test_on_batch(self):
        x = np.random.uniform(size=(8, 5))
        y = np.random.uniform(size=(8, 1))
        sw = np.random.uniform(size=(8, 1))
        model = NoopPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.train_on_batch(x=x, y=y, sample_weight=sw)
        model.test_on_batch(x=x, y=y, sample_weight=sw)
        # Without sample weight.
        model.train_on_batch(x=x, y=y)
        model.test_on_batch(x=x, y=y)
        model.predict_on_batch(x=x)

    def test_saved_model(self):
        model = NoopPipeline()
        x = np.random.uniform(size=(8, 5))
        model_output = model.predict(x)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(
            path, custom_objects={"NoopPipeline": NoopPipeline}
        )

        # Check we got the real object back.
        self.assertIsInstance(restored_model, NoopPipeline)
        # Check that output matches.
        restored_output = restored_model.predict(x)
        self.assertAllClose(model_output, restored_output)


class TestFeaturePreprocessingModel(TestCase):
    def test_fit_with_preprocessing(self):
        x = tf.strings.as_string(np.random.uniform(size=(100, 5)))
        y = np.random.uniform(size=(100, 1))
        sw = np.random.uniform(size=(100, 1))
        model = FeaturePipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.fit(x=x, y=y, sample_weight=sw, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.fit(x=x, y=y, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_evaluate_with_preprocessing(self):
        x = tf.strings.as_string(np.random.uniform(size=(100, 5)))
        y = np.random.uniform(size=(100, 1))
        sw = np.random.uniform(size=(100, 1))
        model = FeaturePipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.evaluate(x=x, y=y, sample_weight=sw, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.evaluate(x=x, y=y, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_predict_with_preprocessing(self):
        x = tf.strings.as_string(np.random.uniform(size=(100, 5)))
        model = FeaturePipeline()
        model.compile(loss="mse")
        model.predict(x=x, batch_size=8)
        model.predict(tf.data.Dataset.from_tensor_slices(x).batch(8))

    def test_on_batch(self):
        x = tf.strings.as_string(np.random.uniform(size=(8, 5)))
        y = np.random.uniform(size=(8, 1))
        sw = np.random.uniform(size=(8, 1))
        model = FeaturePipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.train_on_batch(x=x, y=y, sample_weight=sw)
        model.test_on_batch(x=x, y=y, sample_weight=sw)
        # Without sample weight.
        model.train_on_batch(x=x, y=y)
        model.test_on_batch(x=x, y=y)
        model.predict_on_batch(x=x)

    def test_saved_model(self):
        model = FeaturePipeline()
        x = tf.strings.as_string(np.random.uniform(size=(8, 5)))
        model_output = model.predict(x)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(
            path, custom_objects={"FeaturePipeline": FeaturePipeline}
        )

        # Check we got the real object back.
        self.assertIsInstance(restored_model, FeaturePipeline)
        # Check that output matches.
        restored_output = restored_model.predict(x)
        self.assertAllClose(model_output, restored_output)


class TestLabelPreprocessingModel(TestCase):
    def test_fit_with_preprocessing(self):
        x = np.random.uniform(size=(100, 5))
        y = tf.strings.as_string(np.random.uniform(size=(100, 1)))
        sw = np.random.uniform(size=(100, 1))
        model = LabelPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.fit(x=x, y=y, sample_weight=sw, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.fit(x=x, y=y, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_evaluate_with_preprocessing(self):
        x = np.random.uniform(size=(100, 5))
        y = tf.strings.as_string(np.random.uniform(size=(100, 1)))
        sw = np.random.uniform(size=(100, 1))
        model = LabelPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.evaluate(x=x, y=y, sample_weight=sw, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.evaluate(x=x, y=y, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_predict_with_preprocessing(self):
        x = np.random.uniform(size=(100, 5))
        model = LabelPipeline()
        model.compile(loss="mse")
        model.predict(x=x, batch_size=8)
        model.predict(tf.data.Dataset.from_tensor_slices(x).batch(8))

    def test_on_batch(self):
        x = np.random.uniform(size=(8, 5))
        y = tf.strings.as_string(np.random.uniform(size=(8, 1)))
        sw = np.random.uniform(size=(8, 1))
        model = LabelPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.train_on_batch(x=x, y=y, sample_weight=sw)
        model.test_on_batch(x=x, y=y, sample_weight=sw)
        # Without sample weight.
        model.train_on_batch(x=x, y=y)
        model.test_on_batch(x=x, y=y)
        model.predict_on_batch(x=x)

    def test_saved_model(self):
        model = LabelPipeline()
        x = np.random.uniform(size=(8, 5))
        model_output = model.predict(x)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(
            path, custom_objects={"LabelPipeline": LabelPipeline}
        )

        # Check we got the real object back.
        self.assertIsInstance(restored_model, LabelPipeline)
        # Check that output matches.
        restored_output = restored_model.predict(x)
        self.assertAllClose(model_output, restored_output)


class TestDataPreprocessingModel(TestCase):
    def test_fit_with_preprocessing(self):
        data = tf.strings.as_string(np.random.uniform(size=(100, 1)))
        model = DataPipeline()
        model.compile(loss="mse")
        model.fit(x=data, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices(data).batch(8))

    def test_evaluate_with_preprocessing(self):
        data = tf.strings.as_string(np.random.uniform(size=(100, 1)))
        model = DataPipeline()
        model.compile(loss="mse")
        model.evaluate(x=data, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices(data).batch(8))

    def test_predict_with_preprocessing(self):
        x = tf.strings.as_string(np.random.uniform(size=(100, 1)))
        model = DataPipeline()
        model.compile(loss="mse")
        model.predict(x=x, batch_size=8)
        model.predict(tf.data.Dataset.from_tensor_slices(x).batch(8))

    def test_on_batch(self):
        data = tf.strings.as_string(np.random.uniform(size=(8, 1)))
        model = DataPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.train_on_batch(x=data)
        model.test_on_batch(x=data)
        # Without sample weight.
        model.train_on_batch(x=data)
        model.test_on_batch(x=data)
        model.predict_on_batch(x=data)

    def test_saved_model(self):
        model = DataPipeline()
        data = tf.strings.as_string(np.random.uniform(size=(8, 1)))
        model_output = model.predict(data)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(
            path, custom_objects={"DataPipeline": DataPipeline}
        )

        # Check we got the real object back.
        self.assertIsInstance(restored_model, DataPipeline)
        # Check that output matches.
        restored_output = restored_model.predict(data)
        self.assertAllClose(model_output, restored_output)


class TestFunctional(TestCase):
    def test_fit(self):
        x = tf.strings.as_string(np.random.uniform(size=(100, 5)))
        y = np.random.uniform(size=(100, 1))
        sw = np.random.uniform(size=(100, 1))

        model = FunctionalPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.fit(x=x, y=y, sample_weight=sw, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.fit(x=x, y=y, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_saved_model(self):
        model = FunctionalPipeline()
        x = tf.strings.as_string(np.random.uniform(size=(8, 5)))
        model_output = model.predict(x)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(
            path, custom_objects={"FunctionalPipeline": FunctionalPipeline}
        )

        # Check we got the real object back.
        self.assertIsInstance(restored_model, FunctionalPipeline)
        # Check that output matches.
        restored_output = restored_model.predict(x)
        self.assertAllClose(model_output, restored_output)


class TestFitArguments(TestCase):
    def test_validation_data(self):
        x = tf.strings.as_string(np.random.uniform(size=(80, 5)))
        y = np.random.uniform(size=(80, 1))
        val_x = tf.strings.as_string(np.random.uniform(size=(20, 5)))
        val_y = np.random.uniform(size=(20, 1))

        model = FeaturePipeline()
        model.compile(loss="mse")

        model.fit(x=x, y=y, validation_data=(val_x, val_y), batch_size=8)
        model.fit(
            x=tf.data.Dataset.from_tensor_slices((x, y)).batch(8),
            validation_data=tf.data.Dataset.from_tensor_slices(
                (val_x, val_y)
            ).batch(8),
        )

    def test_validation_split(self):
        x = tf.strings.as_string(np.random.uniform(size=(100, 5)))
        y = np.random.uniform(size=(100, 1))

        model = FeaturePipeline()
        model.compile(loss="mse")

        model.fit(x=x, y=y, validation_split=0.2, batch_size=8)

    def test_error_dataset_and_invalid_arguments(self):
        x = tf.strings.as_string(np.random.uniform(size=(100, 5)))
        y = np.random.uniform(size=(100, 1))
        sw = np.random.uniform(size=(100, 1))
        ds = tf.data.Dataset.from_tensor_slices((x, y))

        model = FeaturePipeline()
        model.compile(loss="mse")
        with self.assertRaises(ValueError):
            model.fit(ds, validation_split=0.2)
        with self.assertRaises(ValueError):
            model.fit(ds, batch_size=0.2)
        with self.assertRaises(ValueError):
            model.fit(ds, y=y)
        with self.assertRaises(ValueError):
            model.fit(ds, sample_weight=sw)


class NumpyPipeline(PipelineModel):
    """This model preprocesses with numpy only, never TensorFlow."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1)
        self.preprocess_count = 0

    def preprocess_samples(self, x, y=None, sample_weight=None):
        self.preprocess_count += 1
        x = np.asarray(x, dtype="float32") / 255.0
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def call(self, inputs):
        return self.dense(inputs)


class TestGrainPipeline(TestCase):
    def setUp(self):
        super().setUp()
        if grain is None:
            self.skipTest("Grain is not installed.")

    def test_builds_a_grain_dataset(self):
        x = np.random.uniform(size=(8, 5))
        y = np.random.uniform(size=(8, 1))
        ds = _convert_inputs_to_dataset(x, y, None, batch_size=4)
        self.assertIsInstance(ds, grain.MapDataset)
        self.assertLen(ds, 2)

    def test_passes_through_a_grain_dataset(self):
        x = np.random.uniform(size=(8, 5))
        ds = _convert_inputs_to_dataset(x, None, None, batch_size=4)
        self.assertIs(_convert_inputs_to_dataset(ds), ds)

    def test_error_grain_dataset_and_invalid_arguments(self):
        x = np.random.uniform(size=(8, 5))
        y = np.random.uniform(size=(8, 1))
        ds = _convert_inputs_to_dataset(x, None, None, batch_size=4)
        model = FeaturePipeline()
        model.compile(loss="mse")
        with self.assertRaises(ValueError):
            model.fit(ds, y=y)
        with self.assertRaises(ValueError):
            model.fit(ds, sample_weight=y)
        with self.assertRaises(ValueError):
            model.fit(ds, batch_size=4)

    def test_python_string_list_input(self):
        # A `list` enumerates the samples of a single string input.
        x = [[str(v) for v in row] for row in np.random.uniform(size=(8, 5))]
        y = np.random.uniform(size=(8, 1))
        model = FeaturePipeline()
        model.compile(loss="mse")
        model.fit(x=x, y=y, batch_size=4)
        model.evaluate(x=x, y=y, batch_size=4)
        model.predict(x=x, batch_size=4)

    def test_numpy_only_preprocessing(self):
        x = np.random.uniform(0, 255, size=(8, 5))
        y = np.random.uniform(size=(8, 1))
        model = NumpyPipeline()
        model.compile(loss="mse")
        model.fit(x=x, y=y, batch_size=4)
        model.evaluate(x=x, y=y, batch_size=4)
        model.predict(x=x, batch_size=4)

    def test_fit_with_validation_data(self):
        # `fit()` preprocesses `validation_data` and `evaluate()` reuses the
        # iterator Keras caches from it.
        x = np.random.uniform(0, 255, size=(8, 5))
        y = np.random.uniform(size=(8, 1))
        model = NumpyPipeline()
        model.compile(loss="mse")
        history = model.fit(
            x=x, y=y, validation_data=(x, y), batch_size=4, epochs=3, verbose=0
        )
        self.assertLen(history.history["val_loss"], 3)
        self.assertAllClose(
            history.history["val_loss"][-1],
            model.evaluate(x=x, y=y, batch_size=4, verbose=0),
        )

    def test_dict_input(self):
        x = {
            "a": np.random.uniform(size=(8, 5)),
            "b": np.random.uniform(size=(8, 5)),
        }
        ds = _convert_inputs_to_dataset(x, None, None, batch_size=4)
        batch = ds[0]
        self.assertEqual(set(batch.keys()), {"a", "b"})
        self.assertEqual(batch["a"].shape, (4, 5))

    def test_mismatched_batch_dimension_raises(self):
        model = FeaturePipeline()
        model.compile(loss="mse")
        with self.assertRaisesRegex(ValueError, "same batch dimension"):
            model.fit(
                x=np.random.uniform(size=(8, 5)),
                y=np.random.uniform(size=(4, 1)),
                batch_size=4,
            )

    def test_ragged_input_falls_back_to_tf_data(self):
        x = tf.ragged.constant([[1, 2, 3], [4, 5]])
        ds = _convert_inputs_to_dataset(x, None, None, batch_size=2)
        self.assertIsInstance(ds, tf.data.Dataset)


class TestInputErrors(TestCase):
    def test_unbatched_input_raises(self):
        model = FeaturePipeline()
        with self.assertRaisesRegex(ValueError, "must have a batch dimension"):
            model.fit(x=tf.constant("test"))
        with self.assertRaisesRegex(ValueError, "must have a batch dimension"):
            model.fit(x=tf.constant(["test"]), y=tf.constant(0))
        with self.assertRaisesRegex(ValueError, "must have a batch dimension"):
            model.fit(
                x=tf.constant(["test"]), y=tf.constant([0]), sample_weight=0.0
            )
        with self.assertRaisesRegex(ValueError, "must have a batch dimension"):
            model.fit(x="test")
