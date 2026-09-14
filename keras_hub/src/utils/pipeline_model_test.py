import os
import warnings

import keras
import numpy as np
import tensorflow as tf

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils import pipeline_model
from keras_hub.src.utils.pipeline_model import PipelineModel
from keras_hub.src.utils.pipeline_model import _build_dataset
from keras_hub.src.utils.pipeline_model import _convert_inputs_to_dataset
from keras_hub.src.utils.pipeline_model import _silence_unknown_length_warning

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


class RaggedOutputPipeline(PipelineModel):
    """This model preprocesses to a ragged tensor, as a tokenizer with no
    `sequence_length` does."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1)

    def preprocess_samples(self, x, y=None, sample_weight=None):
        # Runs the same in the `tf.data` graph and eagerly under grain.
        x = tf.RaggedTensor.from_tensor(tf.convert_to_tensor(x))
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def call(self, inputs):
        return self.dense(inputs)


class _BatchCounter(keras.callbacks.Callback):
    """Records how many batches each epoch actually trained on."""

    def __init__(self):
        super().__init__()
        self.per_epoch = []

    def on_epoch_begin(self, epoch, logs=None):
        self.per_epoch.append(0)

    def on_train_batch_end(self, batch, logs=None):
        self.per_epoch[-1] += 1


class _ConstantSource:
    """A random access source of constant samples, for `grain.DataLoader`."""

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            "x": np.ones((5,), dtype="float32"),
            "y": np.ones((1,), dtype="float32"),
        }


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

    def test_ragged_preprocessing_falls_back_to_tf_data(self):
        # Grain has no ragged type, so a ragged batch comes back as nested
        # lists that `GrainDatasetAdapter` flattens into scalars and rejects.
        # `tf.data` carries it end to end, so the pipeline goes there.
        x = np.random.uniform(size=(8, 5))
        y = np.random.uniform(size=(8, 1))
        ds = _build_dataset(
            x, y, None, 4, RaggedOutputPipeline().preprocess_samples
        )
        self.assertIsInstance(ds, tf.data.Dataset)

        if keras.config.backend() == "torch":
            # Keras' torch iterator rejects a ragged batch with `Invalid
            # dtype: object` whatever produced it, so `tf.data` cannot carry
            # one there either, on master included. The routing above is what
            # this test owns.
            self.skipTest("The torch iterator does not take ragged batches.")
        model = RaggedOutputPipeline()
        model.compile(loss="mse")
        model.fit(x=x, y=y, batch_size=4)
        model.evaluate(x=x, y=y, batch_size=4)
        model.predict(x=x, batch_size=4)

    def test_dense_preprocessing_stays_on_grain(self):
        ds = _build_dataset(
            np.random.uniform(size=(8, 5)),
            np.random.uniform(size=(8, 1)),
            None,
            4,
            NoopPipeline().preprocess_samples,
        )
        self.assertIsInstance(ds, grain.MapDataset)

    def test_error_grain_data_loader(self):
        # `grain.DataLoader` has no `map`, so preprocessing cannot be applied.
        loader = grain.DataLoader(
            data_source=_ConstantSource(8),
            sampler=grain.samplers.IndexSampler(
                num_records=8, shuffle=False, num_epochs=1
            ),
            operations=[grain.transforms.Batch(4)],
        )
        model = FeaturePipeline()
        model.compile(loss="mse")
        with self.assertRaisesRegex(ValueError, "grain.MapDataset"):
            model.fit(loader)
        with self.assertRaisesRegex(ValueError, "grain.MapDataset"):
            model.evaluate(loader)
        with self.assertRaisesRegex(ValueError, "grain.MapDataset"):
            model.predict(loader)
        with self.assertRaisesRegex(ValueError, "grain.MapDataset"):
            model.fit(
                x=np.random.uniform(size=(8, 5)),
                y=np.random.uniform(size=(8, 1)),
                batch_size=4,
                validation_data=loader,
            )

    def test_no_spurious_ran_out_of_data_warning(self):
        # `GrainDatasetAdapter.num_batches` is `None`, so Keras finds the
        # epoch size by running the iterator dry. Before
        # keras-team/keras#23360 that warned as if training was cut short.
        model = NoopPipeline()
        model.compile(loss="mse")
        x = np.random.uniform(size=(8, 5))
        y = np.random.uniform(size=(8, 1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x=x, y=y, batch_size=4, epochs=2)
            model.evaluate(x=x, y=y, batch_size=4)
            model.predict(x=x, batch_size=4)
        messages = [str(w.message) for w in caught]
        self.assertFalse([m for m in messages if "ran out of data" in m])

    def test_ran_out_of_data_warning_with_declared_steps(self):
        # The warning does carry information when the caller said how many
        # steps to expect, on any axis, so it is left alone in that case.
        ds = _convert_inputs_to_dataset(
            np.random.uniform(size=(8, 5)), None, None, batch_size=4
        )

        def caught_warnings(steps):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with _silence_unknown_length_warning(ds, steps):
                    warnings.warn("Your input ran out of data", UserWarning)
            return caught

        self.assertLen(caught_warnings((None, None)), 0)
        self.assertLen(caught_warnings((10, None)), 1)
        # A `validation_steps` shortfall says nothing about the training data,
        # but it is still the caller declaring a count.
        self.assertLen(caught_warnings((None, 10)), 1)

    def test_validation_steps_shortfall_still_warns(self):
        model = NoopPipeline()
        model.compile(loss="mse")
        x = np.random.uniform(size=(8, 5))
        y = np.random.uniform(size=(8, 1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(
                x=x,
                y=y,
                batch_size=4,
                validation_data=(x, y),
                validation_steps=100,
            )
        messages = [str(w.message) for w in caught]
        self.assertTrue([m for m in messages if "ran out of data" in m])

    def test_steps_per_epoch_keeps_every_epoch_fed(self):
        # `num_batches` is `None` for a grain dataset, so `EpochIterator` holds
        # one iterator across epochs. Without a repeat the second epoch would
        # train on a drained one, `[2, 0, 2]` instead of `[2, 2, 2]`.
        model = NoopPipeline()
        model.compile(loss="mse")
        counter = _BatchCounter()
        model.fit(
            x=np.random.uniform(size=(8, 5)),
            y=np.random.uniform(size=(8, 1)),
            batch_size=4,
            epochs=3,
            steps_per_epoch=2,
            callbacks=[counter],
        )
        self.assertEqual(counter.per_epoch, [2, 2, 2])

    def test_class_weight(self):
        # `GrainDatasetAdapter` takes no `class_weight`, so it is folded into
        # `sample_weight` instead of reaching the adapter.
        model = NoopPipeline()
        model.compile(loss="mse")
        model.fit(
            x=np.random.uniform(size=(8, 5)),
            y=np.random.randint(0, 2, size=(8, 1)),
            batch_size=4,
            class_weight={0: 1.0, 1: 2.0},
        )

    def test_error_class_weight_with_sample_weight(self):
        model = NoopPipeline()
        model.compile(loss="mse")
        with self.assertRaisesRegex(ValueError, "at the same time"):
            model.fit(
                x=np.random.uniform(size=(8, 5)),
                y=np.random.randint(0, 2, size=(8, 1)),
                sample_weight=np.ones((8,)),
                batch_size=4,
                class_weight={0: 1.0, 1: 2.0},
            )

    def test_iter_dataset_input(self):
        ds = _convert_inputs_to_dataset(
            np.random.uniform(size=(8, 5)),
            np.random.uniform(size=(8, 1)),
            None,
            batch_size=4,
        ).to_iter_dataset()
        self.assertIsInstance(ds, grain.IterDataset)
        model = NoopPipeline()
        model.compile(loss="mse")
        model.fit(ds)
        model.evaluate(ds)
        model.predict(ds)

    def test_python_number_dtypes_match_tf_data(self):
        # `from_tensor_slices` reads python numbers the way tf does. Numpy
        # widens them, and torch picks MPS on Apple Silicon, which has no
        # float64.
        ds = _convert_inputs_to_dataset(
            [[0.1, 0.9], [0.2, 0.8]], [1, 0], None, batch_size=2
        )
        x, y = next(iter(ds))
        self.assertEqual(x.dtype, "float32")
        self.assertEqual(y.dtype, "int32")

    def test_given_dtypes_are_left_alone(self):
        ds = _convert_inputs_to_dataset(
            np.zeros((2, 2), dtype="float64"), None, None, batch_size=2
        )
        self.assertEqual(next(iter(ds)).dtype, "float64")


class TestTfDataFallback(TestCase):
    """The path taken on a machine with no grain installed.

    CI installs grain, so without this the `tf.data` branch of
    `_convert_inputs_to_dataset` and its unbatched-input error never run.
    """

    def setUp(self):
        super().setUp()
        self._grain = pipeline_model.grain
        pipeline_model.grain = None

    def tearDown(self):
        pipeline_model.grain = self._grain
        super().tearDown()

    def test_builds_a_tf_dataset(self):
        ds = _convert_inputs_to_dataset(
            np.random.uniform(size=(8, 5)),
            np.random.uniform(size=(8, 1)),
            None,
            batch_size=4,
        )
        self.assertIsInstance(ds, tf.data.Dataset)

    def test_fit_evaluate_predict(self):
        model = NoopPipeline()
        model.compile(loss="mse")
        x = np.random.uniform(size=(8, 5))
        y = np.random.uniform(size=(8, 1))
        model.fit(x=x, y=y, batch_size=4)
        model.evaluate(x=x, y=y, batch_size=4)
        model.predict(x=x, batch_size=4)

    def test_fit_with_validation_data(self):
        model = NoopPipeline()
        model.compile(loss="mse")
        x = np.random.uniform(size=(8, 5))
        y = np.random.uniform(size=(8, 1))
        model.fit(x=x, y=y, batch_size=4, validation_data=(x, y))

    def test_class_weight(self):
        # `TFDatasetAdapter` applies this one itself.
        model = NoopPipeline()
        model.compile(loss="mse")
        model.fit(
            x=np.random.uniform(size=(8, 5)),
            y=np.random.randint(0, 2, size=(8, 1)),
            batch_size=4,
            class_weight={0: 1.0, 1: 2.0},
        )

    def test_unbatched_input_raises(self):
        # The rank 0 message `tf.data` raises is remapped to our own.
        model = FeaturePipeline()
        with self.assertRaisesRegex(ValueError, "must have a batch dimension"):
            model.fit(x=tf.constant("test"))


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
