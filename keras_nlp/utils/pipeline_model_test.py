# Copyright 2022 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.utils.pipeline_model import PipelineModel


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

    def preprocess_features(self, x):
        return tf.strings.to_number(x)

    def call(self, inputs):
        return self.dense(inputs)


class LabelPipeline(PipelineModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1)

    def preprocess_labels(self, y):
        return tf.strings.to_number(y)

    def call(self, inputs):
        return self.dense(inputs)


class DataPipeline(PipelineModel):
    """This model generates labels straight from the input data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1)

    def preprocess_samples(self, x, y=None, sample_weight=None):
        return tf.strings.to_number(x), tf.strings.to_number(x), sample_weight

    def call(self, inputs):
        return self.dense(inputs)


class FunctionalPipeline(PipelineModel):
    def __init__(self, **kwargs):
        inputs = keras.Input(shape=(5,))
        outputs = keras.layers.Dense(1)(inputs)
        super().__init__(inputs, outputs, **kwargs)

    def preprocess_features(self, x):
        return tf.strings.to_number(x)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TestNoopPipelineModel(tf.test.TestCase, parameterized.TestCase):
    def test_fit(self):
        x = tf.random.uniform((8, 5))
        y = tf.random.uniform((8, 1))
        sw = tf.random.uniform((8, 1))
        model = NoopPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.fit(x=x, y=y, sample_weight=sw, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.fit(x=x, y=y, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_evaluate(self):
        x = tf.random.uniform((8, 5))
        y = tf.random.uniform((8, 1))
        sw = tf.random.uniform((8, 1))
        model = NoopPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.evaluate(x=x, y=y, sample_weight=sw, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.evaluate(x=x, y=y, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_predict(self):
        x = tf.random.uniform((8, 5))
        model = NoopPipeline()
        model.compile(loss="mse")
        model.predict(x=x, batch_size=8)
        model.predict(tf.data.Dataset.from_tensor_slices(x).batch(8))

    def test_on_batch(self):
        x = tf.random.uniform((8, 5))
        y = tf.random.uniform((8, 1))
        sw = tf.random.uniform((8, 1))
        model = NoopPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.train_on_batch(x=x, y=y, sample_weight=sw)
        model.test_on_batch(x=x, y=y, sample_weight=sw)
        # Without sample weight.
        model.train_on_batch(x=x, y=y)
        model.test_on_batch(x=x, y=y)
        model.predict_on_batch(x=x)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model = NoopPipeline()
        x = tf.random.uniform((8, 5))
        model_output = model.predict(x)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(
            save_path, custom_objects={"NoopPipeline": NoopPipeline}
        )

        # Check we got the real object back.
        self.assertIsInstance(restored_model, NoopPipeline)
        # Check that output matches.
        restored_output = restored_model.predict(x)
        self.assertAllClose(model_output, restored_output)


class TestFeaturePreprocessingModel(tf.test.TestCase, parameterized.TestCase):
    def test_fit_with_preprocessing(self):
        x = tf.strings.as_string(tf.random.uniform((100, 5)))
        y = tf.random.uniform((100, 1))
        sw = tf.random.uniform((100, 1))
        model = FeaturePipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.fit(x=x, y=y, sample_weight=sw, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.fit(x=x, y=y, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_fit_no_preprocessing(self):
        x = tf.random.uniform((100, 5))
        y = tf.random.uniform((100, 1))
        sw = tf.random.uniform((100, 1))
        model = FeaturePipeline(include_preprocessing=False)
        model.compile(loss="mse")
        # With sample weight.
        model.fit(x=x, y=y, sample_weight=sw, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.fit(x=x, y=y, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_evaluate_with_preprocessing(self):
        x = tf.strings.as_string(tf.random.uniform((100, 5)))
        y = tf.random.uniform((100, 1))
        sw = tf.random.uniform((100, 1))
        model = FeaturePipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.evaluate(x=x, y=y, sample_weight=sw, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.evaluate(x=x, y=y, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_evaluate_no_preprocessing(self):
        x = tf.random.uniform((100, 5))
        y = tf.random.uniform((100, 1))
        sw = tf.random.uniform((100, 1))
        model = FeaturePipeline(include_preprocessing=False)
        model.compile(loss="mse")
        # With sample weight.
        model.evaluate(x=x, y=y, sample_weight=sw, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.evaluate(x=x, y=y, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_predict_with_preprocessing(self):
        x = tf.strings.as_string(tf.random.uniform((100, 5)))
        model = FeaturePipeline()
        model.compile(loss="mse")
        model.predict(x=x, batch_size=8)
        model.predict(tf.data.Dataset.from_tensor_slices(x).batch(8))

    def test_predict_no_preprocessing(self):
        x = tf.random.uniform((100, 5))
        model = FeaturePipeline(include_preprocessing=False)
        model.compile(loss="mse")
        model.predict(x=x, batch_size=8)
        model.predict(tf.data.Dataset.from_tensor_slices(x).batch(8))

    def test_on_batch(self):
        x = tf.strings.as_string(tf.random.uniform((8, 5)))
        y = tf.random.uniform((8, 1))
        sw = tf.random.uniform((8, 1))
        model = FeaturePipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.train_on_batch(x=x, y=y, sample_weight=sw)
        model.test_on_batch(x=x, y=y, sample_weight=sw)
        # Without sample weight.
        model.train_on_batch(x=x, y=y)
        model.test_on_batch(x=x, y=y)
        model.predict_on_batch(x=x)

    def test_on_batch_no_preprocessing(self):
        x = tf.random.uniform((8, 5))
        y = tf.random.uniform((8, 1))
        sw = tf.random.uniform((8, 1))
        model = FeaturePipeline(include_preprocessing=False)
        model.compile(loss="mse")
        # With sample weight.
        model.train_on_batch(x=x, y=y, sample_weight=sw)
        model.test_on_batch(x=x, y=y, sample_weight=sw)
        # Without sample weight.
        model.train_on_batch(x=x, y=y)
        model.test_on_batch(x=x, y=y)
        model.predict_on_batch(x=x)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model = FeaturePipeline()
        x = tf.strings.as_string(tf.random.uniform((8, 5)))
        model_output = model.predict(x)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(
            save_path, custom_objects={"FeaturePipeline": FeaturePipeline}
        )

        # Check we got the real object back.
        self.assertIsInstance(restored_model, FeaturePipeline)
        # Check that output matches.
        restored_output = restored_model.predict(x)
        self.assertAllClose(model_output, restored_output)


class TestLabelPreprocessingModel(tf.test.TestCase, parameterized.TestCase):
    def test_fit_with_preprocessing(self):
        x = tf.random.uniform((100, 5))
        y = tf.strings.as_string(tf.random.uniform((100, 1)))
        sw = tf.random.uniform((100, 1))
        model = LabelPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.fit(x=x, y=y, sample_weight=sw, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.fit(x=x, y=y, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_fit_no_preprocessing(self):
        x = tf.random.uniform((100, 5))
        y = tf.random.uniform((100, 1))
        sw = tf.random.uniform((100, 1))
        model = LabelPipeline(include_preprocessing=False)
        model.compile(loss="mse")
        # With sample weight.
        model.fit(x=x, y=y, sample_weight=sw, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.fit(x=x, y=y, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_evaluate_with_preprocessing(self):
        x = tf.random.uniform((100, 5))
        y = tf.strings.as_string(tf.random.uniform((100, 1)))
        sw = tf.random.uniform((100, 1))
        model = LabelPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.evaluate(x=x, y=y, sample_weight=sw, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.evaluate(x=x, y=y, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_evaluate_no_preprocessing(self):
        x = tf.random.uniform((100, 5))
        y = tf.random.uniform((100, 1))
        sw = tf.random.uniform((100, 1))
        model = LabelPipeline(include_preprocessing=False)
        model.compile(loss="mse")
        # With sample weight.
        model.evaluate(x=x, y=y, sample_weight=sw, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.evaluate(x=x, y=y, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_predict_with_preprocessing(self):
        x = tf.random.uniform((100, 5))
        model = LabelPipeline()
        model.compile(loss="mse")
        model.predict(x=x, batch_size=8)
        model.predict(tf.data.Dataset.from_tensor_slices(x).batch(8))

    def test_on_batch(self):
        x = tf.random.uniform((8, 5))
        y = tf.strings.as_string(tf.random.uniform((8, 1)))
        sw = tf.random.uniform((8, 1))
        model = LabelPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.train_on_batch(x=x, y=y, sample_weight=sw)
        model.test_on_batch(x=x, y=y, sample_weight=sw)
        # Without sample weight.
        model.train_on_batch(x=x, y=y)
        model.test_on_batch(x=x, y=y)
        model.predict_on_batch(x=x)

    def test_on_batch_no_preprocessing(self):
        x = tf.random.uniform((8, 5))
        y = tf.random.uniform((8, 1))
        sw = tf.random.uniform((8, 1))
        model = LabelPipeline(include_preprocessing=False)
        model.compile(loss="mse")
        # With sample weight.
        model.train_on_batch(x=x, y=y, sample_weight=sw)
        model.test_on_batch(x=x, y=y, sample_weight=sw)
        # Without sample weight.
        model.train_on_batch(x=x, y=y)
        model.test_on_batch(x=x, y=y)
        model.predict_on_batch(x=x)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model = LabelPipeline()
        x = tf.random.uniform((8, 5))
        model_output = model.predict(x)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(
            save_path, custom_objects={"LabelPipeline": LabelPipeline}
        )

        # Check we got the real object back.
        self.assertIsInstance(restored_model, LabelPipeline)
        # Check that output matches.
        restored_output = restored_model.predict(x)
        self.assertAllClose(model_output, restored_output)


class TestDataPreprocessingModel(tf.test.TestCase, parameterized.TestCase):
    def test_fit_with_preprocessing(self):
        data = tf.strings.as_string(tf.random.uniform((100, 1)))
        model = DataPipeline()
        model.compile(loss="mse")
        model.fit(x=data, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices(data).batch(8))

    def test_fit_no_preprocessing(self):
        x = tf.random.uniform((100, 1))
        y = tf.random.uniform((100, 1))
        model = DataPipeline(include_preprocessing=False)
        model.compile(loss="mse")
        model.fit(x=x, y=y, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_evaluate_with_preprocessing(self):
        data = tf.strings.as_string(tf.random.uniform((100, 1)))
        model = DataPipeline()
        model.compile(loss="mse")
        model.evaluate(x=data, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices(data).batch(8))

    def test_evaluate_no_preprocessing(self):
        x = tf.random.uniform((100, 1))
        y = tf.random.uniform((100, 1))
        model = DataPipeline(include_preprocessing=False)
        model.compile(loss="mse")
        model.evaluate(x=x, y=y, batch_size=8)
        model.evaluate(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_predict_with_preprocessing(self):
        x = tf.strings.as_string(tf.random.uniform((100, 1)))
        model = DataPipeline()
        model.compile(loss="mse")
        model.predict(x=x, batch_size=8)
        model.predict(tf.data.Dataset.from_tensor_slices(x).batch(8))

    def test_predict_no_preprocessing(self):
        x = tf.random.uniform((100, 1))
        model = DataPipeline(include_preprocessing=False)
        model.compile(loss="mse")
        model.predict(x=x, batch_size=8)
        model.predict(tf.data.Dataset.from_tensor_slices(x).batch(8))

    def test_on_batch(self):
        data = tf.strings.as_string(tf.random.uniform((8, 1)))
        model = DataPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.train_on_batch(x=data)
        model.test_on_batch(x=data)
        # Without sample weight.
        model.train_on_batch(x=data)
        model.test_on_batch(x=data)
        model.predict_on_batch(x=data)

    def test_on_batch_no_preprocessing(self):
        x = tf.random.uniform((8, 1))
        y = tf.random.uniform((8, 1))
        sw = tf.random.uniform((8, 1))
        model = DataPipeline(include_preprocessing=False)
        model.compile(loss="mse")
        # With sample weight.
        model.train_on_batch(x=x, y=y, sample_weight=sw)
        model.test_on_batch(x=x, y=y, sample_weight=sw)
        # Without sample weight.
        model.train_on_batch(x=x, y=y)
        model.test_on_batch(x=x, y=y)
        model.predict_on_batch(x=x)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model = DataPipeline()
        data = tf.strings.as_string(tf.random.uniform((8, 1)))
        model_output = model.predict(data)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(
            save_path, custom_objects={"DataPipeline": DataPipeline}
        )

        # Check we got the real object back.
        self.assertIsInstance(restored_model, DataPipeline)
        # Check that output matches.
        restored_output = restored_model.predict(data)
        self.assertAllClose(model_output, restored_output)


class TestFunctional(tf.test.TestCase, parameterized.TestCase):
    def test_fit(self):
        x = tf.strings.as_string(tf.random.uniform((100, 5)))
        y = tf.random.uniform((100, 1))
        sw = tf.random.uniform((100, 1))

        model = FunctionalPipeline()
        model.compile(loss="mse")
        # With sample weight.
        model.fit(x=x, y=y, sample_weight=sw, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.fit(x=x, y=y, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    def test_fit_no_preprocessing(self):
        x = tf.random.uniform((100, 5))
        y = tf.random.uniform((100, 1))
        sw = tf.random.uniform((100, 1))
        model = FunctionalPipeline(include_preprocessing=False)
        model.compile(loss="mse")
        # With sample weight.
        model.fit(x=x, y=y, sample_weight=sw, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(8))
        # Without sample weight.
        model.fit(x=x, y=y, batch_size=8)
        model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(8))

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model = FunctionalPipeline()
        x = tf.strings.as_string(tf.random.uniform((8, 5)))
        model_output = model.predict(x)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(
            save_path, custom_objects={"FunctionalPipeline": FunctionalPipeline}
        )

        # Check we got the real object back.
        self.assertIsInstance(restored_model, FunctionalPipeline)
        # Check that output matches.
        restored_output = restored_model.predict(x)
        self.assertAllClose(model_output, restored_output)


class TestFitArguments(tf.test.TestCase):
    def test_validation_data(self):
        x = tf.strings.as_string(tf.random.uniform((80, 5)))
        y = tf.random.uniform((80, 1))
        val_x = tf.strings.as_string(tf.random.uniform((20, 5)))
        val_y = tf.random.uniform((20, 1))

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
        x = tf.strings.as_string(tf.random.uniform((100, 5)))
        y = tf.random.uniform((100, 1))

        model = FeaturePipeline()
        model.compile(loss="mse")

        model.fit(x=x, y=y, validation_split=0.2, batch_size=8)

    def test_error_dataset_and_invalid_arguments(self):
        x = tf.strings.as_string(tf.random.uniform((100, 5)))
        y = tf.random.uniform((100, 1))
        sw = tf.random.uniform((100, 1))
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


class TestInputErrors(tf.test.TestCase):
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
