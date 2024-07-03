# Copyright 2023 The KerasNLP Authors
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

import json
import os
import pathlib
import re

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use `keras_nlp`, please install Tensorflow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )
import keras
from absl.testing import parameterized
from keras import ops
from keras import tree

from keras_nlp.src.tokenizers.tokenizer import Tokenizer
from keras_nlp.src.utils.tensor_utils import is_float_dtype


def convert_to_comparible_type(x):
    """Convert tensors to comparable types.

    Any string are converted to plain python types. Any jax or torch tensors
    are converted to numpy.
    """
    if getattr(x, "dtype", None) == tf.string:
        if isinstance(x, tf.RaggedTensor):
            x = x.to_list()
        if isinstance(x, tf.Tensor):
            x = x.numpy() if x.shape.rank == 0 else x.numpy().tolist()
        return tree.map_structure(lambda x: x.decode("utf-8"), x)
    if isinstance(x, (tf.Tensor, tf.RaggedTensor)):
        return x
    if hasattr(x, "__array__"):
        return ops.convert_to_numpy(x)
    return x


class TestCase(tf.test.TestCase, parameterized.TestCase):
    """Base test case class for KerasNLP."""

    def assertAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        # This metric dict hack is only needed for tf.keras, and can be
        # removed after we fully migrate to keras-core/Keras 3.
        if x1.__class__.__name__ == "_MetricDict":
            x1 = dict(x1)
        if x2.__class__.__name__ == "_MetricDict":
            x2 = dict(x2)
        x1 = tree.map_structure(convert_to_comparible_type, x1)
        x2 = tree.map_structure(convert_to_comparible_type, x2)
        super().assertAllClose(x1, x2, atol=atol, rtol=rtol, msg=msg)

    def assertEqual(self, x1, x2, msg=None):
        x1 = tree.map_structure(convert_to_comparible_type, x1)
        x2 = tree.map_structure(convert_to_comparible_type, x2)
        super().assertEqual(x1, x2, msg=msg)

    def assertAllEqual(self, x1, x2, msg=None):
        x1 = tree.map_structure(convert_to_comparible_type, x1)
        x2 = tree.map_structure(convert_to_comparible_type, x2)
        super().assertAllEqual(x1, x2, msg=msg)

    def assertDTypeEqual(self, x, expected_dtype, msg=None):
        input_dtype = keras.backend.standardize_dtype(x.dtype)
        super().assertEqual(input_dtype, expected_dtype, msg=msg)

    def run_layer_test(
        self,
        cls,
        init_kwargs,
        input_data,
        expected_output_shape,
        expected_output_data=None,
        expected_num_trainable_weights=0,
        expected_num_non_trainable_weights=0,
        expected_num_non_trainable_variables=0,
        run_training_check=True,
        run_precision_checks=True,
    ):
        """Run basic tests for a modeling layer."""
        # Serialization test.
        layer = cls(**init_kwargs)
        self.run_serialization_test(layer)

        def run_build_asserts(layer):
            self.assertTrue(layer.built)
            self.assertLen(
                layer.trainable_weights,
                expected_num_trainable_weights,
                msg="Unexpected number of trainable_weights",
            )
            self.assertLen(
                layer.non_trainable_weights,
                expected_num_non_trainable_weights,
                msg="Unexpected number of non_trainable_weights",
            )
            self.assertLen(
                layer.non_trainable_variables,
                expected_num_non_trainable_variables,
                msg="Unexpected number of non_trainable_variables",
            )

        def run_output_asserts(layer, output, eager=False):
            output_shape = tree.map_structure(
                lambda x: None if x is None else x.shape, output
            )
            self.assertEqual(
                expected_output_shape,
                output_shape,
                msg="Unexpected output shape",
            )
            output_dtype = tree.flatten(output)[0].dtype
            self.assertEqual(
                keras.backend.standardize_dtype(layer.dtype),
                keras.backend.standardize_dtype(output_dtype),
                msg="Unexpected output dtype",
            )
            if eager and expected_output_data is not None:
                self.assertAllClose(expected_output_data, output)

        def run_training_step(layer, input_data, output_data):
            class TestModel(keras.Model):
                def __init__(self, layer):
                    super().__init__()
                    self.layer = layer

                def call(self, x):
                    if isinstance(x, dict):
                        return self.layer(**x)
                    else:
                        return self.layer(x)

            input_data = tree.map_structure(
                lambda x: ops.convert_to_numpy(x), input_data
            )
            output_data = tree.map_structure(
                lambda x: ops.convert_to_numpy(x), output_data
            )
            model = TestModel(layer)
            # Temporarily disable jit compilation on torch backend.
            jit_compile = keras.config.backend() != "torch"
            model.compile(optimizer="sgd", loss="mse", jit_compile=jit_compile)
            model.fit(input_data, output_data, verbose=0)

        # Build test.
        layer = cls(**init_kwargs)
        if isinstance(input_data, dict):
            shapes = {k + "_shape": v.shape for k, v in input_data.items()}
            layer.build(**shapes)
        else:
            layer.build(input_data.shape)
        run_build_asserts(layer)

        # Symbolic call test.
        keras_tensor_inputs = tree.map_structure(
            lambda x: keras.KerasTensor(x.shape, x.dtype), input_data
        )
        layer = cls(**init_kwargs)
        if isinstance(keras_tensor_inputs, dict):
            keras_tensor_outputs = layer(**keras_tensor_inputs)
        else:
            keras_tensor_outputs = layer(keras_tensor_inputs)
        run_build_asserts(layer)
        run_output_asserts(layer, keras_tensor_outputs)

        # Eager call test and compiled training test.
        layer = cls(**init_kwargs)
        if isinstance(input_data, dict):
            output_data = layer(**input_data)
        else:
            output_data = layer(input_data)
        run_output_asserts(layer, output_data, eager=True)

        if run_training_check:
            run_training_step(layer, input_data, output_data)

        if run_precision_checks:
            self.run_precision_test(cls, init_kwargs, input_data)

    def run_preprocessing_layer_test(
        self,
        cls,
        init_kwargs,
        input_data,
        expected_output=None,
        expected_detokenize_output=None,
    ):
        """Run basic tests for a preprocessing layer."""
        layer = cls(**init_kwargs)
        # Check serialization (without a full save).
        self.run_serialization_test(layer)

        ds = tf.data.Dataset.from_tensor_slices(input_data)

        # Run with direct call.
        if isinstance(input_data, tuple):
            # Mimic tf.data unpacking behavior for preprocessing layers.
            output = layer(*input_data)
        else:
            output = layer(input_data)

        # For tokenizers only, also check detokenize.
        if isinstance(layer, Tokenizer):
            if not expected_detokenize_output:
                expected_detokenize_output = input_data
            detokenize_output = layer.detokenize(output)
            self.assertAllEqual(detokenize_output, expected_detokenize_output)

        # Run with an unbatched dataset.
        output_ds = ds.map(layer).ragged_batch(1_000)
        self.assertAllClose(output, output_ds.get_single_element())

        # Run with a batched dataset.
        output_ds = ds.batch(1_000).map(layer)
        self.assertAllClose(output, output_ds.get_single_element())

        if expected_output:
            self.assertAllClose(output, expected_output)

    def run_preprocessor_test(
        self,
        cls,
        init_kwargs,
        input_data,
        expected_output=None,
        expected_detokenize_output=None,
        token_id_key="token_ids",
    ):
        """Run basic tests for a Model Preprocessor layer."""
        self.run_preprocessing_layer_test(
            cls,
            init_kwargs,
            input_data,
            expected_output=expected_output,
            expected_detokenize_output=expected_detokenize_output,
        )

        layer = cls(**self.init_kwargs)
        if isinstance(input_data, tuple):
            output = layer(*input_data)
        else:
            output = layer(input_data)
        output, _, _ = keras.utils.unpack_x_y_sample_weight(output)
        shape = ops.shape(output[token_id_key])
        self.assertEqual(shape[-1], layer.sequence_length)
        # Update the sequence length.
        layer.sequence_length = 17
        if isinstance(input_data, tuple):
            output = layer(*input_data)
        else:
            output = layer(input_data)
        output, _, _ = keras.utils.unpack_x_y_sample_weight(output)
        shape = ops.shape(output[token_id_key])
        self.assertEqual(shape[-1], 17)

    def run_serialization_test(self, instance):
        """Check idempotency of serialize/deserialize.

        Not this is a much faster test than saving."""
        run_dir_test = (
            not keras.config.backend() == "tensorflow"
            or not isinstance(instance, Tokenizer)
        )
        # get_config roundtrip
        cls = instance.__class__
        cfg = instance.get_config()
        cfg_json = json.dumps(cfg, sort_keys=True, indent=4)
        ref_dir = dir(instance)[:]
        revived_instance = cls.from_config(cfg)
        revived_cfg = revived_instance.get_config()
        revived_cfg_json = json.dumps(revived_cfg, sort_keys=True, indent=4)
        self.assertEqual(cfg_json, revived_cfg_json)
        if run_dir_test:
            self.assertEqual(set(ref_dir), set(dir(revived_instance)))

        # serialization roundtrip
        serialized = keras.saving.serialize_keras_object(instance)
        serialized_json = json.dumps(serialized, sort_keys=True, indent=4)
        revived_instance = keras.saving.deserialize_keras_object(
            json.loads(serialized_json)
        )
        revived_cfg = revived_instance.get_config()
        revived_cfg_json = json.dumps(revived_cfg, sort_keys=True, indent=4)
        self.assertEqual(cfg_json, revived_cfg_json)
        if run_dir_test:
            new_dir = dir(revived_instance)[:]
            for lst in [ref_dir, new_dir]:
                if "__annotations__" in lst:
                    lst.remove("__annotations__")
            self.assertEqual(set(ref_dir), set(new_dir))

    def run_precision_test(self, cls, init_kwargs, input_data):
        # Never test mixed precision on torch CPU. Torch lacks support.
        if keras.config.backend() == "torch":
            import torch

            if not torch.cuda.is_available():
                return

        for policy in ["mixed_float16", "mixed_bfloat16", "bfloat16"]:
            policy = keras.mixed_precision.Policy(policy)
            layer = cls(**{**init_kwargs, "dtype": policy})
            if isinstance(layer, keras.Model):
                output_data = layer(input_data)
            elif isinstance(input_data, dict):
                output_data = layer(**input_data)
            else:
                output_data = layer(input_data)
            for tensor in tree.flatten(output_data):
                if is_float_dtype(tensor.dtype):
                    self.assertDTypeEqual(tensor, policy.compute_dtype)
            for weight in layer.weights:
                if is_float_dtype(weight.dtype):
                    self.assertDTypeEqual(weight, policy.variable_dtype)
            for sublayer in layer._flatten_layers():
                if isinstance(sublayer, keras.layers.Softmax):
                    continue
                if isinstance(sublayer, keras.layers.InputLayer):
                    continue
                self.assertEqual(policy.compute_dtype, sublayer.compute_dtype)
                self.assertEqual(policy.variable_dtype, sublayer.variable_dtype)

    def run_quantization_test(self, cls, init_kwargs, input_data):
        policy = keras.DTypePolicy("float32")
        for mode in ["int8", "float8"]:
            layer = cls(**{**init_kwargs, "dtype": policy})
            layer.quantize(mode)
            # Try eager call
            if isinstance(layer, keras.Model):
                _ = layer(input_data)
            elif isinstance(input_data, dict):
                _ = layer(**input_data)
            else:
                _ = layer(input_data)
            # Verify sublayer's dtype policy
            for sublayer in layer._flatten_layers():
                if type(sublayer) is keras.layers.Dense:
                    self.assertEqual(
                        f"{mode}_from_float32", sublayer.dtype_policy.name
                    )
            # Try saving and reloading the model
            temp_filepath = os.path.join(self.get_temp_dir(), "layer.keras")
            layer.save(temp_filepath)
            reloaded_layer = keras.models.load_model(temp_filepath)
            self.assertAllClose(layer(input_data), reloaded_layer(input_data))

    def run_model_saving_test(
        self,
        cls,
        init_kwargs,
        input_data,
    ):
        """Save and load a model from disk and assert output is unchanged."""
        model = cls(**init_kwargs)
        model_output = model(input_data)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, cls)

        # Check that output matches.
        restored_output = restored_model(input_data)
        self.assertAllClose(model_output, restored_output)

    def run_backbone_test(
        self,
        cls,
        init_kwargs,
        input_data,
        expected_output_shape,
        variable_length_data=None,
        run_mixed_precision_check=True,
        run_quantization_check=True,
    ):
        """Run basic tests for a backbone, including compilation."""
        backbone = cls(**init_kwargs)
        # Check serialization (without a full save).
        self.run_serialization_test(backbone)

        # Call model eagerly.
        output = backbone(input_data)
        if isinstance(expected_output_shape, dict):
            for key in expected_output_shape:
                self.assertEqual(output[key].shape, expected_output_shape[key])
        else:
            self.assertEqual(output.shape, expected_output_shape)

        # Check we can embed tokens eagerly.
        output = backbone.token_embedding(ops.zeros((2, 3), dtype="int32"))

        # Check variable length sequences.
        if variable_length_data is None:
            # If no variable length data passed, assume the second axis of all
            # inputs is our sequence axis and create it ourselves.
            variable_length_data = [
                tree.map_structure(lambda x: x[:, :seq_length, ...], input_data)
                for seq_length in (2, 3, 4)
            ]
        for batch in variable_length_data:
            backbone(batch)

        # Check compiled predict function.
        backbone.predict(input_data)
        # Convert to numpy first, torch GPU tensor -> tf.data will error.
        numpy_data = tree.map_structure(ops.convert_to_numpy, input_data)
        # Create a dataset.
        input_dataset = tf.data.Dataset.from_tensor_slices(numpy_data).batch(2)
        backbone.predict(input_dataset)

        # Check name maps to classname.
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", cls.__name__)
        name = re.sub("([a-z])([A-Z])", r"\1_\2", name).lower()
        self.assertRegexpMatches(backbone.name, name)

        # Check mixed precision.
        if run_mixed_precision_check:
            self.run_precision_test(cls, init_kwargs, input_data)

        # Check quantization.
        if run_quantization_check:
            self.run_quantization_test(cls, init_kwargs, input_data)

    def run_task_test(
        self,
        cls,
        init_kwargs,
        train_data,
        expected_output_shape=None,
        batch_size=2,
    ):
        """Run basic tests for a backbone, including compilation."""
        task = cls(**init_kwargs)
        # Check serialization (without a full save).
        self.run_serialization_test(task)
        preprocessor = task.preprocessor
        ds = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
        x, y, sw = keras.utils.unpack_x_y_sample_weight(train_data)

        # Test predict.
        output = task.predict(x)
        if expected_output_shape is not None:
            output_shape = tree.map_structure(lambda x: x.shape, output)
            self.assertAllClose(output_shape, expected_output_shape)
        # With a dataset.
        output_ds = task.predict(ds)
        self.assertAllClose(output, output_ds)
        # With split preprocessing.
        task.preprocessor = None
        output_split = task.predict(ds.map(preprocessor))
        task.preprocessor = preprocessor
        self.assertAllClose(output, output_split)

        # Test fit.
        task.fit(x, y, sample_weight=sw)
        # With a dataset.
        task.fit(ds)
        # With split preprocessing.
        task.preprocessor = None
        task.fit(ds.map(preprocessor))
        task.preprocessor = preprocessor

    def run_preset_test(
        self,
        cls,
        preset,
        input_data,
        init_kwargs={},
        expected_output=None,
        expected_output_shape=None,
        expected_partial_output=None,
    ):
        """Run instantiation and a forward pass for a preset."""
        with self.assertRaises(Exception):
            cls.from_preset("clowntown", **init_kwargs)

        instance = cls.from_preset(preset, **init_kwargs)

        if isinstance(input_data, tuple):
            # Mimic tf.data unpacking behavior for preprocessing layers.
            output = instance(*input_data)
        else:
            output = instance(input_data)

        if isinstance(instance, keras.Model):
            instance = cls.from_preset(
                preset, load_weights=False, **init_kwargs
            )
            instance(input_data)

        if expected_output is not None:
            self.assertAllClose(output, expected_output)

        if expected_output_shape is not None:
            output_shape = tree.map_structure(lambda x: x.shape, output)
            self.assertAllClose(output_shape, expected_output_shape)

        if expected_partial_output is not None:
            # Allow passing a partial output snippet of the last dimension.
            # We want check stability, but the full output would be too long.
            def compare(actual, expected):
                expected = ops.convert_to_numpy(expected)
                self.assertEqual(len(expected.shape), 1)
                actual = ops.reshape(actual, (-1,))[: expected.shape[0]]
                self.assertAllClose(actual, expected, atol=0.01, rtol=0.01)

            tree.map_structure(compare, output, expected_partial_output)

    def get_test_data_dir(self):
        return str(pathlib.Path(__file__).parent / "test_data")
