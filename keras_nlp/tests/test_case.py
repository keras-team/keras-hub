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

import tensorflow as tf
import tree
from absl.testing import parameterized
from keras_core.backend import is_float_dtype
from keras_core.backend import standardize_dtype

from keras_nlp.backend import config
from keras_nlp.backend import keras
from keras_nlp.backend import ops


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
    if ops.is_tensor(x):
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
        input_dtype = standardize_dtype(x.dtype)
        super().assertEqual(input_dtype, expected_dtype, msg=msg)

    def run_layer_test(
        self,
        layer_cls,
        init_kwargs,
        input_data,
        expected_output_shape,
        expected_output_data=None,
        expected_num_trainable_weights=0,
        expected_num_non_trainable_weights=0,
        expected_num_non_trainable_variables=0,
        run_training_check=True,
        run_mixed_precision_check=True,
    ):
        # Serialization test.
        layer = layer_cls(**init_kwargs)
        self.run_class_serialization_test(layer)

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
                standardize_dtype(layer.dtype),
                standardize_dtype(output_dtype),
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

            model = TestModel(layer)
            model.compile(optimizer="sgd", loss="mse", jit_compile=True)
            model.fit(input_data, output_data, verbose=0)

        if config.multi_backend():
            # Build test.
            layer = layer_cls(**init_kwargs)
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
            layer = layer_cls(**init_kwargs)
            if isinstance(keras_tensor_inputs, dict):
                keras_tensor_outputs = layer(**keras_tensor_inputs)
            else:
                keras_tensor_outputs = layer(keras_tensor_inputs)
            run_build_asserts(layer)
            run_output_asserts(layer, keras_tensor_outputs)

        # Eager call test and compiled training test.
        layer = layer_cls(**init_kwargs)
        if isinstance(input_data, dict):
            output_data = layer(**input_data)
        else:
            output_data = layer(input_data)
        run_output_asserts(layer, output_data, eager=True)

        if run_training_check:
            run_training_step(layer, input_data, output_data)

        # Never test mixed precision on torch CPU. Torch lacks support.
        if run_mixed_precision_check and config.backend() == "torch":
            import torch

            run_mixed_precision_check = torch.cuda.is_available()

        if run_mixed_precision_check:
            layer = layer_cls(**{**init_kwargs, "dtype": "mixed_float16"})
            if isinstance(input_data, dict):
                output_data = layer(**input_data)
            else:
                output_data = layer(input_data)
            for tensor in tree.flatten(output_data):
                if is_float_dtype(tensor.dtype):
                    self.assertDTypeEqual(tensor, "float16")
            for weight in layer.weights:
                if is_float_dtype(weight.dtype):
                    self.assertDTypeEqual(weight, "float32")

    def run_class_serialization_test(self, instance):
        # get_config roundtrip
        cls = instance.__class__
        cfg = instance.get_config()
        cfg_json = json.dumps(cfg, sort_keys=True, indent=4)
        ref_dir = dir(instance)[:]
        revived_instance = cls.from_config(cfg)
        revived_cfg = revived_instance.get_config()
        revived_cfg_json = json.dumps(revived_cfg, sort_keys=True, indent=4)
        self.assertEqual(cfg_json, revived_cfg_json)
        # Dir tests only work on keras-core.
        if config.multi_backend():
            self.assertEqual(ref_dir, dir(revived_instance))

        # serialization roundtrip
        serialized = keras.saving.serialize_keras_object(instance)
        serialized_json = json.dumps(serialized, sort_keys=True, indent=4)
        revived_instance = keras.saving.deserialize_keras_object(
            json.loads(serialized_json)
        )
        revived_cfg = revived_instance.get_config()
        revived_cfg_json = json.dumps(revived_cfg, sort_keys=True, indent=4)
        self.assertEqual(cfg_json, revived_cfg_json)
        # Dir tests only work on keras-core.
        if config.multi_backend():
            new_dir = dir(revived_instance)[:]
            for lst in [ref_dir, new_dir]:
                if "__annotations__" in lst:
                    lst.remove("__annotations__")
            self.assertEqual(ref_dir, new_dir)
