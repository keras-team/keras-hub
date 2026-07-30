import gc
import io
import json
import os
import pathlib
import re
import tempfile

import keras
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras import ops
from keras import tree
from keras.layers import ReversibleEmbedding

from keras_hub.src.models.retinanet.feature_pyramid import FeaturePyramid
from keras_hub.src.tokenizers.tokenizer import Tokenizer
from keras_hub.src.utils.tensor_utils import is_float_dtype


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
    """Base test case class for KerasHub."""

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
                        if isinstance(layer, FeaturePyramid):
                            return self.layer(x)
                        else:
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
            if isinstance(layer, FeaturePyramid):
                layer.build(shapes)
            else:
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
            if isinstance(layer, FeaturePyramid):
                keras_tensor_outputs = layer(keras_tensor_inputs)
            else:
                keras_tensor_outputs = layer(**keras_tensor_inputs)
        else:
            keras_tensor_outputs = layer(keras_tensor_inputs)
        run_build_asserts(layer)
        run_output_asserts(layer, keras_tensor_outputs)

        # Eager call test and compiled training test.
        layer = cls(**init_kwargs)
        if isinstance(input_data, dict):
            if isinstance(layer, FeaturePyramid):
                output_data = layer(input_data)
            else:
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
        return_output=False,
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

        if return_output:
            return output

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
            # Ensure the correct `dtype` is set for sublayers or submodels in
            # `init_kwargs`.
            original_init_kwargs = init_kwargs.copy()
            for k, v in init_kwargs.items():
                if isinstance(v, keras.Layer):
                    config = v.get_config()
                    config["dtype"] = policy
                    init_kwargs[k] = v.__class__.from_config(config)
            layer = cls(**{**init_kwargs, "dtype": policy})
            if isinstance(layer, keras.Model):
                output_data = layer(input_data)
                output_spec = layer.compute_output_spec(input_data)
            elif isinstance(input_data, dict):
                if isinstance(layer, FeaturePyramid):
                    output_data = layer(input_data)
                    output_spec = layer.compute_output_spec(input_data)
                else:
                    output_data = layer(**input_data)
                    output_spec = layer.compute_output_spec(**input_data)
            else:
                output_data = layer(input_data)
                output_spec = layer.compute_output_spec(input_data)
            for tensor in tree.flatten(output_data):
                if is_float_dtype(tensor.dtype):
                    self.assertDTypeEqual(tensor, policy.compute_dtype)
            for spec in tree.flatten(output_spec):
                if is_float_dtype(spec.dtype):
                    self.assertDTypeEqual(spec, policy.compute_dtype)
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
            # Restore `init_kwargs`.
            init_kwargs = original_init_kwargs

    def run_quantization_test(self, instance, cls, init_kwargs, input_data):
        # TODO: revert the following if. This works around a torch
        # quantization failure in `MultiHeadAttention` with Keras 3.7.
        if keras.config.backend() == "torch":
            return

        def _get_supported_layers(mode):
            supported_layers = [keras.layers.Dense, keras.layers.EinsumDense]
            if mode == "int8":
                supported_layers.append(keras.layers.Embedding)
                supported_layers.append(ReversibleEmbedding)
            return supported_layers

        for mode in ["int8", "float8"]:
            # Manually configure DTypePolicyMap to avoid intensive computation
            # in `Model.quantize`.
            policy_map = keras.dtype_policies.DTypePolicyMap("float32")
            for layer in instance._flatten_layers():
                if type(layer) in _get_supported_layers(mode):
                    policy_map[layer.path] = keras.dtype_policies.get(
                        f"{mode}_from_float32"
                    )
            # Ensure the correct `dtype` is set for sublayers or submodels in
            # `init_kwargs`.
            original_init_kwargs = init_kwargs.copy()
            for k, v in init_kwargs.items():
                if isinstance(v, keras.Layer):
                    config = v.get_config()
                    config["dtype"] = policy_map
                    init_kwargs[k] = v.__class__.from_config(config)
            # Instantiate the layer.
            model = cls(**{**init_kwargs, "dtype": policy_map})
            # Call layer eagerly.
            if isinstance(model, keras.Model):
                _ = model(input_data)
            elif isinstance(input_data, dict):
                _ = model(**input_data)
            else:
                _ = model(input_data)
            # Verify sublayer's dtype policy.
            for sublayer in model._flatten_layers():
                if type(sublayer) in _get_supported_layers(mode):
                    self.assertEqual(mode, sublayer.quantization_mode)
            # `get_config` roundtrip.
            cfg = model.get_config()
            revived_model = cls.from_config(cfg)
            revived_cfg = revived_model.get_config()
            self.assertEqual(cfg, revived_cfg)
            # Check weights loading.
            weights = model.get_weights()
            revived_model.set_weights(weights)
            # Restore `init_kwargs`.
            init_kwargs = original_init_kwargs

    def run_model_saving_test(
        self,
        cls,
        init_kwargs,
        input_data,
        atol=0.000001,
        rtol=0.000001,
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
        self.assertAllClose(model_output, restored_output, atol=atol, rtol=rtol)

    def _verify_litert_outputs(
        self,
        keras_output,
        litert_output,
        sig_outputs,
        expected_output_shape=None,
        verify_numerics=True,
        comparison_mode="strict",
        output_thresholds=None,
    ):
        """Verify LiteRT outputs against expected shape and Keras outputs.

        Args:
            keras_output: Keras model output (can be None if not verifying
                numerics)
            litert_output: LiteRT interpreter output
            sig_outputs: Output names from SignatureDef
            expected_output_shape: Expected output shape (optional)
            verify_numerics: Whether to verify numerical correctness
            comparison_mode: "strict" or "statistical"
            output_thresholds: Thresholds for statistical comparison
        """
        # Handle single output case: if Keras has single output but LiteRT
        # returns dict
        if (
            not isinstance(keras_output, dict)
            and isinstance(litert_output, dict)
            and len(litert_output) == 1
        ):
            litert_output = list(litert_output.values())[0]

        # Verify output shape if specified
        if expected_output_shape is not None:
            self.assertEqual(litert_output.shape, expected_output_shape)

        # Verify numerical correctness if requested
        if verify_numerics:
            self._verify_litert_numerics(
                keras_output,
                litert_output,
                sig_outputs,
                output_thresholds,
                comparison_mode,
            )

    def _verify_litert_numerics(
        self,
        keras_output,
        litert_output,
        sig_outputs,
        output_thresholds,
        comparison_mode,
    ):
        """Verify numerical accuracy between Keras and LiteRT outputs.

        This method compares outputs using the SignatureDef output names to
        match Keras outputs with LiteRT outputs properly.

        Args:
            keras_output: Keras model output (tensor or dict)
            litert_output: LiteRT interpreter output (tensor or dict)
            sig_outputs: List of output names from SignatureDef
            output_thresholds: Dict of thresholds for comparison
            comparison_mode: "strict" or "statistical"
        """
        if isinstance(keras_output, dict) and isinstance(litert_output, dict):
            # Both outputs are dicts - compare using SignatureDef output names
            for output_name in sig_outputs:
                if output_name not in keras_output:
                    self.fail(
                        f"SignatureDef output '{output_name}' not found in "
                        f"Keras outputs.\n"
                        f"Keras keys: {list(keras_output.keys())}"
                    )
                if output_name not in litert_output:
                    self.fail(
                        f"SignatureDef output '{output_name}' not found in "
                        f"LiteRT outputs.\n"
                        f"LiteRT keys: {list(litert_output.keys())}"
                    )

                keras_val_np = ops.convert_to_numpy(keras_output[output_name])
                litert_val = litert_output[output_name]
                output_threshold = output_thresholds.get(
                    output_name,
                    output_thresholds.get("*", {"max": 10.0, "mean": 0.1}),
                )
                self._compare_outputs(
                    keras_val_np,
                    litert_val,
                    comparison_mode,
                    output_name,
                    output_threshold["max"],
                    output_threshold["mean"],
                )
        elif not isinstance(keras_output, dict) and not isinstance(
            litert_output, dict
        ):
            # Both outputs are single tensors - direct comparison
            keras_output_np = ops.convert_to_numpy(keras_output)
            output_threshold = output_thresholds.get(
                "*", {"max": 1e-2, "mean": 1e-3}
            )
            self._compare_outputs(
                keras_output_np,
                litert_output,
                comparison_mode,
                key=None,
                max_threshold=output_threshold["max"],
                mean_threshold=output_threshold["mean"],
            )
        else:
            keras_type = type(keras_output).__name__
            litert_type = type(litert_output).__name__
            self.fail(
                f"Output structure mismatch: Keras returns "
                f"{keras_type}, LiteRT returns {litert_type}"
            )

    @staticmethod
    def _build_litert_torch_input_signature(input_data):
        """Build a concrete input signature for torch-backend LiteRT export.

        The torch export path does not support dynamic shapes, so it needs a
        fully specified `keras.InputSpec` tree derived from the sample data.
        """
        dtype_map = {
            "float64": "float32",
            "int64": "int32",
        }

        def _to_spec(x):
            x = ops.convert_to_numpy(x)
            dtype = keras.backend.standardize_dtype(x.dtype)
            dtype = dtype_map.get(dtype, dtype)
            return keras.InputSpec(shape=x.shape, dtype=dtype)

        return [tree.map_structure(_to_spec, input_data)]

    @staticmethod
    def _map_litert_torch_inputs(converted_input_data, sig_inputs):
        """Map dict inputs to their torch-export signature input names.

        Depending on the litert-torch version, a flattened dict input is
        named either with the original key suffixed (`args_0_<key>`) or
        purely positionally (`args_0`, `args_1`, ...). Prefer an exact key
        match; otherwise fall back to positional order (the model's input
        definition order, which the test ``input_data`` mirrors).
        """
        keys = list(converted_input_data.keys())
        stripped = {re.sub(r"^args_\d+_", "", n): n for n in sig_inputs}
        if all(key in stripped for key in keys):
            return {stripped[key]: converted_input_data[key] for key in keys}

        def _index(name):
            match = re.search(r"\d+", name)
            return int(match.group()) if match else 0

        ordered = sorted(sig_inputs, key=_index)
        return {
            ordered[i]: converted_input_data[key] for i, key in enumerate(keys)
        }

    def run_litert_export_test(
        self,
        cls=None,
        init_kwargs=None,
        input_data=None,
        expected_output_shape=None,
        model=None,
        verify_numerics=True,
        # No LiteRT output in model saving test; remove undefined return
        output_thresholds=None,
        **export_kwargs,
    ):
        """Export model to LiteRT format and verify outputs.

        Args:
            cls: Model class to test (optional if model is provided)
            init_kwargs: Initialization arguments for the model (optional
                if model is provided)
            input_data: Input data to test with (dict or tensor)
            expected_output_shape: Expected output shape from LiteRT inference
            model: Pre-created model instance (optional, if provided cls and
                init_kwargs are ignored)
            verify_numerics: Whether to verify numerical correctness
                between Keras and LiteRT outputs. Set to False for preset
                models with load_weights=False where outputs are random.
            comparison_mode: "strict" (default) or "statistical".
                - "strict": All elements must be within default tolerances
                    (1e-6)
                - "statistical": Check mean/max absolute differences against
                    provided thresholds
            output_thresholds: Dict mapping output names to threshold dicts
                with "max" and "mean" keys. Use "*" as wildcard for defaults.
                Example: {"output1": {"max": 1e-4, "mean": 1e-5},
                         "*": {"max": 1e-3, "mean": 1e-4}}
            **export_kwargs: Additional keyword arguments to pass to
                model.export().
        """
        # Extract comparison_mode from export_kwargs if provided
        comparison_mode = export_kwargs.pop("comparison_mode", "strict")
        backend = keras.backend.backend()

        # The rewritten LiteRT export path currently runs on the PyTorch
        # backend only.
        if backend != "torch":
            self.skipTest(
                "LiteRT export is supported on the PyTorch backend only."
            )

        # The torch export path is provided by the optional litert-torch
        # package.
        try:
            import litert_torch  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            self.skipTest(
                "litert-torch is required for LiteRT export with the "
                "torch backend"
            )

        # Use the ai-edge-litert interpreter exclusively. The legacy
        # tf.lite.Interpreter is deprecated and removed in recent TensorFlow
        # releases, so we intentionally do not fall back to it.
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            self.skipTest(
                "LiteRT export tests require the 'ai-edge-litert' package."
            )

        if output_thresholds is None:
            output_thresholds = {"*": {"max": 10.0, "mean": 0.1}}

        if model is None:
            if cls is None or init_kwargs is None:
                raise ValueError(
                    "Either 'model' or 'cls' and 'init_kwargs' must be provided"
                )
            model = cls(**init_kwargs)
            _ = model(input_data)

        interpreter = None
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                export_path = os.path.join(temp_dir, "model.tflite")

                # The torch export path needs a concrete input signature, since
                # it does not support dynamic shapes.
                if "input_signature" not in export_kwargs:
                    export_kwargs["input_signature"] = (
                        self._build_litert_torch_input_signature(input_data)
                    )

                # Step 1: Export model and get Keras output
                model.export(export_path, format="litert", **export_kwargs)
                self.assertTrue(os.path.exists(export_path))
                self.assertGreater(os.path.getsize(export_path), 0)

                keras_output = model(input_data) if verify_numerics else None

                # Step 2: Load interpreter and verify SignatureDef
                interpreter = Interpreter(model_path=export_path)
                signature_defs = interpreter.get_signature_list()
                self.assertIn(
                    "serving_default",
                    signature_defs,
                    "Missing serving_default signature",
                )

                serving_sig = signature_defs["serving_default"]
                sig_inputs = serving_sig.get("inputs", [])
                sig_outputs = serving_sig.get("outputs", [])

                self.assertGreater(
                    len(sig_inputs),
                    0,
                    "Should have at least one input in SignatureDef",
                )
                self.assertGreater(
                    len(sig_outputs),
                    0,
                    "Should have at least one output in SignatureDef",
                )

                # Verify input signature
                if isinstance(input_data, dict):
                    # torch export renames inputs to `args_0_<key>`, so we
                    # only check the input count here.
                    self.assertEqual(
                        len(input_data),
                        len(sig_inputs),
                        f"Input count mismatch: model has "
                        f"{len(input_data)} inputs but SignatureDef has "
                        f"{len(sig_inputs)}: {sig_inputs}",
                    )
                else:
                    # For numpy arrays, just verify we have exactly one input
                    # (since we're passing a single tensor)
                    if len(sig_inputs) != 1:
                        self.fail(
                            "Expected 1 input for numpy array input_data, "
                            f"but SignatureDef has {len(sig_inputs)}: "
                            f"{sig_inputs}"
                        )

                # Verify output signature
                if verify_numerics and isinstance(keras_output, dict):
                    expected_outputs = set(keras_output.keys())
                    actual_outputs = set(sig_outputs)
                    if expected_outputs != actual_outputs:
                        self.fail(
                            f"Output name mismatch: Expected "
                            f"{sorted(expected_outputs)}, "
                            f"but SignatureDef has {sorted(actual_outputs)}"
                        )

                # Step 3: Run LiteRT inference
                os.remove(export_path)
                # Simple inference implementation
                runner = interpreter.get_signature_runner("serving_default")

                # Convert input data dtypes to match TFLite expectations
                dtype_map = {
                    "bool": "int32",
                    "float64": "float32",
                    "int64": "int32",
                }

                def convert_for_tflite(x):
                    """Convert tensor/array to TFLite-compatible dtypes."""
                    x = ops.convert_to_numpy(x)
                    dtype = keras.backend.standardize_dtype(x.dtype)
                    target = dtype_map.get(dtype)
                    if target is not None:
                        x = x.astype(target)
                    return x

                if isinstance(input_data, dict):
                    converted_input_data = tree.map_structure(
                        convert_for_tflite, input_data
                    )
                    # litert-torch renames dict inputs (positionally as
                    # `args_<n>` or as `args_<n>_<key>`); map them back and
                    # cast to the interpreter's expected dtype.
                    runner_kwargs = self._map_litert_torch_inputs(
                        converted_input_data, sig_inputs
                    )
                    expected_dtypes = {
                        d["name"]: d["dtype"]
                        for d in interpreter.get_input_details()
                    }
                    for sig_name, value in list(runner_kwargs.items()):
                        for dname, dtype in expected_dtypes.items():
                            if sig_name in dname and value.dtype != dtype:
                                runner_kwargs[sig_name] = value.astype(dtype)
                                break
                    litert_output = runner(**runner_kwargs)
                else:
                    # For single tensor inputs, get the input name
                    sig_inputs = serving_sig.get("inputs", [])
                    input_name = sig_inputs[
                        0
                    ]  # We verified len(sig_inputs) == 1 above
                    converted_input = convert_for_tflite(input_data)
                    litert_output = runner(**{input_name: converted_input})

                # Step 4: Verify outputs
                self._verify_litert_outputs(
                    keras_output,
                    litert_output,
                    sig_outputs,
                    expected_output_shape=expected_output_shape,
                    verify_numerics=verify_numerics,
                    comparison_mode=comparison_mode,
                    output_thresholds=output_thresholds,
                )
        finally:
            if interpreter is not None:
                del interpreter
            if model is not None and cls is not None:
                del model
            gc.collect()

    def _create_tflite_interpreter(self, tflite_path):
        """Create a TFLite interpreter for verifying LiteRT-LM bundles.

        We avoid XNNPACK because `litert_torch` bundles may contain ops/shapes
        that the XNNPACK delegate cannot reshape at prepare time. We use the
        built-in op resolver without default delegates so all LiteRT-LM ops
        (including CUMSUM for multimodal models) remain available.
        """
        try:
            from ai_edge_litert.interpreter import Interpreter
            from ai_edge_litert.interpreter import OpResolverType

            return Interpreter(
                model_path=tflite_path,
                experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            )
        except Exception:
            pass
        try:
            return tf.lite.Interpreter(
                model_path=tflite_path,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            )
        except Exception:
            pass
        return tf.lite.Interpreter(model_path=tflite_path)

    def _parse_litertlm_bundle(self, litertlm_path):
        """Read a `.litertlm` bundle and return its raw data + metadata table.

        Returns:
            A tuple of ``(data, metadata)`` where ``data`` is the bundle bytes
            and ``metadata`` is the parsed ``LiteRTLMMetaData`` flatbuffer.
        """
        from litert_lm_builder import litertlm_peek

        with open(litertlm_path, "rb") as f:
            data = f.read()
        metadata = litertlm_peek.read_litertlm_header(
            litertlm_path, io.StringIO()
        )
        return data, metadata

    def _extract_litertlm_tflite_interpreters(self, litertlm_path):
        """Extract every TFLite model from a `.litertlm` bundle."""
        from litert_lm_builder import litertlm_core as core

        data, metadata = self._parse_litertlm_bundle(litertlm_path)

        interpreters = []
        for i in range(metadata.SectionMetadata().ObjectsLength()):
            obj = metadata.SectionMetadata().Objects(i)
            if (
                core.any_section_data_type_to_string(obj.DataType())
                != "TFLiteModel"
            ):
                continue
            tflite_data = data[obj.BeginOffset() : obj.EndOffset()]
            tflite_path = os.path.join(
                self.get_temp_dir(),
                f"litertlm_model_{len(interpreters)}.tflite",
            )
            with open(tflite_path, "wb") as f:
                f.write(tflite_data)
            interpreters.append(self._create_tflite_interpreter(tflite_path))
        return interpreters

    def _parse_litertlm_llm_metadata(self, litertlm_path):
        """Parse the ``LlmMetadata`` protobuf from a `.litertlm` bundle."""
        from litert_lm_builder import litertlm_core as core
        from litert_lm_builder.runtime.proto import llm_metadata_pb2

        data, metadata = self._parse_litertlm_bundle(litertlm_path)

        for i in range(metadata.SectionMetadata().ObjectsLength()):
            obj = metadata.SectionMetadata().Objects(i)
            if (
                core.any_section_data_type_to_string(obj.DataType())
                != "LlmMetadataProto"
            ):
                continue
            llm_meta_buf = data[obj.BeginOffset() : obj.EndOffset()]
            meta = llm_metadata_pb2.LlmMetadata()
            meta.ParseFromString(llm_meta_buf)
            return meta
        return None

    def _compare_outputs(
        self,
        keras_val,
        litert_val,
        comparison_mode,
        key=None,
        max_threshold=10.0,
        mean_threshold=0.1,
    ):
        """Compare Keras and LiteRT outputs using specified comparison mode.

        Args:
            keras_val: Keras model output (numpy array)
            litert_val: LiteRT model output (numpy array)
            comparison_mode: "strict" or "statistical"
            key: Output key name for error messages (optional)
            max_threshold: Maximum absolute difference threshold for statistical
                mode
            mean_threshold: Mean absolute difference threshold for statistical
                mode
        """
        key_msg = f" for output key '{key}'" if key else ""

        # Check if shapes are compatible for comparison
        self.assertEqual(
            keras_val.shape,
            litert_val.shape,
            f"Shape mismatch{key_msg}: Keras shape "
            f"{keras_val.shape}, LiteRT shape {litert_val.shape}. "
            "Numerical comparison cannot proceed due to incompatible shapes.",
        )

        if comparison_mode == "strict":
            # Original strict element-wise comparison with default tolerances
            self.assertAllClose(
                keras_val,
                litert_val,
                atol=1e-6,
                rtol=1e-6,
                msg=f"Mismatch{key_msg}",
            )
        elif comparison_mode == "statistical":
            # Statistical comparison

            # Calculate element-wise absolute differences
            abs_diff = np.abs(keras_val - litert_val)

            # Element-wise statistics
            mean_abs_diff = np.mean(abs_diff)
            max_abs_diff = np.max(abs_diff)

            # Assert reasonable bounds on statistical differences
            self.assertLessEqual(
                mean_abs_diff,
                mean_threshold,
                f"Mean absolute difference too high: {mean_abs_diff:.6e}"
                f"{key_msg} (threshold: {mean_threshold})",
            )
            self.assertLessEqual(
                max_abs_diff,
                max_threshold,
                f"Max absolute difference too high: {max_abs_diff:.6e}"
                f"{key_msg} (threshold: {max_threshold})",
            )
        else:
            raise ValueError(
                f"Unknown comparison_mode: {comparison_mode}. Must be "
                "'strict' or 'statistical'"
            )

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
        if backbone.token_embedding is not None:
            # Check we can embed tokens eagerly.
            output = backbone.token_embedding(ops.zeros((2, 3), dtype="int32"))

            # Check variable length sequences.
            if variable_length_data is None:
                # If no variable length data passed, assume the second axis of
                # all inputs is our sequence axis and create it ourselves.
                variable_length_data = [
                    tree.map_structure(
                        lambda x: x[:, :seq_length, ...], input_data
                    )
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
        self.assertRegex(backbone.name, name)

        # Check mixed precision.
        if run_mixed_precision_check:
            self.run_precision_test(cls, init_kwargs, input_data)

        # Check quantization.
        if run_quantization_check:
            self.run_quantization_test(backbone, cls, init_kwargs, input_data)

    def run_vision_backbone_test(
        self,
        cls,
        init_kwargs,
        input_data,
        expected_output_shape,
        spatial_output_keys=None,
        expected_pyramid_output_keys=None,
        expected_pyramid_image_sizes=None,
        variable_length_data=None,
        run_mixed_precision_check=True,
        run_quantization_check=True,
        run_data_format_check=True,
    ):
        """Run basic tests for a vision backbone, including compilation."""
        can_run_data_format_check = True
        if (
            keras.config.backend() == "tensorflow"
            and not tf.config.list_physical_devices("GPU")
        ):
            # Never test the "channels_first" format on tensorflow CPU.
            # Tensorflow lacks support for "channels_first" convolution.
            can_run_data_format_check = False

        ori_data_format = keras.config.image_data_format()
        keras.config.set_image_data_format("channels_last")
        self.run_backbone_test(
            cls=cls,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape=expected_output_shape,
            variable_length_data=variable_length_data,
            run_mixed_precision_check=run_mixed_precision_check,
            run_quantization_check=run_quantization_check,
        )

        if expected_pyramid_output_keys:
            backbone = cls(**init_kwargs)
            model = keras.models.Model(
                backbone.inputs, backbone.pyramid_outputs
            )
            output_data = model(input_data)

            self.assertIsInstance(output_data, dict)
            self.assertEqual(
                sorted(output_data.keys()),
                sorted(backbone.pyramid_outputs.keys()),
            )
            self.assertEqual(
                sorted(output_data.keys()), sorted(expected_pyramid_output_keys)
            )
            # check height and width of each level.
            for i, (k, v) in enumerate(output_data.items()):
                self.assertEqual(
                    tuple(v.shape[1:3]), expected_pyramid_image_sizes[i]
                )

        # Check data_format. We assume that `input_data` is in "channels_last"
        # format.
        if run_data_format_check and can_run_data_format_check:
            keras.config.set_image_data_format("channels_first")
            input_data_shape = ops.shape(input_data)
            if len(input_data_shape) == 3:
                input_data = ops.transpose(input_data, axes=(2, 0, 1))
            elif len(input_data_shape) == 4:
                input_data = ops.transpose(input_data, axes=(0, 3, 1, 2))
            if isinstance(expected_output_shape, dict):
                # Handle dictionary of shapes.
                transposed_shapes = {}
                for key, shape in expected_output_shape.items():
                    if spatial_output_keys and key not in spatial_output_keys:
                        transposed_shapes[key] = shape
                        continue
                    if len(shape) == 3:
                        transposed_shapes[key] = (shape[0], shape[2], shape[1])
                    elif len(shape) == 4:
                        transposed_shapes[key] = (
                            shape[0],
                            shape[3],
                            shape[1],
                            shape[2],
                        )
                    else:
                        transposed_shapes[key] = shape
                expected_output_shape = transposed_shapes
            elif len(expected_output_shape) == 3:
                x = expected_output_shape
                expected_output_shape = (x[0], x[2], x[1])
            elif len(expected_output_shape) == 4:
                x = expected_output_shape
                expected_output_shape = (x[0], x[3], x[1], x[2])
            original_init_kwargs = init_kwargs.copy()
            init_kwargs = original_init_kwargs.copy()
            # Handle nested `keras.Model` instances passed within `init_kwargs`.
            for k, v in init_kwargs.items():
                if isinstance(v, keras.Model) and hasattr(v, "data_format"):
                    config = v.get_config()
                    config["data_format"] = "channels_first"
                    if (
                        "image_shape" in config
                        and config["image_shape"] is not None
                        and len(config["image_shape"]) == 3
                    ):
                        config["image_shape"] = tuple(
                            reversed(config["image_shape"])
                        )
                    init_kwargs[k] = v.__class__.from_config(config)
            if "image_shape" in init_kwargs:
                init_kwargs = init_kwargs.copy()
                init_kwargs["image_shape"] = tuple(
                    reversed(init_kwargs["image_shape"])
                )
            self.run_backbone_test(
                cls=cls,
                init_kwargs=init_kwargs,
                input_data=input_data,
                expected_output_shape=expected_output_shape,
                variable_length_data=variable_length_data,
                run_mixed_precision_check=run_mixed_precision_check,
                run_quantization_check=run_quantization_check,
            )

        # Restore the original `image_data_format`.
        keras.config.set_image_data_format(ori_data_format)

    def run_task_test(
        self,
        cls,
        init_kwargs,
        train_data,
        expected_output_shape=None,
        batch_size=2,
        compile_kwargs=None,
    ):
        """Run basic tests for a backbone, including compilation."""
        task = cls(**init_kwargs)
        if compile_kwargs:
            task.compile(**compile_kwargs)
        # Check serialization (without a full save).
        self.run_serialization_test(task)
        preprocessor = task.preprocessor
        ds = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
        x, y, sw = keras.utils.unpack_x_y_sample_weight(train_data)

        # Test: the tree struct output by the
        # preprocessor must match what model expects.
        preprocessed_data = preprocessor(*train_data)[0]
        tree.assert_same_structure(
            preprocessed_data,
            task._inputs_struct,
            check_types=False,
        )

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
        # Turn off default compilation, should error during `fit()`.
        task = cls(**init_kwargs, compile=False)
        with self.assertRaisesRegex(ValueError, "You must call `compile"):
            task.fit(ds)

    def run_preset_test(
        self,
        cls,
        preset,
        input_data,
        init_kwargs={},
        expected_output=None,
        expected_output_shape=None,
        expected_partial_output=None,
        expected_labels=None,
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

        if expected_labels is not None:
            output = ops.argmax(output, axis=-1)
            self.assertAllEqual(output, expected_labels)

    def get_test_data_dir(self):
        return str(pathlib.Path(__file__).parent / "test_data")

    def load_test_image(self, target_size=None):
        # From https://commons.wikimedia.org/wiki/File:California_quail.jpg
        path = os.path.join(self.get_test_data_dir(), "test_image.jpg")
        img = keras.utils.load_img(
            path, target_size=target_size, keep_aspect_ratio=True
        )
        return np.array(img)

    def _litertlm_multimodal_keras_prefill(
        self, model, spec, inputs, num_layers, cache_length
    ):
        """Prefill ``model`` eagerly with the encoders run outside the adapter.

        The TFLite graph under test is traced from
        ``KerasHubLiteRTAdapter.forward_prefill``, so building the Keras
        reference by calling that same module would apply any adapter defect
        to both sides of a comparison and cancel it out. This runs the
        vision encoder here and then goes straight to
        ``model.call_with_cache``, the way ``_verify_litertlm_numerics``
        already does for text.

        Args:
            model: The built multimodal ``CausalLM``.
            spec: The ``LiteRTLMExportSpec`` resolved for ``model``.
            inputs: The prefill sample-input dict, also fed to TFLite.
            num_layers: Number of decoder layers in the KV cache.
            cache_length: The cache length the bundle was traced with.

        Returns:
            The updated KV cache, stacked as ``[batch, num_layers, 2, ...]``.
        """
        import inspect

        import torch

        from keras_hub.src.utils.litertlm.model_specs import _get_vision_encoder

        call_kwargs = {
            "vision_mask": inputs.get("vision_mask"),
            "vision_indices": inputs.get("vision_indices"),
        }
        tokens = inputs["tokens"]
        with torch.no_grad():
            if "images" in inputs or "pixel_values" in inputs:
                vision_encoder = _get_vision_encoder(model.backbone)
                if spec.vision_input_style == "embedded_pixel_values":
                    # This style keeps the vision encoder inside the backbone,
                    # so there is nothing to run outside it.
                    call_kwargs["pixel_values"] = inputs["images"]
                elif spec.vision_input_style == "patch_values":
                    call_kwargs["img_embeddings"] = vision_encoder(
                        {
                            "pixel_values": inputs["pixel_values"],
                            "pixel_position_ids": inputs["pixel_position_ids"],
                        }
                    )
                else:
                    images = inputs["images"]
                    if spec.flatten_image_batch:
                        images = images.reshape(-1, *images.shape[2:])
                    call_kwargs["img_embeddings"] = vision_encoder(images)
            params = inspect.signature(model.call_with_cache).parameters
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in params}
            call_kwargs.update(
                spec.get_forced_call_with_cache_kwargs(tokens, cache_length)
            )
            k_stack = torch.stack(
                [inputs[f"kv_cache_k_{i}"] for i in range(num_layers)], dim=1
            )
            v_stack = torch.stack(
                [inputs[f"kv_cache_v_{i}"] for i in range(num_layers)], dim=1
            )
            _, _, updated_cache = model.call_with_cache(
                tokens,
                torch.stack([k_stack, v_stack], dim=2),
                int(inputs["input_pos"][0]),
                **call_kwargs,
            )
        return updated_cache

    def _verify_litertlm_multimodal_numerics(
        self,
        model,
        interpreter,
        prefill_seq_len,
        atol=1e-4,
        rtol=1e-4,
        seed=0,
        verification_level=None,
    ):
        """Compare Keras eager and TFLite outputs for a multimodal bundle.

        Multimodal sibling of ``_verify_litertlm_numerics``. Prefill compares
        KV-cache tensors only: the multimodal prefill signature emits no
        logits by design (see ``adapter.py`` ``forward_prefill``, which calls
        ``_call_with_cache(..., return_logits=False)``; the runtime extracts
        last-token logits via a dedicated decode step). The first decode step
        consumes the last prompt token and compares logits between the TFLite
        bundle and the Keras model, mirroring the text helper's structure.

        Both the TFLite prefill signature and the Keras reference are fed the
        identical sample-input dict produced by
        ``export._build_prefill_inputs`` -- the exact dict the export pipeline
        feeds the prefill signature at trace time -- so the two sides differ
        only in how those inputs are consumed, and the helper stays in
        lockstep with the export pipeline across refactors (the multimodal
        prefill signature's input names/shapes deliberately do NOT match the
        preprocessor output dict, so feeding the preprocessor dict to both
        sides would be wrong). The Keras side consumes them through
        ``_litertlm_multimodal_keras_prefill``, which runs the vision
        encoder itself instead of through the traced adapter, so the two
        sides are independent computations of the same quantity.

        The builder returns zeros; zeros make parity pass trivially (both
        sides compute on zeros). Only the data-bearing tensors (``tokens`` and
        the raw image feature tensors) are replaced with seeded random
        values; index/mask/``input_pos``/kv-cache-seed tensors are left as the
        builder set them, because they encode the traced structure.

        Tolerance policy: defaults to Gemma3's proven ``1e-4``. Do NOT relax
        silently. If a future family needs a looser tolerance, the CALLER
        passes it AND carries a one-line code comment at the call site
        justifying it -- this helper never widens tolerances on its own.

        Args:
            model: The built KerasHub multimodal ``CausalLM`` -- the same
                instance passed to ``model.export``.
            interpreter: The main TFLite interpreter from the bundle (the one
                carrying both a ``prefill*`` and a ``decode`` signature).
            prefill_seq_len: The int the bundle was exported with. Multimodal
                is single-bucket, so ``cache_length == prefill_seq_len``.
            atol: Absolute tolerance (default ``1e-4``, Gemma3's proven value).
            rtol: Relative tolerance (default ``1e-4``).
            seed: Seed for the random data-bearing inputs.
            verification_level: Optional override of the auto-detected level
                string; ``None`` auto-detects from vision presence.

        Returns:
            A dict describing what was actually verified, so callers/reports
            can record the achieved level rather than over-claiming:
            ``verification_level`` (str), ``prefill_kv_max_abs_err`` (float),
            ``decode_logits_max_abs_err`` (float), ``decode_kv_max_abs_err``
            (float), and ``has_vision`` (bool).
        """
        import torch

        # Local imports mirror the lazy-import convention of the text helper
        # above and of `export.py`/`model_specs.py`: this low-level,
        # widely-imported test module must not carry a module-level dependency
        # on the optional/heavy litertlm export package.
        from keras_hub.src.utils.litertlm import export as _export
        from keras_hub.src.utils.litertlm.model_specs import resolve_export_spec

        # Reconstruct the exact ExportPlan the export pipeline built, so the
        # sample inputs we feed match the traced signature by construction.
        spec = resolve_export_spec(model)
        cache_length = prefill_seq_len  # multimodal invariant
        cache_cfg = spec.get_cache_config(model, cache_length=cache_length)
        num_layers = cache_cfg["num_layers"]
        num_kv_heads = cache_cfg["num_kv_heads"]
        head_dim = cache_cfg["head_dim"]

        vision_cfg = spec.get_vision_config(model)
        has_vision = vision_cfg is not None
        if not has_vision:
            self.fail(
                "_verify_litertlm_multimodal_numerics called on a text-only "
                "model; use _verify_litertlm_numerics instead."
            )

        vision_input_style = spec.vision_input_style
        dtype = _export._torch_dtype_from_model(model)

        plan = _export.ExportPlan(
            spec=spec,
            num_layers=num_layers,
            cache_length=cache_length,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            prefill_seq_lens=[prefill_seq_len],
            dtype=dtype,
            has_vision=has_vision,
            vision_cfg=vision_cfg,
            vision_input_style=vision_input_style,
            # This harness always mirrors an unconfigured `model.export(...)`
            # call (no `sampler_config`), matching what the family tests
            # under verification actually export.
            sampler_config=None,
            # The harness never passes an `llm_model_type` override.
            model_type_overridden=False,
        )
        prefill_inputs = _export._build_prefill_inputs(plan)[prefill_seq_len]

        # `_build_prefill_inputs` returns zeros for index/mask tensors; the
        # data-bearing tensors are randomized below, but leaving indices/masks
        # zero makes the entire vision merge path an identity (slot 0 is
        # restored to the text embedding by Gemma3/Gemma4's
        # `interleave_embeddings`).
        # We therefore synthesize real placement indices and masks so the parity
        # check actually exercises the vision tower. We also keep a copy
        # of the zero-index structure for a sensitivity check.
        rng = np.random.default_rng(seed)
        vocab_size = model.backbone.vocabulary_size

        # Compute the actual number of tokens each encoder produces, because
        # some configs declare a theoretical maximum that exceeds the actual
        # encoder output for the trace-time input shape.
        actual_vision_tokens = None
        with torch.no_grad():
            if "pixel_values" in prefill_inputs:
                from keras_hub.src.utils.litertlm.model_specs import (
                    _get_vision_encoder,
                )

                vision_encoder = _get_vision_encoder(model.backbone)
                vision_out = vision_encoder(
                    {
                        "pixel_values": prefill_inputs["pixel_values"],
                        "pixel_position_ids": prefill_inputs[
                            "pixel_position_ids"
                        ],
                    }
                )
                # vision_out shape is (batch, max_images, tokens_per_image, dim)
                actual_vision_tokens = vision_out.shape[1] * vision_out.shape[2]

        # Real placement: start after the BOS slot to avoid the restore-to-text
        # behavior at index 0. Cap the number of placed tokens to what fits in
        # the sequence and to the actual encoder output count.
        seq_len = prefill_inputs["tokens"].shape[1]
        max_places = seq_len - 1
        cursor = 1
        if "vision_indices" in prefill_inputs:
            num_vision_tokens = min(
                prefill_inputs["vision_indices"].shape[1],
                actual_vision_tokens
                or prefill_inputs["vision_indices"].shape[1],
                max_places,
            )
            vision_start = cursor
            prefill_inputs["vision_indices"] = torch.zeros_like(
                prefill_inputs["vision_indices"]
            )
            prefill_inputs["vision_indices"][:, :num_vision_tokens] = (
                torch.arange(
                    vision_start,
                    vision_start + num_vision_tokens,
                    dtype=torch.int32,
                ).unsqueeze(0)
            )
            vision_mask = torch.zeros_like(prefill_inputs["vision_mask"])
            vision_mask[:, vision_start : vision_start + num_vision_tokens] = 1
            prefill_inputs["vision_mask"] = vision_mask
            cursor += num_vision_tokens
            max_places -= num_vision_tokens
        if "pixel_position_ids" in prefill_inputs:
            # Gemma4's test preprocessor produces all-1s 2D patch positions.
            # Mirror that rather than a synthetic grid so the vision-encoder
            # positional embedding path is exercised with the same values the
            # export was traced against.
            prefill_inputs["pixel_position_ids"] = torch.ones_like(
                prefill_inputs["pixel_position_ids"]
            )

        zero_structure_prefill_inputs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in prefill_inputs.items()
        }
        if "vision_indices" in zero_structure_prefill_inputs:
            zero_structure_prefill_inputs["vision_indices"] = torch.zeros_like(
                zero_structure_prefill_inputs["vision_indices"]
            )
            zero_structure_prefill_inputs["vision_mask"] = torch.zeros_like(
                zero_structure_prefill_inputs["vision_mask"]
            )

        for name, t in list(prefill_inputs.items()):
            shape = tuple(t.shape)
            if name == "tokens":
                prefill_inputs[name] = torch.from_numpy(
                    rng.integers(1, vocab_size, size=shape).astype("int32")
                )
                zero_structure_prefill_inputs[name] = prefill_inputs[
                    name
                ].clone()
            elif name in ("images", "pixel_values"):
                prefill_inputs[name] = torch.from_numpy(
                    rng.standard_normal(shape).astype("float32")
                )
                zero_structure_prefill_inputs[name] = prefill_inputs[
                    name
                ].clone()

        # Auto-detect the verification level unless overridden.
        if verification_level is None:
            verification_level = "end_to_end_vision"

        # -- TFLite prefill (mirror the text helper's signature selection) --
        sig_list = list(interpreter._get_full_signature_list().keys())
        if "prefill" in sig_list:
            prefill_sig = "prefill"
        else:
            matching = sorted(
                s
                for s in sig_list
                if s.startswith("prefill_")
                and int(s.split("_")[1]) >= prefill_seq_len
            )
            prefill_sig = matching[0] if matching else None
        if prefill_sig is None:
            self.fail(
                "No usable prefill signature found for multimodal parity."
            )

        tflite_prefill_feed = {
            name: t.detach().cpu().numpy() for name, t in prefill_inputs.items()
        }
        tflite_prefill_out = interpreter.get_signature_runner(prefill_sig)(
            **tflite_prefill_feed
        )

        # -- Keras prefill, fed the identical inputs --
        keras_cache = self._litertlm_multimodal_keras_prefill(
            model, spec, prefill_inputs, num_layers, cache_length
        )
        keras_cache_np = keras_cache.detach().cpu().numpy()

        # Real placement indices/masks must move the reference KV cache away
        # from the all-zero placement the sample builder produces, or the
        # towers feed nothing and the comparison below is vacuous.
        zero_structure_cache = self._litertlm_multimodal_keras_prefill(
            model, spec, zero_structure_prefill_inputs, num_layers, cache_length
        )
        sensitivity_diff = float(
            np.max(
                np.abs(
                    keras_cache_np - zero_structure_cache.detach().cpu().numpy()
                )
            )
        )
        self.assertGreater(
            sensitivity_diff,
            1e-6,
            f"Vision tower is not wired into the reference: "
            f"sensitivity diff {sensitivity_diff} <= 1e-6. "
            f"The multimodal parity check is vacuous.",
        )

        # -- Compare prefill KV caches (prefill emits no logits) --
        prefill_kv_max_abs_err = 0.0
        for i in range(num_layers):
            for j, kv in enumerate(("k", "v")):
                key = f"kv_cache_{kv}_{i}"
                keras_kv = keras_cache_np[:, i, j, ...]
                tflite_kv = tflite_prefill_out[key]
                prefill_kv_max_abs_err = max(
                    prefill_kv_max_abs_err,
                    float(np.max(np.abs(keras_kv - tflite_kv))),
                )
                self.assertAllClose(
                    keras_kv,
                    tflite_kv,
                    atol=atol,
                    rtol=rtol,
                    msg=f"Multimodal prefill KV mismatch at {key}",
                )

        # -- First decode step (last prompt token), compare logits --
        # Multimodal export enforces `cache_length == prefill_seq_len`, so
        # there is no decode headroom past the prompt: `decode_pos` is the
        # last prefilled slot. (This differs from the text helper, which bumps
        # cache_length to leave a headroom slot.)
        decode_pos = min(prefill_seq_len, cache_length - 1)
        tokens_np = prefill_inputs["tokens"].detach().cpu().numpy()
        # The decode token's identity does not matter for numeric parity (both
        # backends run the identical op on it); reuse the prompt's last token.
        decode_token = tokens_np[:, -1:].copy()

        # Keras decode continues from the Keras prefill cache, applying the
        # same family-forced `call_with_cache` kwargs the exported decode graph
        # baked in (only Gemma3n has any).
        decode_tokens = torch.from_numpy(decode_token)
        with torch.no_grad():
            keras_logits, _, keras_cache_dec = model.call_with_cache(
                decode_tokens,
                keras_cache,
                decode_pos,
                **spec.get_forced_call_with_cache_kwargs(
                    decode_tokens, cache_length
                ),
            )
        keras_logits = keras_logits.detach().cpu().numpy()
        keras_cache_dec = keras_cache_dec.detach().cpu().numpy()

        # TFLite decode: feed the TFLite prefill KV out.
        decode_feed = {
            "tokens": decode_token,
            "input_pos": np.array([decode_pos], dtype=np.int32),
        }
        for i in range(num_layers):
            for kv in ("k", "v"):
                key = f"kv_cache_{kv}_{i}"
                decode_feed[key] = tflite_prefill_out[key]
        tflite_decode_out = interpreter.get_signature_runner("decode")(
            **decode_feed
        )
        tflite_logits = tflite_decode_out["logits"]

        decode_logits_max_abs_err = float(
            np.max(np.abs(keras_logits - tflite_logits))
        )
        self.assertEqual(
            keras_logits.shape,
            tflite_logits.shape,
            f"Multimodal decode logits shape mismatch: "
            f"Keras {keras_logits.shape} vs TFLite {tflite_logits.shape}",
        )
        self.assertAllClose(
            keras_logits,
            tflite_logits,
            atol=atol,
            rtol=rtol,
            msg="Multimodal first-decode logits mismatch",
        )

        # Compare decode-step KV caches as well, matching the text helper.
        decode_kv_max_abs_err = 0.0
        for i in range(num_layers):
            for j, kv in enumerate(("k", "v")):
                key = f"kv_cache_{kv}_{i}"
                if key not in tflite_decode_out:
                    continue
                keras_kv = keras_cache_dec[:, i, j, ...]
                tflite_kv = tflite_decode_out[key]
                decode_kv_max_abs_err = max(
                    decode_kv_max_abs_err,
                    float(np.max(np.abs(keras_kv - tflite_kv))),
                )
                self.assertAllClose(
                    keras_kv,
                    tflite_kv,
                    atol=atol,
                    rtol=rtol,
                    msg=f"Multimodal decode KV mismatch at {key}",
                )

        return {
            "verification_level": verification_level,
            "prefill_kv_max_abs_err": prefill_kv_max_abs_err,
            "decode_logits_max_abs_err": decode_logits_max_abs_err,
            "decode_kv_max_abs_err": decode_kv_max_abs_err,
            "has_vision": has_vision,
        }
