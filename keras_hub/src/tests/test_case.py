import gc
import json
import os
import pathlib
import re
import tempfile

import keras
import numpy as np
import packaging.version
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
                model.export(), such as allow_custom_ops=True or
                enable_select_tf_ops=True.
        """
        # Skip test if Keras version is less than 3.13
        if packaging.version.Version(
            keras.__version__
        ) < packaging.version.Version("3.13.0"):
            self.skipTest("LiteRT export requires Keras >= 3.13")

        # Extract comparison_mode from export_kwargs if provided
        comparison_mode = export_kwargs.pop("comparison_mode", "strict")
        if keras.backend.backend() != "tensorflow":
            self.skipTest("LiteRT export only supports TensorFlow backend")

        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            Interpreter = tf.lite.Interpreter

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
                    expected_inputs = set(input_data.keys())
                    actual_inputs = set(sig_inputs)
                    # Check that all expected inputs are in the signature
                    # (allow signature to have additional optional inputs)
                    missing_inputs = expected_inputs - actual_inputs
                    if missing_inputs:
                        self.fail(
                            f"Missing inputs in SignatureDef: "
                            f"{sorted(missing_inputs)}. "
                            f"Expected: {sorted(expected_inputs)}, "
                            f"SignatureDef has: {sorted(actual_inputs)}"
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
                def convert_for_tflite(x):
                    """Convert tensor/array to TFLite-compatible dtypes."""
                    if hasattr(x, "dtype"):
                        if isinstance(x, np.ndarray):
                            if x.dtype == bool:
                                return x.astype(np.int32)
                            elif x.dtype == np.float64:
                                return x.astype(np.float32)
                            elif x.dtype == np.int64:
                                return x.astype(np.int32)
                        else:  # TensorFlow tensor
                            if x.dtype == tf.bool:
                                return ops.cast(x, "int32").numpy()
                            elif x.dtype == tf.float64:
                                return ops.cast(x, "float32").numpy()
                            elif x.dtype == tf.int64:
                                return ops.cast(x, "int32").numpy()
                            else:
                                return x.numpy() if hasattr(x, "numpy") else x
                    elif hasattr(x, "numpy"):
                        return x.numpy()
                    return x

                if isinstance(input_data, dict):
                    converted_input_data = tree.map_structure(
                        convert_for_tflite, input_data
                    )
                    litert_output = runner(**converted_input_data)
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
    ):
        """Run basic tests for a backbone, including compilation."""
        task = cls(**init_kwargs)
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
