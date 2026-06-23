import gc
import json
import os
import pathlib
import re
import struct
import tempfile
import time

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

    def _create_tflite_interpreter(self, tflite_path):
        """Create a TFLite interpreter for verifying LiteRT-LM bundles.

        We avoid XNNPACK because `litert_torch` bundles may contain ops/shapes
        that the XNNPACK delegate cannot reshape at prepare time. We use the
        built-in op resolver without default delegates so all LiteRT-LM ops
        (including CUMSUM for multimodal models) remain available.

        This relies on `ai-edge-litert`; the deprecated `tf.lite.Interpreter`
        path has been removed because it is no longer present in recent
        TensorFlow releases.
        """
        from ai_edge_litert.interpreter import Interpreter
        from ai_edge_litert.interpreter import OpResolverType

        return Interpreter(
            model_path=tflite_path,
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )

    def _extract_litertlm_tflite_interpreters(self, litertlm_path):
        """Extract every TFLite model from a `.litertlm` bundle."""
        from litert_lm_builder import litertlm_core as core

        with open(litertlm_path, "rb") as f:
            data = f.read()
        header_end = struct.unpack("<Q", data[24:32])[0]
        metadata_buf = data[32:header_end]
        metadata = core.schema.LiteRTLMMetaData.GetRootAsLiteRTLMMetaData(
            metadata_buf, 0
        )

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

        with open(litertlm_path, "rb") as f:
            data = f.read()
        header_end = struct.unpack("<Q", data[24:32])[0]
        metadata_buf = data[32:header_end]
        metadata = core.schema.LiteRTLMMetaData.GetRootAsLiteRTLMMetaData(
            metadata_buf, 0
        )

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

    def _verify_litertlm_numerics(
        self,
        model,
        interpreter,
        input_data,
        atol,
        rtol,
    ):
        """Compare Keras eager and TFLite prefill/decode outputs."""
        import torch

        tokens_np = ops.convert_to_numpy(input_data)
        if tokens_np.ndim != 2:
            raise ValueError(
                "`input_data` for LiteRT-LM numeric parity must be a 2-D "
                f"token tensor. Received shape: {tokens_np.shape}"
            )

        B, T = tokens_np.shape
        backbone = model.backbone
        L = getattr(backbone, "num_layers", None)
        if L is None:
            L = getattr(backbone, "num_hidden_layers", None)
        if L is None:
            raise ValueError(
                "Model backbone must expose `num_layers` or "
                "`num_hidden_layers`."
            )
        H = getattr(
            backbone,
            "num_key_value_heads",
            getattr(backbone, "num_heads", None),
        )
        if H is None:
            raise ValueError(
                "Model backbone must expose `num_key_value_heads` or "
                "`num_heads`."
            )
        D = getattr(backbone, "head_dim", None)
        if D is None:
            hidden_dim = getattr(backbone, "hidden_dim", None)
            num_qh = getattr(
                backbone,
                "num_query_heads",
                getattr(backbone, "num_heads", None),
            )
            if hidden_dim is None or num_qh is None or num_qh <= 0:
                raise ValueError(
                    "Could not determine attention head dimension."
                )
            D = hidden_dim // num_qh

        cache_length = getattr(backbone, "max_sequence_length", None)
        if cache_length is None:
            preprocessor = getattr(model, "preprocessor", None)
            cache_length = getattr(preprocessor, "sequence_length", T)
        if cache_length is None:
            cache_length = T

        # Gemma3n uses a different KV-cache axis order than standard models.
        cache_layout = (
            "gemma3n"
            if type(backbone).__name__.startswith("Gemma3n")
            else "standard"
        )
        if cache_layout == "gemma3n":
            keras_cache_shape = (B, L, 2, H, cache_length, D)
            per_layer_shape = (B, H, cache_length, D)
        else:
            keras_cache_shape = (B, L, 2, cache_length, H, D)
            per_layer_shape = (B, cache_length, H, D)

        # Find the best prefill signature (bucketed or single).
        sig_list = list(interpreter._get_full_signature_list().keys())
        prefill_sig = None
        if "prefill" in sig_list:
            prefill_sig = "prefill"
        else:
            matching = sorted(
                [
                    s
                    for s in sig_list
                    if s.startswith("prefill_") and int(s.split("_")[1]) >= T
                ]
            )
            if matching:
                prefill_sig = matching[0]
        if prefill_sig is None:
            self.fail("No usable prefill signature found for numeric parity.")

        cache_keras = np.zeros(keras_cache_shape, dtype=np.float32)
        prefill_inputs = {
            "tokens": tokens_np,
            "input_pos": np.arange(T, dtype=np.int32),
        }
        for i in range(L):
            prefill_inputs[f"kv_cache_k_{i}"] = np.zeros(
                per_layer_shape, dtype=np.float32
            )
            prefill_inputs[f"kv_cache_v_{i}"] = np.zeros(
                per_layer_shape, dtype=np.float32
            )

        prefill_runner = interpreter.get_signature_runner(prefill_sig)
        tflite_prefill_out = prefill_runner(**prefill_inputs)

        # Keras prefill.
        with torch.no_grad():
            _, _, keras_cache = model.call_with_cache(
                torch.from_numpy(tokens_np),
                torch.from_numpy(cache_keras),
                0,
            )
        keras_cache = keras_cache.numpy()

        # Compare prefill KV caches.
        for i in range(L):
            self.assertAllClose(
                keras_cache[:, i, 0, ...],
                tflite_prefill_out[f"kv_cache_k_{i}"],
                atol=atol,
                rtol=rtol,
            )
            self.assertAllClose(
                keras_cache[:, i, 1, ...],
                tflite_prefill_out[f"kv_cache_v_{i}"],
                atol=atol,
                rtol=rtol,
            )

        # Single decode step at position 0.
        decode_pos = 0
        decode_token = tokens_np[:, decode_pos : decode_pos + 1].copy()
        with torch.no_grad():
            keras_logits_dec, _, keras_cache_dec = model.call_with_cache(
                torch.from_numpy(decode_token),
                torch.from_numpy(keras_cache),
                decode_pos,
            )
        keras_logits_dec = keras_logits_dec.numpy()
        keras_cache_dec = keras_cache_dec.numpy()

        decode_inputs = {
            "tokens": decode_token,
            "input_pos": np.array([decode_pos], dtype=np.int32),
        }
        for i in range(L):
            decode_inputs[f"kv_cache_k_{i}"] = tflite_prefill_out[
                f"kv_cache_k_{i}"
            ]
            decode_inputs[f"kv_cache_v_{i}"] = tflite_prefill_out[
                f"kv_cache_v_{i}"
            ]
        decode_runner = interpreter.get_signature_runner("decode")
        tflite_dec_out = decode_runner(**decode_inputs)

        self.assertAllClose(
            keras_logits_dec,
            tflite_dec_out["logits"],
            atol=atol,
            rtol=rtol,
        )
        for i in range(L):
            self.assertAllClose(
                keras_cache_dec[:, i, 0, ...],
                tflite_dec_out[f"kv_cache_k_{i}"],
                atol=atol,
                rtol=rtol,
            )
            self.assertAllClose(
                keras_cache_dec[:, i, 1, ...],
                tflite_dec_out[f"kv_cache_v_{i}"],
                atol=atol,
                rtol=rtol,
            )

    def _verify_litertlm_generation(
        self,
        litertlm_path,
        prompt="hi",
        max_num_tokens=8,
    ):
        """Load a ``.litertlm`` bundle with the LiteRT-LM runtime and generate.

        This is a smoke test: with randomly initialized tiny models the output
        text is meaningless, but the runtime must successfully produce a
        non-empty response. It verifies that the tokenizer, metadata, and
        prefill/decode graphs are consistent enough for the engine to execute.
        """
        try:
            import litert_lm
        except ImportError:
            self.skipTest(
                "End-to-end LiteRT-LM generation verification requires "
                "`litert-lm`. Install it with: pip install litert-lm"
            )

        # Keep the smoke test focused on functional failures; LiteRT runtime
        # accelerator-enumeration logs are not actionable here.
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

        engine = litert_lm.Engine(
            litertlm_path,
            backend=litert_lm.Backend.CPU(),
            max_num_tokens=max_num_tokens,
        )
        conversation = engine.create_conversation()
        response = conversation.send_message(prompt)
        self.assertIsInstance(response, dict)
        self.assertIn("content", response)
        contents = response["content"]
        self.assertTrue(contents, "LiteRT-LM runtime returned empty content.")
        texts = [
            item.get("text", "")
            for item in contents
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        self.assertTrue(
            any(texts),
            "LiteRT-LM runtime did not produce any text output.",
        )

    def run_litertlm_export_test(
        self,
        cls=None,
        init_kwargs=None,
        model=None,
        input_data=None,
        prefill_seq_len=None,
        verify_numerics=True,
        verify_model_type=None,
        verify_generation=False,
        generation_prompt="hi",
        generation_max_tokens=8,
        atol=1e-4,
        rtol=1e-4,
        expected_error_regex=None,
        **export_kwargs,
    ):
        """Export a KerasHub model to LiteRT-LM and verify the bundle.

        Args:
            cls: Model class to instantiate if ``model`` is not provided.
            init_kwargs: Initialization arguments for ``cls``.
            model: Pre-built model instance. If provided, ``cls`` and
                ``init_kwargs`` are ignored.
            input_data: Token ids tensor for text-only numeric parity, or
                ``None`` to skip numeric verification.
            prefill_seq_len: Sequence length passed to ``model.export``. May be
                an ``int`` or a list of bucket sizes.
            verify_numerics: Whether to run Keras vs TFLite numeric parity. Set
                to ``False`` for multimodal models or preset models with
                random weights.
            verify_model_type: Expected ``LlmMetadata`` oneof name, e.g.
                ``"gemma3"``, ``"gemma4"`` or ``"generic_model"``.
            verify_generation: Whether to load the exported bundle with the
                LiteRT-LM Python runtime and run a short generation smoke
                test. Useful for verifying tokenizer + metadata + runtime
                consistency, even with dummy weights.
            generation_prompt: Prompt used for the runtime smoke test.
            generation_max_tokens: Maximum tokens to generate in the smoke
                test.
            atol: Absolute tolerance for numeric parity.
            rtol: Relative tolerance for numeric parity.
            expected_error_regex: If provided, the test asserts that
                ``model.export(..., format="litertlm", ...)`` raises a
                ``ValueError`` whose message matches this regex. This
                centralizes unsupported-tokenizer tests and exits before
                requiring ``litert-torch`` or the PyTorch backend.
            **export_kwargs: Additional arguments forwarded to
                ``model.export(..., format="litertlm", ...)``.
        """

        def _debug_print(msg):
            print(msg)

        # Centralized unsupported-tokenizer assertion path. Tokenizer
        # validation in export.py runs before backend/dependency checks, so
        # this path works on any backend and without litert-torch installed.
        if expected_error_regex is not None:
            if model is None:
                if cls is None or init_kwargs is None:
                    raise ValueError(
                        "Either `model` or both `cls` and `init_kwargs` must "
                        "be provided."
                    )
                model = cls(**init_kwargs)
            path = os.path.join(self.get_temp_dir(), "model.litertlm")
            if prefill_seq_len is not None:
                export_kwargs.setdefault("prefill_seq_len", prefill_seq_len)
            with self.assertRaisesRegex(
                ValueError, expected_error_regex
            ):
                model.export(path, format="litertlm", **export_kwargs)
            return

        if keras.config.backend() != "torch":
            self.skipTest("LiteRT-LM export requires the PyTorch backend.")

        import importlib.util

        if importlib.util.find_spec("litert_torch") is None:
            self.skipTest(
                "LiteRT-LM export requires `litert-torch`. "
                "Install it with: pip install litert-torch"
            )

        if importlib.util.find_spec("litert_lm_builder") is None:
            self.skipTest(
                "LiteRT-LM export requires `litert-lm-builder`. "
                "Install it with: pip install litert-lm-builder"
            )

        total_start = time.perf_counter()

        if model is None:
            if cls is None or init_kwargs is None:
                raise ValueError(
                    "Either `model` or both `cls` and `init_kwargs` must be "
                    "provided."
                )
            build_start = time.perf_counter()
            model = cls(**init_kwargs)
            _debug_print(
                f"[litertlm] build model: "
                f"{time.perf_counter() - build_start:.2f}s"
            )
        else:
            _debug_print("[litertlm] build model: 0.00s")

        path = os.path.join(self.get_temp_dir(), "model.litertlm")
        if prefill_seq_len is not None:
            export_kwargs.setdefault("prefill_seq_len", prefill_seq_len)

        export_start = time.perf_counter()
        model.export(path, format="litertlm", **export_kwargs)
        _debug_print(
            f"[litertlm] export: {time.perf_counter() - export_start:.2f}s"
        )

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

        extract_start = time.perf_counter()
        interpreters = self._extract_litertlm_tflite_interpreters(path)
        self.assertTrue(
            interpreters,
            "No TFLite model found in the .litertlm bundle.",
        )
        _debug_print(
            f"[litertlm] extract tflite: "
            f"{time.perf_counter() - extract_start:.2f}s"
        )

        sig_start = time.perf_counter()
        all_signatures = {}
        for interpreter in interpreters:
            all_signatures.update(interpreter._get_full_signature_list())
        prefill_sigs = [
            name for name in all_signatures if name.startswith("prefill")
        ]
        self.assertTrue(
            prefill_sigs,
            f"No prefill signature found. Signatures: {list(all_signatures)}",
        )
        self.assertIn(
            "decode",
            all_signatures,
            f"No decode signature found. Signatures: {list(all_signatures)}",
        )

        main_interpreter = None
        for interpreter in interpreters:
            sigs = interpreter._get_full_signature_list()
            if any(s.startswith("prefill") for s in sigs) and "decode" in sigs:
                main_interpreter = interpreter
                break
        if main_interpreter is None:
            main_interpreter = interpreters[0]
        _debug_print(
            f"[litertlm] verify signatures: "
            f"{time.perf_counter() - sig_start:.2f}s"
        )

        meta_start = time.perf_counter()
        if verify_model_type is not None:
            llm_metadata = self._parse_litertlm_llm_metadata(path)
            self.assertIsNotNone(
                llm_metadata,
                "LlmMetadata section not found in .litertlm bundle.",
            )
            model_type_msg = llm_metadata.llm_model_type
            actual_type = model_type_msg.WhichOneof("model_type")
            self.assertEqual(
                actual_type,
                verify_model_type,
                f"Expected LlmModelType '{verify_model_type}', "
                f"got '{actual_type}'.",
            )
        _debug_print(
            f"[litertlm] verify metadata: "
            f"{time.perf_counter() - meta_start:.2f}s"
        )

        numeric_start = time.perf_counter()
        if verify_numerics and input_data is not None:
            numeric_input = input_data
            if isinstance(input_data, dict):
                numeric_input = input_data.get("token_ids")
            if numeric_input is not None:
                # Skip numeric parity for multimodal inputs; the helper only
                # validates text token prefill/decode KV-cache parity.
                text_only_keys = {"token_ids", "padding_mask"}
                if isinstance(
                    input_data, dict
                ) and not text_only_keys.issuperset(input_data.keys()):
                    _debug_print(
                        "[litertlm] numeric parity skipped: multimodal input"
                    )
                else:
                    # The exported TFLite prefill signature is traced with
                    # batch_size=1, so numeric parity must use a single sample.
                    numeric_input = ops.convert_to_numpy(numeric_input)
                    if numeric_input.ndim >= 2 and numeric_input.shape[0] > 1:
                        numeric_input = numeric_input[:1]
                    self._verify_litertlm_numerics(
                        model,
                        main_interpreter,
                        numeric_input,
                        atol=atol,
                        rtol=rtol,
                    )
        _debug_print(
            f"[litertlm] numeric parity: "
            f"{time.perf_counter() - numeric_start:.2f}s"
        )

        generation_start = time.perf_counter()
        if verify_generation:
            self._verify_litertlm_generation(
                path,
                prompt=generation_prompt,
                max_num_tokens=generation_max_tokens,
            )
        _debug_print(
            f"[litertlm] runtime generation: "
            f"{time.perf_counter() - generation_start:.2f}s"
        )

        _debug_print(
            f"[litertlm] total: {time.perf_counter() - total_start:.2f}s"
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
