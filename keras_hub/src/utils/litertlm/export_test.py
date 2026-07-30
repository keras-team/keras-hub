import importlib.util
import os
import types
import unittest
import unittest.mock

import keras
import numpy as np
import torch

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer
from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM
from keras_hub.src.models.gemma3.gemma3_causal_lm_preprocessor import (
    Gemma3CausalLMPreprocessor,
)
from keras_hub.src.models.gemma3.gemma3_image_converter import (
    Gemma3ImageConverter,
)
from keras_hub.src.models.gemma3.gemma3_vision_encoder import (
    Gemma3VisionEncoder,
)
from keras_hub.src.tests.mocks.mock_gemma3_tokenizer import MockGemma3Tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.litertlm import export
from keras_hub.src.utils.litertlm.adapter import _cpu_default_device_scope
from keras_hub.src.utils.litertlm.adapter import _run_vision_encoder_for_style
from keras_hub.src.utils.litertlm.model_specs import GREEDY_SAMPLER_CONFIG
from keras_hub.src.utils.litertlm.model_specs import SamplerConfig

_LITERT_TORCH_AVAILABLE = importlib.util.find_spec("litert_torch") is not None
_LITERT_LM_BUILDER_AVAILABLE = (
    importlib.util.find_spec("litert_lm_builder") is not None
)


@unittest.skipUnless(
    keras.config.backend() == "torch",
    "LiteRT-LM export requires the PyTorch backend.",
)
@unittest.skipIf(
    not _LITERT_TORCH_AVAILABLE,
    "LiteRT-LM export requires `litert-torch`. "
    "Install it with: pip install litert-torch",
)
@unittest.skipIf(
    not _LITERT_LM_BUILDER_AVAILABLE,
    "LiteRT-LM export requires `litert-lm-builder`. "
    "Install it with: pip install litert-lm-builder",
)
class TestLiteRTLmExport(TestCase):
    def setUp(self):
        super().setUp()
        proto = os.path.join(self.get_test_data_dir(), "gemma_test_vocab.spm")
        self.tokenizer = GemmaTokenizer(proto=proto)
        self.backbone = GemmaBackbone(
            vocabulary_size=self.tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=32,
            head_dim=8,
            intermediate_dim=64,
            max_sequence_length=8,
        )
        self.preprocessor = GemmaCausalLMPreprocessor(
            tokenizer=self.tokenizer, sequence_length=8
        )
        self.model = GemmaCausalLM(
            backbone=self.backbone, preprocessor=self.preprocessor
        )
        self._set_random_weights(self.model)

    def _set_random_weights(self, model, seed=42):
        rng = np.random.default_rng(seed)
        weights = model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        model.set_weights(weights)

    def _assert_kv_cache_close(
        self, keras_cache, tflite_out, atol, rtol, num_layers
    ):
        """Per-layer K/V parity between a Keras cache and TFLite outputs.

        ``num_layers`` is supplied by the caller rather than read off
        ``keras_cache`` so that a cache built with the wrong layer axis cannot
        turn the comparison loop into a no-op.
        """
        self.assertEqual(keras_cache.shape[1], num_layers)
        for i in range(num_layers):
            self.assertAllClose(
                keras_cache[:, i, 0, ...],
                tflite_out[f"kv_cache_k_{i}"],
                atol=atol,
                rtol=rtol,
            )
            self.assertAllClose(
                keras_cache[:, i, 1, ...],
                tflite_out[f"kv_cache_v_{i}"],
                atol=atol,
                rtol=rtol,
            )

    def _call_with_cache_no_grad(
        self, model, tokens, cache, start_index, **kwargs
    ):
        """Eager `call_with_cache` under `no_grad`; numpy in, numpy out."""
        with torch.no_grad():
            logits, _, cache = model.call_with_cache(
                torch.from_numpy(tokens),
                torch.from_numpy(cache),
                start_index,
                **kwargs,
            )
        return logits.detach().cpu().numpy(), cache.detach().cpu().numpy()

    def test_export_tiny_gemma(self):
        path = os.path.join(self.get_temp_dir(), "test.litertlm")
        self.model.export(path, format="litertlm", prefill_seq_len=8)

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_export_with_bucketing(self):
        """Verify that multiple prefill_seq_len creates multiple signatures."""
        path = os.path.join(self.get_temp_dir(), "test_buckets.litertlm")
        self.model.export(
            path,
            format="litertlm",
            prefill_seq_len=[4, 8],
        )

        self.assertTrue(os.path.exists(path))

        # Extract TFLite from all bucketed interpreters and verify signatures.
        interpreters = self._extract_litertlm_tflite_interpreters(path)
        all_signatures = {}
        for interpreter in interpreters:
            all_signatures.update(interpreter._get_full_signature_list())
        signatures = list(all_signatures.keys())

        self.assertIn("prefill_4", signatures)
        self.assertIn("prefill_8", signatures)
        self.assertIn("decode", signatures)

    def test_export_outputs_match_keras(self):
        """Verify that exported TFLite outputs match Keras eager outputs."""
        # Export
        litertlm_path = os.path.join(self.get_temp_dir(), "verify.litertlm")
        self.model.export(litertlm_path, format="litertlm", prefill_seq_len=8)

        # Extract TFLite
        interpreter = self._extract_litertlm_tflite_interpreters(litertlm_path)[
            0
        ]

        B, T, L = 1, 8, 2
        H = self.backbone.num_key_value_heads
        D = self.backbone.head_dim
        tokens_np = (
            np.arange(1, 1 + T, dtype=np.int32).reshape(B, T)
            % self.tokenizer.vocabulary_size()
        )
        cache_keras = np.zeros((B, L, 2, T, H, D), dtype=np.float32)

        # Keras prefill
        keras_logits, keras_cache = self._call_with_cache_no_grad(
            self.model, tokens_np, cache_keras, 0
        )

        # TFLite prefill
        prefill_runner = interpreter.get_signature_runner("prefill")
        prefill_inputs = {
            "tokens": tokens_np,
            "input_pos": np.arange(T, dtype=np.int32),
        }
        for i in range(L):
            prefill_inputs[f"kv_cache_k_{i}"] = cache_keras[:, i, 0, ...]
            prefill_inputs[f"kv_cache_v_{i}"] = cache_keras[:, i, 1, ...]
        tflite_prefill_out = prefill_runner(**prefill_inputs)

        # Prefill returns only KV caches (no logits) per LiteRT-LM spec.
        # Logits are verified via the decode step below.

        # Compare prefill KV caches
        self._assert_kv_cache_close(
            keras_cache, tflite_prefill_out, atol=1e-4, rtol=1e-4, num_layers=L
        )

        # Keras decode at position 3
        decode_pos = 3
        decode_token = tokens_np[:, decode_pos : decode_pos + 1].copy()
        keras_logits_dec, keras_cache_dec = self._call_with_cache_no_grad(
            self.model, decode_token, keras_cache, decode_pos
        )

        # TFLite decode
        decode_runner = interpreter.get_signature_runner("decode")
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
        tflite_dec_out = decode_runner(**decode_inputs)

        # Compare decode logits
        self.assertAllClose(
            keras_logits_dec,
            tflite_dec_out["logits"],
            atol=1e-4,
            rtol=1e-4,
        )

        # Compare decode KV caches
        self._assert_kv_cache_close(
            keras_cache_dec, tflite_dec_out, atol=1e-4, rtol=1e-4, num_layers=L
        )

    def test_export_with_backend_constraint(self):
        """Verify export with valid backend_constraints succeeds."""
        for backend in ("cpu", "gpu", "npu", "gpu_artisan"):
            with self.subTest(backend=backend):
                path = os.path.join(
                    self.get_temp_dir(), f"test_backend_{backend}.litertlm"
                )
                self.model.export(
                    path,
                    format="litertlm",
                    prefill_seq_len=8,
                    backend_constraint=backend,
                )
                self.assertTrue(os.path.exists(path))

    def test_export_lowercases_backend_constraint(self):
        """Verify a mixed-case backend_constraint reaches the builder call
        already lowercased.

        `_validate_export_args` lowercases `backend_constraint` to validate
        it, but must also return the normalized value so
        `export_to_litertlm` threads *that* value through to
        `_assemble_bundle` / `builder.add_tflite_model`, rather than the
        original (possibly mixed-case) argument. `litert_lm_builder` itself
        happens to lowercase `backend_constraint` again before persisting it
        as metadata, so a real end-to-end bundle read would pass even with
        the bug (the original argument would still show up lowercased in
        the file); this spies on the actual call to
        `LitertLmFileBuilder.add_tflite_model` to check what our own code
        passes, independent of that downstream normalization.
        """
        import litert_lm_builder

        path = os.path.join(self.get_temp_dir(), "test_backend_case.litertlm")
        original_add_tflite_model = (
            litert_lm_builder.LitertLmFileBuilder.add_tflite_model
        )
        captured_backend_constraints = []

        def _spy_add_tflite_model(self, *args, **kwargs):
            captured_backend_constraints.append(
                kwargs.get("backend_constraint")
            )
            return original_add_tflite_model(self, *args, **kwargs)

        with unittest.mock.patch.object(
            litert_lm_builder.LitertLmFileBuilder,
            "add_tflite_model",
            _spy_add_tflite_model,
        ):
            self.model.export(
                path,
                format="litertlm",
                prefill_seq_len=8,
                backend_constraint="GPU",
            )

        self.assertTrue(os.path.exists(path))
        self.assertTrue(captured_backend_constraints)
        for value in captured_backend_constraints:
            self.assertEqual(value, "gpu")

    def test_export_invalid_backend_constraint(self):
        """Verify invalid backend_constraint raises ValueError."""
        path = os.path.join(
            self.get_temp_dir(), "test_invalid_backend.litertlm"
        )
        with self.assertRaisesRegex(
            ValueError,
            "Invalid backend_constraint",
        ):
            self.model.export(
                path,
                format="litertlm",
                prefill_seq_len=8,
                backend_constraint="invalid_backend",
            )

    def test_export_model_type_metadata(self):
        """Verify the .litertlm metadata contains the correct model type."""
        path = os.path.join(self.get_temp_dir(), "test_metadata.litertlm")
        self.model.export(path, format="litertlm", prefill_seq_len=8)

        llm_metadata = self._parse_litertlm_llm_metadata(path)
        self.assertIsNotNone(llm_metadata)
        model_type_msg = llm_metadata.llm_model_type
        actual_type = model_type_msg.WhichOneof("model_type")
        self.assertEqual(actual_type, "generic_model")

    def test_export_omits_sampler_params_by_default(self):
        """No `sampler_config` -> `sampler_params` absent from metadata.

        Mirrors litert-torch export_hf: the field is written only when a
        caller explicitly requests it. keras-hub ships no default sampler.
        """
        path = os.path.join(self.get_temp_dir(), "no_sampler.litertlm")
        self.model.export(path, format="litertlm", prefill_seq_len=8)

        llm_metadata = self._parse_litertlm_llm_metadata(path)
        self.assertIsNotNone(llm_metadata)
        self.assertFalse(
            llm_metadata.HasField("sampler_params"),
            "sampler_params must be omitted when no sampler_config is passed.",
        )

    def test_export_greedy_sampler_config_roundtrip(self):
        """`GREEDY_SAMPLER_CONFIG` -> TOP_K type + k=1 in re-read metadata.

        We intentionally emit TOP_K instead of GREEDY because litertlm-android
        0.13.1 and host litert_lm 0.13.1 do not implement sampler type 3
        (GREEDY). TOP_K with k=1 is functionally equivalent.
        """
        from litert_lm_builder.runtime.proto import sampler_params_pb2

        path = os.path.join(self.get_temp_dir(), "greedy_sampler.litertlm")
        self.model.export(
            path,
            format="litertlm",
            prefill_seq_len=8,
            sampler_config=GREEDY_SAMPLER_CONFIG,
        )

        llm_metadata = self._parse_litertlm_llm_metadata(path)
        self.assertIsNotNone(llm_metadata)
        self.assertTrue(llm_metadata.HasField("sampler_params"))
        sp = llm_metadata.sampler_params
        self.assertEqual(sp.type, sampler_params_pb2.SamplerParameters.TOP_K)
        self.assertEqual(sp.k, 1)

    def test_sampler_config_validation_and_export_rejects_bad_type(self):
        """Invalid `SamplerConfig` values and non-config types raise."""
        # Dataclass-level validation: top_k < 1 is invalid.
        with self.assertRaises(ValueError):
            SamplerConfig(top_k=0)
        # All-None is invalid (would produce an empty sampler_params).
        with self.assertRaises(ValueError):
            SamplerConfig()
        # top_p out of range.
        with self.assertRaises(ValueError):
            SamplerConfig(top_p=1.5)

        # Exporter-level guard: a non-SamplerConfig value is rejected before
        # any tracing/bundling work.
        path = os.path.join(self.get_temp_dir(), "bad_sampler.litertlm")
        with self.assertRaises(ValueError):
            self.model.export(
                path,
                format="litertlm",
                prefill_seq_len=8,
                sampler_config={"top_k": 1},  # dict, not a SamplerConfig
            )

    def _build_tiny_gemma3_multimodal_model(
        self, num_layers=1, max_images=1, random_weights=False
    ):
        """Build a minimal Gemma3 vision-capable model.

        The defaults serve the bucketing-ban tests, which only need the
        rejection to fire, not any particular model content; the
        structural/numeric multimodal tests pass `num_layers=2,
        max_images=2, random_weights=True`. The mock tokenizer gets a
        SentencePiece asset because the export raises without one.
        """
        tokenizer = MockGemma3Tokenizer()
        self._attach_sentencepiece_tokenizer_asset(
            tokenizer,
            os.path.join(self.get_test_data_dir(), "gemma_test_vocab.spm"),
        )

        image_converter = Gemma3ImageConverter(image_size=(16, 16))
        preprocessor = Gemma3CausalLMPreprocessor(
            image_converter=image_converter,
            tokenizer=tokenizer,
            sequence_length=20,
            max_images_per_prompt=max_images,
            num_vision_tokens_per_image=4,
        )
        vision_encoder = Gemma3VisionEncoder(
            image_size=16,
            patch_size=4,
            pool_size=2,
            num_layers=num_layers,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )
        backbone = Gemma3Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=16,
            num_layers=num_layers,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            vision_encoder=vision_encoder,
        )
        model = Gemma3CausalLM(preprocessor=preprocessor, backbone=backbone)
        if random_weights:
            self._set_random_weights(model)
        return model

    def test_export_multimodal_bucketing_raises(self):
        """Verify multimodal export rejects mismatched prefill_seq_len."""
        model = self._build_tiny_gemma3_multimodal_model()

        path = os.path.join(
            self.get_temp_dir(), "test_multimodal_buckets.litertlm"
        )
        with self.assertRaisesRegex(
            ValueError,
            "Multimodal LiteRT-LM export currently requires",
        ):
            model.export(path, format="litertlm", prefill_seq_len=[8, 20])

    def test_export_multimodal_bucketing_error_is_family_wide(self):
        """The bucketing-rejection error must describe the restriction as
        enforced for all vision-capable families pending a per-family
        assessment -- not as a Gemma3-specific attention-mask limitation.

        The restriction itself is unchanged (see
        `test_export_multimodal_bucketing_raises`); only its stated
        justification was corrected. Guards against the message regressing to
        the old "This is a limitation of the Gemma3 attention mask
        computation" wording that over-attributed a family-wide default to
        one family.
        """
        model = self._build_tiny_gemma3_multimodal_model()
        path = os.path.join(
            self.get_temp_dir(), "test_multimodal_buckets_msg.litertlm"
        )
        with self.assertRaises(ValueError) as ctx:
            model.export(path, format="litertlm", prefill_seq_len=[8, 20])
        message = str(ctx.exception)
        # Stable prefix (also asserted by the sibling rejection test).
        self.assertIn("Multimodal LiteRT-LM export currently requires", message)
        # Accuracy: scoped to all vision families, not just Gemma3.
        self.assertIn("all vision-capable families", message)
        for family in ("Gemma3", "Gemma3n", "Gemma4", "PaliGemma"):
            self.assertIn(family, message)
        # Must NOT re-assert the old Gemma3-only attribution.
        self.assertNotIn(
            "This is a limitation of the Gemma3 attention mask",
            message,
        )

    def test_text_only_model_has_no_vision_inputs(self):
        """Verify text-only models do not expose vision inputs in signatures."""
        path = os.path.join(self.get_temp_dir(), "test_text_only.litertlm")
        self.model.export(path, format="litertlm", prefill_seq_len=8)

        interpreters = self._extract_litertlm_tflite_interpreters(path)
        interpreter = interpreters[0]
        prefill_sig = interpreter._get_full_signature_list()["prefill"]
        prefill_inputs = set(prefill_sig["inputs"])
        self.assertNotIn("images", prefill_inputs)
        self.assertNotIn("vision_indices", prefill_inputs)
        self.assertNotIn("vision_mask", prefill_inputs)

    def test_export_multimodal_tiny_gemma3(self):
        """Export a tiny Gemma3 vision+text model and verify structure."""
        model = self._build_tiny_gemma3_multimodal_model(
            num_layers=2, max_images=2, random_weights=True
        )

        path = os.path.join(self.get_temp_dir(), "test_multimodal.litertlm")
        model.export(path, format="litertlm", prefill_seq_len=20)

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

        # Extract TFLite and verify signatures contain vision inputs.
        interpreter = self._extract_litertlm_tflite_interpreters(path)[0]
        signatures = list(interpreter._get_full_signature_list().keys())

        self.assertIn("prefill", signatures)
        self.assertIn("decode", signatures)

        prefill_sig = interpreter._get_full_signature_list()["prefill"]
        prefill_inputs = set(prefill_sig["inputs"])
        self.assertIn("images", prefill_inputs)
        self.assertIn("vision_indices", prefill_inputs)
        self.assertIn("vision_mask", prefill_inputs)

    def test_multimodal_numeric_parity_gemma3(self):
        """Host-side multimodal (baked-in vision) numeric parity.

        Gemma3 is the reference family for the baked-in (encoder inside the
        PREFILL_DECODE graph) vision path; the tolerance is 1e-4, not relaxed.
        """
        # Random (not default) weights so the parity check is meaningful --
        # otherwise both backends would compute on identical default values.
        model = self._build_tiny_gemma3_multimodal_model(
            num_layers=2, max_images=2, random_weights=True
        )

        prefill_seq_len = 20
        path = os.path.join(self.get_temp_dir(), "test_mm_parity.litertlm")
        model.export(path, format="litertlm", prefill_seq_len=prefill_seq_len)

        interpreters = self._extract_litertlm_tflite_interpreters(path)
        main = None
        for it in interpreters:
            sigs = it._get_full_signature_list()
            if any(s.startswith("prefill") for s in sigs) and "decode" in sigs:
                main = it
                break
        main = main or interpreters[0]

        result = self._verify_litertlm_multimodal_numerics(
            model,
            main,
            prefill_seq_len=prefill_seq_len,
            atol=1e-4,
            rtol=1e-4,
        )

        # Prove it verified something real, at the level it claims.
        self.assertTrue(result["has_vision"])
        self.assertEqual(result["verification_level"], "end_to_end_vision")
        self.assertLess(result["prefill_kv_max_abs_err"], 1e-4)
        self.assertLess(result["decode_logits_max_abs_err"], 1e-4)


@unittest.skipUnless(
    keras.config.backend() == "torch",
    "LiteRT-LM export is only supported with the PyTorch backend.",
)
class TestLiteRTLmAdapterHelpers(TestCase):
    def test_cpu_default_device_scope_restores_device(self):
        """_cpu_default_device_scope restores the original default device."""
        original = torch.get_default_device()
        with _cpu_default_device_scope():
            self.assertEqual(torch.get_default_device(), torch.device("cpu"))
        self.assertEqual(torch.get_default_device(), original)


@unittest.skipUnless(
    keras.config.backend() != "torch",
    "This test only runs on non-PyTorch backends.",
)
class TestLiteRTLmExportBackendChecks(TestCase):
    def test_export_rejects_non_torch_backend(self):
        """The exporter raises a clear error on non-PyTorch backends."""
        if keras.config.backend() == "torch":
            self.skipTest("This test only runs on non-PyTorch backends.")

        proto = os.path.join(self.get_test_data_dir(), "gemma_test_vocab.spm")
        tokenizer = GemmaTokenizer(proto=proto)
        backbone = GemmaBackbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=32,
            head_dim=8,
            intermediate_dim=64,
            max_sequence_length=8,
        )
        preprocessor = GemmaCausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=8
        )
        model = GemmaCausalLM(backbone=backbone, preprocessor=preprocessor)

        with self.assertRaisesRegex(
            ValueError,
            "LiteRT-LM export is only supported with the PyTorch backend",
        ):
            model.export(
                os.path.join(self.get_temp_dir(), "test.litertlm"),
                format="litertlm",
                prefill_seq_len=8,
            )


@unittest.skipUnless(
    keras.config.backend() == "torch",
    "LiteRT-LM export is only supported with the PyTorch backend.",
)
class TestTorchDtypeFromModel(TestCase):
    def test_resolves_dtype_string(self):
        """A dtype string resolves to the matching `torch.dtype`."""
        model = types.SimpleNamespace(compute_dtype="float16")
        self.assertIs(export._torch_dtype_from_model(model), torch.float16)

    def test_rejects_non_dtype_compute_dtype(self):
        """A `compute_dtype` that is not a dtype string/`torch.dtype` fails."""
        model = types.SimpleNamespace(
            compute_dtype=None,
            backbone=types.SimpleNamespace(compute_dtype=None),
        )
        with self.assertRaisesRegex(
            ValueError,
            "must be a dtype string.*Received: compute_dtype=None",
        ):
            export._torch_dtype_from_model(model)

    def test_rejects_unmappable_dtype_string(self):
        """A dtype string with no PyTorch equivalent gets a LiteRT-LM error."""
        model = types.SimpleNamespace(compute_dtype="string")
        with self.assertRaisesRegex(
            ValueError,
            "`compute_dtype` must map to a PyTorch dtype",
        ) as cm:
            export._torch_dtype_from_model(model)
        self.assertIsInstance(cm.exception.__cause__, ValueError)


@unittest.skipUnless(
    keras.config.backend() == "torch",
    "LiteRT-LM export is only supported with the PyTorch backend.",
)
class TestVisionEncoderOutputContract(TestCase):
    def test_tensor_output_reaches_the_caller(self):
        """A tensor-returning encoder output is passed through unchanged."""
        features = torch.zeros((1, 4, 8))
        out = _run_vision_encoder_for_style(
            lambda images: features,
            "raw_images",
            False,
            images=torch.zeros((1, 1, 4, 4, 3)),
            pixel_values=None,
            pixel_position_ids=None,
        )
        self.assertIs(out, features)

    def test_raw_images_rejects_non_tensor_output(self):
        """A dict-returning raw_images encoder fails instead of leaking it."""
        features = torch.zeros((1, 4, 8))
        with self.assertRaisesRegex(
            ValueError, "return a single feature tensor"
        ):
            _run_vision_encoder_for_style(
                lambda images: {"features": features, "extra": features},
                "raw_images",
                False,
                images=torch.zeros((1, 1, 4, 4, 3)),
                pixel_values=None,
                pixel_position_ids=None,
            )

    def test_patch_values_rejects_non_tensor_output(self):
        """A tuple-returning patch_values encoder fails the same way."""
        features = torch.zeros((1, 4, 8))
        with self.assertRaisesRegex(
            ValueError, "return a single feature tensor"
        ):
            _run_vision_encoder_for_style(
                lambda inputs: (features,),
                "patch_values",
                False,
                images=None,
                pixel_values=features,
                pixel_position_ids=features,
            )
