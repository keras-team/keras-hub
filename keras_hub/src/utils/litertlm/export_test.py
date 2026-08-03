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
