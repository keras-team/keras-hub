import importlib.util
import json
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
from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_hub.src.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.models.llama3.llama3_causal_lm import Llama3CausalLM
from keras_hub.src.models.llama3.llama3_causal_lm_preprocessor import (
    Llama3CausalLMPreprocessor,
)
from keras_hub.src.models.llama3.llama3_tokenizer import Llama3Tokenizer
from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.models.qwen3.qwen3_causal_lm import Qwen3CausalLM
from keras_hub.src.models.qwen3.qwen3_causal_lm_preprocessor import (
    Qwen3CausalLMPreprocessor,
)
from keras_hub.src.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.litertlm import export
from keras_hub.src.utils.litertlm.adapter import _cpu_default_device_scope
from keras_hub.src.utils.litertlm.hf_tokenizer_converter import (
    materialize_hf_tokenizer_json,
)
from keras_hub.src.utils.litertlm.model_specs import GREEDY_SAMPLER_CONFIG
from keras_hub.src.utils.litertlm.model_specs import SamplerConfig

_LITERT_TORCH_AVAILABLE = importlib.util.find_spec("litert_torch") is not None
_LITERT_LM_BUILDER_AVAILABLE = (
    importlib.util.find_spec("litert_lm_builder") is not None
)

try:
    import tokenizers
except ImportError:
    tokenizers = None


# Content tokens shared by the tiny BPE vocabs in the HF-tokenizer
# export/roundtrip tests. Tokenizer assertions are exact-id oracles, so
# this token order is fixed.
_TINY_BPE_CONTENT_TOKENS = [
    "h",
    "i",
    "Ġ",
    "Ġh",
    "e",
    "l",
    "o",
    "w",
    "r",
    "d",
    "t",
    "s",
    "a",
    "b",
    "ab",
]


def _tiny_bpe_vocab(special_tokens, extra_tokens=("n", "k", "u", "m")):
    """Tiny BPE vocab: per-family special tokens, then shared content."""
    tokens = (
        list(special_tokens) + _TINY_BPE_CONTENT_TOKENS + list(extra_tokens)
    )
    return {token: token_id for token_id, token in enumerate(tokens)}


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

    def test_export_with_hf_tokenizer_path(self):
        """Verify export with a user-provided HuggingFace tokenizer.json."""
        try:
            import litert_lm
            import tokenizers
        except ImportError:
            self.skipTest("This test requires `litert-lm` and `tokenizers`.")

        vocab_size = self.tokenizer.vocabulary_size()

        # Build a tiny HuggingFace BPE tokenizer with the same vocab size.
        vocab = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
        }
        for i in range(4, vocab_size):
            vocab[f"tok{i}"] = i

        hf_tokenizer = tokenizers.Tokenizer(
            tokenizers.models.BPE(vocab=vocab, merges=[])
        )
        hf_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        hf_tokenizer.add_special_tokens(["<pad>", "<s>", "</s>", "<unk>"])

        hf_tokenizer_path = os.path.join(self.get_temp_dir(), "tokenizer.json")
        hf_tokenizer.save(hf_tokenizer_path)

        path = os.path.join(self.get_temp_dir(), "test_hf_tokenizer.litertlm")
        self.model.export(
            path,
            format="litertlm",
            prefill_seq_len=8,
            hf_tokenizer_path=hf_tokenizer_path,
        )

        self.assertTrue(os.path.exists(path))

        # Smoke-test that the LiteRT-LM runtime can construct an Engine from
        # the bundle. Full generation requires a real-world HF tokenizer; the
        # synthetic one above is sufficient to prove `add_hf_tokenizer` was
        # used and the bundle structure is valid.
        engine = litert_lm.Engine(
            path,
            backend=litert_lm.Backend.CPU(),
            max_num_tokens=4,
        )
        self.assertIsNotNone(engine)

    def test_export_with_hf_tokenizer_path_mismatched_vocab_raises(self):
        """`hf_tokenizer_path` pointing at a wildly different vocab size
        must be rejected before any tracing/bundling work happens.

        Uses a bare-bones hand-written ``tokenizer.json`` (rather than a
        real ``tokenizers.Tokenizer``, as ``test_export_with_hf_tokenizer_path``
        above does) because the vocab-size sanity check runs during
        argument validation, before the file would ever be loaded by the
        `tokenizers` library or `litert_lm_builder`.
        """
        model_vocab_size = self.tokenizer.vocabulary_size()
        # Comfortably past both the absolute-diff and ratio thresholds in
        # `_check_hf_tokenizer_vocab_compatible`.
        mismatched_vocab_size = model_vocab_size * 100 + 1000
        vocab = {f"tok{i}": i for i in range(mismatched_vocab_size)}
        hf_tokenizer_path = os.path.join(
            self.get_temp_dir(), "mismatched_tokenizer.json"
        )
        with open(hf_tokenizer_path, "w", encoding="utf-8") as f:
            json.dump({"model": {"vocab": vocab}}, f)

        path = os.path.join(
            self.get_temp_dir(), "test_mismatched_hf_tokenizer.litertlm"
        )
        with self.assertRaisesRegex(
            ValueError,
            "appears incompatible with the model",
        ):
            self.model.export(
                path,
                format="litertlm",
                prefill_seq_len=8,
                hf_tokenizer_path=hf_tokenizer_path,
            )

    def test_export_gpt2_with_auto_hf_tokenizer(self):
        """Export a tiny GPT2 model with auto-converted HF tokenizer."""
        vocab = _tiny_bpe_vocab(["<|endoftext|>"])
        merges = ["a b"]
        tokenizer = GPT2Tokenizer(vocabulary=vocab, merges=merges)

        backbone = GPT2Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=4,
            hidden_dim=32,
            intermediate_dim=64,
            max_sequence_length=8,
        )
        preprocessor = GPT2CausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=8
        )
        model = GPT2CausalLM(backbone=backbone, preprocessor=preprocessor)

        self._set_random_weights(model)

        path = os.path.join(self.get_temp_dir(), "test_gpt2_auto_hf.litertlm")
        model.export(path, format="litertlm", prefill_seq_len=8)
        self.assertTrue(os.path.exists(path))

    def test_export_llama3_with_auto_hf_tokenizer(self):
        """Export a tiny Llama3 model with auto-converted HF tokenizer."""
        vocab = _tiny_bpe_vocab(
            [
                "<|endoftext|>",
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ]
        )
        merges = ["a b"]
        tokenizer = Llama3Tokenizer(vocabulary=vocab, merges=merges)

        backbone = Llama3Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=32,
            intermediate_dim=64,
            max_sequence_length=8,
        )
        preprocessor = Llama3CausalLMPreprocessor(
            tokenizer=tokenizer, sequence_length=8
        )
        model = Llama3CausalLM(backbone=backbone, preprocessor=preprocessor)

        self._set_random_weights(model)

        path = os.path.join(self.get_temp_dir(), "test_llama3_auto_hf.litertlm")
        model.export(path, format="litertlm", prefill_seq_len=8)
        self.assertTrue(os.path.exists(path))

        # Regression coverage for the stop-token fix (see `Llama3Spec` in
        # model_specs.py): Llama3's chat-turn-stop token `<|eot_id|>` (id 5
        # in the vocab above) must reach the exported metadata alongside the
        # primary EOS `<|end_of_text|>` (id 2) -- previously only the
        # Gemma-specific `<end_of_turn>` literal was checked, so Llama3
        # never got a chat-stop token beyond its primary (non-chat) EOS.
        llm_metadata = self._parse_litertlm_llm_metadata(path)
        self.assertIsNotNone(llm_metadata)
        stop_token_ids = {
            stop_token.token_ids.ids[0]
            for stop_token in llm_metadata.stop_tokens
        }
        self.assertIn(2, stop_token_ids)  # <|end_of_text|>
        self.assertIn(5, stop_token_ids)  # <|eot_id|>

    def test_export_qwen3_with_auto_hf_tokenizer(self):
        """Export a tiny Qwen3 model with auto-converted HF tokenizer."""
        vocab = _tiny_bpe_vocab(["<|endoftext|>", "<|im_end|>"])
        merges = ["a b"]
        tokenizer = Qwen3Tokenizer(vocabulary=vocab, merges=merges)

        backbone = Qwen3Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            hidden_dim=32,
            intermediate_dim=64,
            max_sequence_length=8,
        )
        preprocessor = Qwen3CausalLMPreprocessor(
            tokenizer=tokenizer,
            sequence_length=8,
            add_start_token=False,
        )
        model = Qwen3CausalLM(backbone=backbone, preprocessor=preprocessor)

        self._set_random_weights(model)

        path = os.path.join(self.get_temp_dir(), "test_qwen3_auto_hf.litertlm")
        model.export(path, format="litertlm", prefill_seq_len=8)
        self.assertTrue(os.path.exists(path))


class TestHfTokenizerVocabCompatibility(TestCase):
    """Unit tests for the `hf_tokenizer_path` vocab-size sanity check.

    These exercise `_hf_tokenizer_vocab_size` and
    `_check_hf_tokenizer_vocab_compatible` directly against hand-written
    `tokenizer.json` fixtures and a minimal fake model, independent of any
    Keras backend or real KerasHub model -- both helpers are plain
    JSON/attribute-lookup logic with no tensor operations.
    """

    def _write_tokenizer_json(self, vocab_size, extra_added_tokens=0):
        path = os.path.join(self.get_temp_dir(), "tokenizer.json")
        vocab = {f"tok{i}": i for i in range(vocab_size)}
        added_tokens = [
            {"id": vocab_size + i, "content": f"<extra{i}>"}
            for i in range(extra_added_tokens)
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"model": {"vocab": vocab}, "added_tokens": added_tokens}, f
            )
        return path

    def _fake_model(self, vocabulary_size):
        backbone = types.SimpleNamespace(vocabulary_size=vocabulary_size)
        return types.SimpleNamespace(backbone=backbone)

    def test_hf_tokenizer_vocab_size_counts_vocab_and_added_tokens(self):
        path = self._write_tokenizer_json(vocab_size=20, extra_added_tokens=3)
        self.assertEqual(export._hf_tokenizer_vocab_size(path), 23)

    def test_check_hf_tokenizer_vocab_compatible_matching_does_not_raise(self):
        # A handful of reserved/special tokens (well within the "few
        # hundred" absolute threshold) must not raise.
        path = self._write_tokenizer_json(vocab_size=1000)
        model = self._fake_model(vocabulary_size=1000)
        export._check_hf_tokenizer_vocab_compatible(path, model)

    def test_check_hf_tokenizer_vocab_compatible_mismatch_raises(self):
        path = self._write_tokenizer_json(vocab_size=50000)
        model = self._fake_model(vocabulary_size=32)
        with self.assertRaisesRegex(
            ValueError, "appears incompatible with the model"
        ):
            export._check_hf_tokenizer_vocab_compatible(path, model)


@unittest.skipIf(
    tokenizers is None,
    "BytePair-to-HF tokenizer roundtrip test requires `tokenizers`.",
)
class TestBytePairToHFTokenizer(TestCase):
    def test_byte_pair_to_hf_tokenizer_roundtrip(self):
        """Verify converted tokenizer.json round-trips through HF tokenizers."""
        if keras.config.backend() != "torch":
            self.skipTest(
                "BytePair tokenizer roundtrip requires torch backend."
            )

        vocab = _tiny_bpe_vocab(
            ["<|endoftext|>"], extra_tokens=["hello", "Ġworld"]
        )
        merges = ["a b"]
        tokenizer = GPT2Tokenizer(vocabulary=vocab, merges=merges)

        hf_tokenizer_path = materialize_hf_tokenizer_json(
            tokenizer, self.get_temp_dir()
        )
        hf_tokenizer = tokenizers.Tokenizer.from_file(hf_tokenizer_path)

        for text in [
            "hello",
            "hello world",
            "hi",
            "a b",
            "12345 and 67890",
        ]:
            with self.subTest(text=text):
                keras_ids = list(tokenizer(text))
                hf_ids = hf_tokenizer.encode(text).ids
                self.assertEqual(
                    keras_ids,
                    hf_ids,
                    f"Token ids differ for {text!r}",
                )
                keras_text = tokenizer.detokenize(keras_ids)
                hf_text = hf_tokenizer.decode(hf_ids)
                self.assertEqual(
                    keras_text,
                    hf_text,
                    f"Detokenized text differs for {text!r}",
                )


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
