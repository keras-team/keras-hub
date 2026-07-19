import importlib.util
import json
import os
import tempfile
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
    convert_byte_pair_to_hf,
)

_LITERT_TORCH_AVAILABLE = importlib.util.find_spec("litert_torch") is not None
_LITERT_LM_BUILDER_AVAILABLE = (
    importlib.util.find_spec("litert_lm_builder") is not None
)

try:
    import tokenizers
except ImportError:
    tokenizers = None


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

    def test_export_tiny_gemma(self):
        path = os.path.join(self.get_temp_dir(), "test.litertlm")
        self.model.export(path, format="litertlm", prefill_seq_len=8)

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

        # Runtime smoke test: the exported bundle must load in the LiteRT-LM
        # executor and produce non-empty text, even with random weights.
        self._verify_litertlm_generation(path)

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
        with torch.no_grad():
            keras_logits, _, keras_cache = self.model.call_with_cache(
                torch.from_numpy(tokens_np),
                torch.from_numpy(cache_keras),
                0,
            )
        keras_logits = keras_logits.detach().cpu().numpy()
        keras_cache = keras_cache.detach().cpu().numpy()

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
        for i in range(L):
            self.assertAllClose(
                keras_cache[:, i, 0, ...],
                tflite_prefill_out[f"kv_cache_k_{i}"],
                atol=1e-4,
                rtol=1e-4,
            )
            self.assertAllClose(
                keras_cache[:, i, 1, ...],
                tflite_prefill_out[f"kv_cache_v_{i}"],
                atol=1e-4,
                rtol=1e-4,
            )

        # Keras decode at position 3
        decode_pos = 3
        decode_token = tokens_np[:, decode_pos : decode_pos + 1].copy()
        with torch.no_grad():
            keras_logits_dec, _, keras_cache_dec = self.model.call_with_cache(
                torch.from_numpy(decode_token),
                torch.from_numpy(keras_cache),
                decode_pos,
            )
        keras_logits_dec = keras_logits_dec.detach().cpu().numpy()
        keras_cache_dec = keras_cache_dec.detach().cpu().numpy()

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
        for i in range(L):
            self.assertAllClose(
                keras_cache_dec[:, i, 0, ...],
                tflite_dec_out[f"kv_cache_k_{i}"],
                atol=1e-4,
                rtol=1e-4,
            )
            self.assertAllClose(
                keras_cache_dec[:, i, 1, ...],
                tflite_dec_out[f"kv_cache_v_{i}"],
                atol=1e-4,
                rtol=1e-4,
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

    def _build_tiny_gemma3_multimodal_model(self):
        """Build a minimal Gemma3 vision-capable model for bucketing-ban
        tests. Shared by `test_export_multimodal_bucketing_raises` and
        `test_export_multimodal_bucketing_error_is_family_wide` -- both only
        need the rejection to fire, not any particular model content."""
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
        from keras_hub.src.tests.mocks.mock_gemma3_tokenizer import (
            MockGemma3Tokenizer,
        )

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
            max_images_per_prompt=1,
            num_vision_tokens_per_image=4,
        )
        vision_encoder = Gemma3VisionEncoder(
            image_size=16,
            patch_size=4,
            pool_size=2,
            num_layers=1,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )
        backbone = Gemma3Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=16,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            vision_encoder=vision_encoder,
        )
        return Gemma3CausalLM(preprocessor=preprocessor, backbone=backbone)

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

        This is the accuracy half of V-7: the restriction itself is unchanged
        (see `test_export_multimodal_bucketing_raises`); only its stated
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
        self.assertIn(
            "Multimodal LiteRT-LM export currently requires", message
        )
        # Accuracy: scoped to all vision families, not just Gemma3.
        self.assertIn("all vision-capable families", message)
        for family in ("Gemma3", "Gemma3n", "Gemma4", "PaliGemma"):
            self.assertIn(family, message)
        # Must NOT re-assert the old Gemma3-only attribution.
        self.assertNotIn(
            "This is a limitation of the Gemma3 attention mask",
            message,
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
        from keras_hub.src.tests.mocks.mock_gemma3_tokenizer import (
            MockGemma3Tokenizer,
        )

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
            max_images_per_prompt=2,
            num_vision_tokens_per_image=4,
        )
        vision_encoder = Gemma3VisionEncoder(
            image_size=16,
            patch_size=4,
            pool_size=2,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )
        backbone = Gemma3Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=16,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            vision_encoder=vision_encoder,
        )
        model = Gemma3CausalLM(preprocessor=preprocessor, backbone=backbone)
        self._set_random_weights(model)

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
        """WS3.2: host-side multimodal (baked-in vision) numeric parity.

        Locks a regression baseline against the CURRENT code before the
        multimodal architecture refactor. Gemma3 is chosen because its
        baked-in vision parity is already proven elsewhere; the tolerance is
        1e-4 (Gemma3's proven value), not relaxed.
        """
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
        from keras_hub.src.tests.mocks.mock_gemma3_tokenizer import (
            MockGemma3Tokenizer,
        )

        tokenizer = MockGemma3Tokenizer()
        # Without a tokenizer asset the export raises; attach one exactly as
        # the sibling `test_export_multimodal_tiny_gemma3` does.
        self._attach_sentencepiece_tokenizer_asset(
            tokenizer,
            os.path.join(self.get_test_data_dir(), "gemma_test_vocab.spm"),
        )

        image_converter = Gemma3ImageConverter(image_size=(16, 16))
        preprocessor = Gemma3CausalLMPreprocessor(
            image_converter=image_converter,
            tokenizer=tokenizer,
            sequence_length=20,
            max_images_per_prompt=2,
            num_vision_tokens_per_image=4,
        )
        vision_encoder = Gemma3VisionEncoder(
            image_size=16,
            patch_size=4,
            pool_size=2,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )
        backbone = Gemma3Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=16,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            vision_encoder=vision_encoder,
        )
        model = Gemma3CausalLM(preprocessor=preprocessor, backbone=backbone)
        # Random (not default) weights so the parity check is meaningful --
        # otherwise both backends would compute on identical default values.
        self._set_random_weights(model)

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
        self.assertFalse(result["has_audio"])
        self.assertEqual(result["verification_level"], "end_to_end_vision")
        self.assertLess(result["prefill_kv_max_abs_err"], 1e-4)
        self.assertLess(result["decode_logits_max_abs_err"], 1e-4)

    def test_export_separate_vision_encoder_gemma3(self):
        """Export Gemma3 with separate vision encoder/adapter models."""
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
        from keras_hub.src.tests.mocks.mock_gemma3_tokenizer import (
            MockGemma3Tokenizer,
        )

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
            max_images_per_prompt=2,
            num_vision_tokens_per_image=4,
        )
        vision_encoder = Gemma3VisionEncoder(
            image_size=16,
            patch_size=4,
            pool_size=2,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )
        backbone = Gemma3Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=16,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            vision_encoder=vision_encoder,
        )
        model = Gemma3CausalLM(preprocessor=preprocessor, backbone=backbone)
        self._set_random_weights(model)

        path = os.path.join(
            self.get_temp_dir(), "test_separate_vision.litertlm"
        )
        model.export(
            path,
            format="litertlm",
            prefill_seq_len=20,
            separate_vision_encoder=True,
        )

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

        # Extract all TFLite models from the bundle.
        interpreters = self._extract_litertlm_tflite_interpreters(path)
        all_signatures = {}
        for interpreter in interpreters:
            all_signatures.update(interpreter._get_full_signature_list())

        signature_names = set(all_signatures.keys())
        self.assertIn("prefill", signature_names)
        self.assertIn("decode", signature_names)
        self.assertIn("vision_encoder", signature_names)
        self.assertIn("vision_adapter", signature_names)

        prefill_inputs = set(all_signatures["prefill"]["inputs"])
        self.assertNotIn("images", prefill_inputs)
        self.assertNotIn("pixel_values", prefill_inputs)
        self.assertNotIn("pixel_position_ids", prefill_inputs)
        self.assertIn("mm_embedding", prefill_inputs)

        vision_encoder_inputs = set(all_signatures["vision_encoder"]["inputs"])
        self.assertIn("images", vision_encoder_inputs)

        # Numeric parity: chain the separate vision_encoder -> vision_adapter
        # -> prefill/decode TFLite signatures and compare against the Keras
        # eager reference (which runs the vision encoder inline and feeds
        # img_embeddings directly), the same comparison
        # test_export_multimodal_outputs_match_keras does for the combined
        # (non-separate) vision-encoder path.
        interpreters_by_sig = {}
        for interpreter in interpreters:
            for sig_name in interpreter._get_full_signature_list():
                interpreters_by_sig[sig_name] = interpreter
        vision_encoder_interpreter = interpreters_by_sig["vision_encoder"]
        vision_adapter_interpreter = interpreters_by_sig["vision_adapter"]
        prefill_decode_interpreter = interpreters_by_sig["prefill"]

        B, T, L = 1, 20, 2
        H = backbone.num_key_value_heads
        D = backbone.head_dim
        tokens_np = (
            np.arange(1, 1 + T, dtype=np.int32).reshape(B, T)
            % tokenizer.vocabulary_size()
        )
        cache_keras = np.zeros((B, L, 2, T, H, D), dtype=np.float32)
        images_np = np.ones((B, 2, 16, 16, 3), dtype=np.float32)
        vision_indices_np = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        vision_mask_np = np.zeros((B, T), dtype=np.int32)
        vision_mask_np[0, :8] = 1

        # Keras reference: run the vision encoder inline, exactly as the
        # combined (non-separate) path does.
        with torch.no_grad():
            keras_img_embeddings = backbone.vision_encoder(
                torch.from_numpy(images_np)
            )
            keras_logits, _, keras_cache = model.call_with_cache(
                torch.from_numpy(tokens_np),
                torch.from_numpy(cache_keras),
                0,
                img_embeddings=keras_img_embeddings,
                vision_mask=torch.from_numpy(vision_mask_np),
                padding_mask=None,
                vision_indices=torch.from_numpy(vision_indices_np),
                cache_update_mask=None,
            )
        keras_img_embeddings = keras_img_embeddings.detach().cpu().numpy()
        keras_cache = keras_cache.detach().cpu().numpy()

        # TFLite: chain vision_encoder -> vision_adapter -> prefill.
        vision_encoder_runner = vision_encoder_interpreter.get_signature_runner(
            "vision_encoder"
        )
        tflite_features = vision_encoder_runner(images=images_np)["features"]
        # The Gemma3 vision encoder is not a single-image encoder, so its
        # output is already (batch, num_images, tokens_per_image, dim);
        # collapse the leading two dims to match the vision_adapter's
        # expected (batch * num_images, ...) input, mirroring
        # KerasHubVisionEncoderAdapter's contract.
        tflite_features_flat = tflite_features.reshape(
            -1, tflite_features.shape[-2], tflite_features.shape[-1]
        )
        vision_adapter_runner = vision_adapter_interpreter.get_signature_runner(
            "vision_adapter"
        )
        tflite_mm_embedding = vision_adapter_runner(
            features=tflite_features_flat
        )["mm_embedding"]

        # Vision-tower parity: the TFLite vision encoder's raw features
        # should match Keras's, before any language-model computation
        # amplifies (or hides) a divergence.
        self.assertAllClose(
            keras_img_embeddings.reshape(tflite_features.shape),
            tflite_features,
            atol=1e-4,
            rtol=1e-4,
        )

        prefill_runner = prefill_decode_interpreter.get_signature_runner(
            "prefill"
        )
        prefill_inputs = {
            "tokens": tokens_np,
            "input_pos": np.arange(T, dtype=np.int32),
            # The prefill signature's mm_embedding input is
            # (max_images, tokens_per_image, dim), matching
            # tflite_mm_embedding's shape directly -- no reshape needed.
            "mm_embedding": tflite_mm_embedding,
            "vision_indices": vision_indices_np,
            "vision_mask": vision_mask_np,
        }
        for i in range(L):
            prefill_inputs[f"kv_cache_k_{i}"] = cache_keras[:, i, 0, ...]
            prefill_inputs[f"kv_cache_v_{i}"] = cache_keras[:, i, 1, ...]
        tflite_prefill_out = prefill_runner(**prefill_inputs)

        # End-to-end parity: KV caches after prefilling through the full
        # separate vision_encoder -> vision_adapter -> prefill chain should
        # match the Keras eager reference.
        for i in range(L):
            self.assertAllClose(
                keras_cache[:, i, 0, ...],
                tflite_prefill_out[f"kv_cache_k_{i}"],
                atol=1e-3,
                rtol=1e-3,
            )
            self.assertAllClose(
                keras_cache[:, i, 1, ...],
                tflite_prefill_out[f"kv_cache_v_{i}"],
                atol=1e-3,
                rtol=1e-3,
            )

    def _litertlm_section_model_types(self, path):
        """Return the ``model_type`` string of every section in a bundle.

        Reads the same ``LiteRTLMMetaData`` flatbuffer
        ``_extract_litertlm_tflite_interpreters`` parses, but instead of
        materializing each TFLite payload, decodes every section object's
        ``Items()`` key/value pairs and returns the ``model_type`` value
        (e.g. ``"tf_lite_prefill_decode"``, ``"tf_lite_end_of_vision"`` --
        the ``TfLiteModelType.value`` strings ``add_tflite_model`` writes,
        not the bare enum names) for every section that has one. This is a
        section-*existence* assertion that does not depend on TFLite
        signature names, which matters here since an end-of-image model's
        signature name is fixed by this exporter (``"end_of_vision"``) but
        a signature-name check alone would not distinguish "the model got
        bundled" from "the model got bundled as the right section type".
        """
        from litert_lm_builder import (
            litertlm_header_schema_py_generated as schema_mod,
        )

        _, metadata = self._parse_litertlm_bundle(path)
        section_metadata = metadata.SectionMetadata()
        model_types = []
        for i in range(section_metadata.ObjectsLength()):
            obj = section_metadata.Objects(i)
            for j in range(obj.ItemsLength()):
                kv = obj.Items(j)
                key = kv.Key()
                key_str = key.decode() if isinstance(key, bytes) else key
                if key_str != "model_type":
                    continue
                if kv.ValueType() != schema_mod.VData.StringValue:
                    continue
                value_table = kv.Value()
                string_value = schema_mod.StringValue()
                string_value.Init(value_table.Bytes, value_table.Pos)
                value = string_value.Value()
                model_types.append(
                    value.decode() if isinstance(value, bytes) else value
                )
        return model_types

    def test_export_separate_vision_encoder_has_end_of_vision_section(self):
        """A separate-vision Gemma3 export gains an END_OF_VISION section.

        Before WS1.4, a separate-vision-encoder bundle's TFLite sections
        were exactly ``{PREFILL_DECODE, VISION_ENCODER, VISION_ADAPTER}``
        (see ``test_export_separate_vision_encoder_gemma3`` above, which
        never asserts an END_OF_VISION section because none existed) --
        this test builds the identical bundle and asserts the new
        ``tf_lite_end_of_vision`` section is now present alongside the
        three that were already there, mirroring litert-torch's
        `export_hf` module packing an optional ``eoi.tflite`` model (see
        `KerasHubEndOfImageAdapter` in ``adapter.py``).
        """
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
        from keras_hub.src.tests.mocks.mock_gemma3_tokenizer import (
            MockGemma3Tokenizer,
        )

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
            max_images_per_prompt=2,
            num_vision_tokens_per_image=4,
        )
        vision_encoder = Gemma3VisionEncoder(
            image_size=16,
            patch_size=4,
            pool_size=2,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )
        backbone = Gemma3Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=16,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            vision_encoder=vision_encoder,
        )
        model = Gemma3CausalLM(preprocessor=preprocessor, backbone=backbone)
        self._set_random_weights(model)

        path = os.path.join(
            self.get_temp_dir(), "test_separate_vision_eoi.litertlm"
        )
        model.export(
            path,
            format="litertlm",
            prefill_seq_len=20,
            separate_vision_encoder=True,
        )

        section_model_types = self._litertlm_section_model_types(path)
        self.assertIn("tf_lite_end_of_vision", section_model_types)
        # The ordinary sections from the pre-WS1.4 bundle are still present
        # unchanged -- adding END_OF_VISION does not remove or replace any
        # of them.
        self.assertIn("tf_lite_prefill_decode", section_model_types)
        self.assertIn("tf_lite_vision_encoder", section_model_types)
        self.assertIn("tf_lite_vision_adapter", section_model_types)

        # The END_OF_VISION model itself: an input-less "end_of_vision"
        # signature producing a single "eoi_embedding" output, matching
        # `KerasHubEndOfImageAdapter.forward`.
        interpreters = self._extract_litertlm_tflite_interpreters(path)
        all_signatures = {}
        for interpreter in interpreters:
            all_signatures.update(interpreter._get_full_signature_list())
        self.assertIn("end_of_vision", all_signatures)
        eoi_signature = all_signatures["end_of_vision"]
        self.assertEqual(eoi_signature["inputs"], {})
        self.assertIn("eoi_embedding", eoi_signature["outputs"])

        interpreters_by_sig = {}
        for interpreter in interpreters:
            for sig_name in interpreter._get_full_signature_list():
                interpreters_by_sig[sig_name] = interpreter
        eoi_runner = interpreters_by_sig["end_of_vision"].get_signature_runner(
            "end_of_vision"
        )
        eoi_out = eoi_runner()["eoi_embedding"]

        # Numeric parity: the bundled embedding should be exactly the
        # Keras reference's `token_embedding` lookup for Gemma3's
        # end-of-image token id, the same value
        # `KerasHubEndOfImageAdapter.forward` computes at trace time.
        from keras_hub.src.utils.litertlm.model_specs import Gemma3Spec

        eoi_token_ids = Gemma3Spec().get_end_of_vision_token_ids(tokenizer)
        self.assertIsNotNone(eoi_token_ids)
        with torch.no_grad():
            keras_eoi_embedding = backbone.token_embedding(
                torch.tensor([eoi_token_ids], dtype=torch.int32)
            )
        self.assertAllClose(
            keras_eoi_embedding.detach().cpu().numpy(),
            eoi_out,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_export_multimodal_outputs_match_keras(self):
        """Verify multimodal Keras eager and TFLite outputs match."""
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
        from keras_hub.src.tests.mocks.mock_gemma3_tokenizer import (
            MockGemma3Tokenizer,
        )

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
            max_images_per_prompt=2,
            num_vision_tokens_per_image=4,
        )
        vision_encoder = Gemma3VisionEncoder(
            image_size=16,
            patch_size=4,
            pool_size=2,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )
        backbone = Gemma3Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=16,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            vision_encoder=vision_encoder,
        )
        model = Gemma3CausalLM(preprocessor=preprocessor, backbone=backbone)
        self._set_random_weights(model)

        # Export
        litertlm_path = os.path.join(
            self.get_temp_dir(), "verify_multimodal.litertlm"
        )
        model.export(litertlm_path, format="litertlm", prefill_seq_len=20)

        # Extract TFLite
        interpreter = self._extract_litertlm_tflite_interpreters(litertlm_path)[
            0
        ]

        B, T, L = 1, 20, 2
        H = backbone.num_key_value_heads
        D = backbone.head_dim
        tokens_np = (
            np.arange(1, 1 + T, dtype=np.int32).reshape(B, T)
            % tokenizer.vocabulary_size()
        )
        cache_keras = np.zeros((B, L, 2, T, H, D), dtype=np.float32)

        # Preprocess images to get vision inputs.
        images_np = np.ones((B, 2, 16, 16, 3), dtype=np.float32)
        vision_indices_np = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        vision_mask_np = np.zeros((B, T), dtype=np.int32)
        vision_mask_np[0, :8] = 1

        # Run vision encoder for Keras reference.
        with torch.no_grad():
            img_embeddings = backbone.vision_encoder(
                torch.from_numpy(images_np)
            )

        # Keras prefill
        with torch.no_grad():
            keras_logits, _, keras_cache = model.call_with_cache(
                torch.from_numpy(tokens_np),
                torch.from_numpy(cache_keras),
                0,
                img_embeddings=img_embeddings,
                vision_mask=torch.from_numpy(vision_mask_np),
                padding_mask=None,
                vision_indices=torch.from_numpy(vision_indices_np),
                cache_update_mask=None,
            )
        keras_logits = keras_logits.detach().cpu().numpy()
        keras_cache = keras_cache.detach().cpu().numpy()

        # TFLite prefill
        prefill_runner = interpreter.get_signature_runner("prefill")
        prefill_inputs = {
            "tokens": tokens_np,
            "input_pos": np.arange(T, dtype=np.int32),
            "images": images_np,
            "vision_indices": vision_indices_np,
            "vision_mask": vision_mask_np,
        }
        for i in range(L):
            prefill_inputs[f"kv_cache_k_{i}"] = cache_keras[:, i, 0, ...]
            prefill_inputs[f"kv_cache_v_{i}"] = cache_keras[:, i, 1, ...]
        tflite_prefill_out = prefill_runner(**prefill_inputs)

        # Compare prefill KV caches. Measured max abs diff on this tiny
        # random-init model is ~1e-6 (see git history for the measurement);
        # 1e-4 (matching the plain text-only parity test) leaves ~100x
        # margin while still catching real regressions.
        for i in range(L):
            self.assertAllClose(
                keras_cache[:, i, 0, ...],
                tflite_prefill_out[f"kv_cache_k_{i}"],
                atol=1e-4,
                rtol=1e-4,
            )
            self.assertAllClose(
                keras_cache[:, i, 1, ...],
                tflite_prefill_out[f"kv_cache_v_{i}"],
                atol=1e-4,
                rtol=1e-4,
            )

        # Keras decode at position 3 (no images needed)
        decode_pos = 3
        decode_token = tokens_np[:, decode_pos : decode_pos + 1].copy()
        with torch.no_grad():
            keras_logits_dec, _, keras_cache_dec = model.call_with_cache(
                torch.from_numpy(decode_token),
                torch.from_numpy(keras_cache),
                decode_pos,
                img_embeddings=None,
                vision_mask=None,
                padding_mask=None,
                vision_indices=None,
                cache_update_mask=None,
            )
        keras_logits_dec = keras_logits_dec.detach().cpu().numpy()
        keras_cache_dec = keras_cache_dec.detach().cpu().numpy()

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

        # Compare decode logits. This was previously atol=rtol=5e-2 with a
        # comment claiming vision-conditioned activations amplify small
        # attention-algorithm differences enough to require it; measured
        # directly, the actual max abs diff on this tiny random-init model is
        # ~1e-6, so 1e-4 (matching the plain text-only parity test) is well
        # justified and catches regressions 500x smaller than before.
        self.assertAllClose(
            keras_logits_dec,
            tflite_dec_out["logits"],
            atol=1e-4,
            rtol=1e-4,
        )

        # Compare decode KV caches (same measured margin as above).
        for i in range(L):
            self.assertAllClose(
                keras_cache_dec[:, i, 0, ...],
                tflite_dec_out[f"kv_cache_k_{i}"],
                atol=1e-4,
                rtol=1e-4,
            )
            self.assertAllClose(
                keras_cache_dec[:, i, 1, ...],
                tflite_dec_out[f"kv_cache_v_{i}"],
                atol=1e-4,
                rtol=1e-4,
            )

    def test_export_gpt2_with_auto_hf_tokenizer(self):
        """Export a tiny GPT2 model with auto-converted HF tokenizer."""
        vocab = {
            "<|endoftext|>": 0,
            "h": 1,
            "i": 2,
            "Ġ": 3,
            "Ġh": 4,
            "e": 5,
            "l": 6,
            "o": 7,
            "w": 8,
            "r": 9,
            "d": 10,
            "t": 11,
            "s": 12,
            "a": 13,
            "b": 14,
            "ab": 15,
            "n": 16,
            "k": 17,
            "u": 18,
            "m": 19,
        }
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

        self._verify_litertlm_generation(path, prompt="hi", max_num_tokens=4)

    def test_export_llama3_with_auto_hf_tokenizer(self):
        """Export a tiny Llama3 model with auto-converted HF tokenizer."""
        vocab = {
            "<|endoftext|>": 0,
            "<|begin_of_text|>": 1,
            "<|end_of_text|>": 2,
            "<|start_header_id|>": 3,
            "<|end_header_id|>": 4,
            "<|eot_id|>": 5,
            "h": 6,
            "i": 7,
            "Ġ": 8,
            "Ġh": 9,
            "e": 10,
            "l": 11,
            "o": 12,
            "w": 13,
            "r": 14,
            "d": 15,
            "t": 16,
            "s": 17,
            "a": 18,
            "b": 19,
            "ab": 20,
            "n": 21,
            "k": 22,
            "u": 23,
            "m": 24,
        }
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

        self._verify_litertlm_generation(path, prompt="hi", max_num_tokens=4)

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
        vocab = {
            "<|endoftext|>": 0,
            "<|im_end|>": 1,
            "h": 2,
            "i": 3,
            "Ġ": 4,
            "Ġh": 5,
            "e": 6,
            "l": 7,
            "o": 8,
            "w": 9,
            "r": 10,
            "d": 11,
            "t": 12,
            "s": 13,
            "a": 14,
            "b": 15,
            "ab": 16,
            "n": 17,
            "k": 18,
            "u": 19,
            "m": 20,
        }
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

        self._verify_litertlm_generation(path, prompt="hi", max_num_tokens=4)


class TestLiteRTLmAdapterHelpers(TestCase):
    def test_cpu_default_device_scope_restores_device(self):
        """_cpu_default_device_scope restores the original default device."""
        original = torch.get_default_device()
        with _cpu_default_device_scope():
            self.assertEqual(torch.get_default_device(), torch.device("cpu"))
        self.assertEqual(torch.get_default_device(), original)

    def test_prepare_image_embeddings_raw_images_rejects_pixel_values(self):
        """A raw_images spec fed pixel_values raises the typed mismatch error.

        WS1.2: the adapter dispatches on spec.vision_input_style, and a
        style/args disagreement is a hard error (previously a silent
        `return None, None`).
        """
        from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
        from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM
        from keras_hub.src.models.gemma3.gemma3_vision_encoder import (
            Gemma3VisionEncoder,
        )
        from keras_hub.src.utils.litertlm.adapter import KerasHubLiteRTAdapter
        from keras_hub.src.utils.litertlm.model_specs import resolve_export_spec

        vision_encoder = Gemma3VisionEncoder(
            image_size=16,
            patch_size=4,
            pool_size=2,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )
        backbone = Gemma3Backbone(
            vocabulary_size=32,
            image_size=16,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            vision_encoder=vision_encoder,
        )
        model = Gemma3CausalLM(preprocessor=None, backbone=backbone)
        spec = resolve_export_spec(model)
        self.assertEqual(spec.vision_input_style, "raw_images")

        num_layers, cache_length = 2, 20
        adapter = KerasHubLiteRTAdapter(
            model, num_layers, cache_length, export_spec=spec
        )
        tokens = torch.zeros((1, 4), dtype=torch.int32)
        pixel_values = torch.zeros((1, 1, 4, 48), dtype=torch.float32)

        with self.assertRaisesRegex(
            ValueError, "vision_input_style='raw_images' requires"
        ):
            adapter._prepare_image_embeddings(
                tokens=tokens,
                images=None,
                pixel_values=pixel_values,
                pixel_position_ids=None,
                mm_embedding=None,
            )

    def test_vision_encoder_adapter_dispatches_on_spec_style(self):
        """KerasHubVisionEncoderAdapter.forward is keyed on the spec style,
        not on which argument is supplied: feeding the wrong tensor for the
        declared style raises the typed mismatch error before the encoder is
        ever called (WS1.2 item a).

        Uses a stub backbone whose vision_encoder is a sentinel that would
        raise if invoked -- proving the dispatch rejects the call up front,
        without needing a real (expensive) Functional encoder.
        """
        from keras_hub.src.utils.litertlm.adapter import (
            KerasHubVisionEncoderAdapter,
        )

        def _never_call(*args, **kwargs):
            raise AssertionError(
                "vision_encoder must not be called on a style/args mismatch"
            )

        backbone = types.SimpleNamespace(vision_encoder=_never_call)
        keras_model = types.SimpleNamespace(backbone=backbone)

        # raw_images adapter fed pixel_values (wrong) -> raises.
        raw_adapter = KerasHubVisionEncoderAdapter(
            keras_model,
            vision_input_style="raw_images",
            flatten_image_batch=False,
        )
        with self.assertRaisesRegex(
            ValueError, "vision_input_style='raw_images' requires"
        ):
            raw_adapter.forward(
                pixel_values=torch.zeros((1, 1, 4, 48), dtype=torch.float32),
                pixel_position_ids=torch.zeros((1, 1, 4), dtype=torch.int32),
            )

        # patch_values adapter fed images (wrong) -> raises.
        patch_adapter = KerasHubVisionEncoderAdapter(
            keras_model,
            vision_input_style="patch_values",
            flatten_image_batch=False,
        )
        with self.assertRaisesRegex(
            ValueError, "vision_input_style='patch_values' requires"
        ):
            patch_adapter.forward(
                images=torch.zeros((1, 1, 16, 16, 3), dtype=torch.float32),
            )


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
        import keras

        if keras.config.backend() != "torch":
            self.skipTest(
                "BytePair tokenizer roundtrip requires torch backend."
            )

        vocab = {
            "<|endoftext|>": 0,
            "h": 1,
            "i": 2,
            "Ġ": 3,
            "Ġh": 4,
            "e": 5,
            "l": 6,
            "o": 7,
            "w": 8,
            "r": 9,
            "d": 10,
            "t": 11,
            "s": 12,
            "a": 13,
            "b": 14,
            "ab": 15,
            "hello": 16,
            "Ġworld": 17,
        }
        merges = ["a b"]
        tokenizer = GPT2Tokenizer(vocabulary=vocab, merges=merges)

        hf_dict = convert_byte_pair_to_hf(tokenizer)
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as f:
            json.dump(hf_dict, f, ensure_ascii=False, indent=2)
            hf_tokenizer_path = f.name

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
