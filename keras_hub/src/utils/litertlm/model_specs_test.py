"""Registry-integrity tests for `model_specs.py`'s `_EXPORT_SPEC_REGISTRY`.

These tests are intentionally dependency-free -- no torch, no litert-torch,
no litert-lm-builder -- so they run on every backend/CI leg, not just the
torch-backend leg the rest of the litertlm test suite is gated on (see
`export_test.py`, `qwen3_5_causal_lm_test.py`). `model_specs.py` itself has no
external imports at module level, and `resolve_export_spec` only attempts to
import the actual model classes it checks `isinstance` against, swallowing
`ImportError` per entry -- correct/necessary at runtime, since many entries
reference optional model classes, but it also means a typo'd `module_path` or
`class_name` in `_EXPORT_SPEC_REGISTRY` would silently make that family
permanently fall back to the generic spec, with nothing catching it. These
tests catch exactly that.
"""

import importlib
import types

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.models.llama.llama_backbone import LlamaBackbone
from keras_hub.src.models.llama.llama_causal_lm import LlamaCausalLM
from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.models.llama3.llama3_causal_lm import Llama3CausalLM
from keras_hub.src.models.qwen.qwen_backbone import QwenBackbone
from keras_hub.src.models.qwen.qwen_causal_lm import QwenCausalLM
from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.models.qwen3.qwen3_causal_lm import Qwen3CausalLM
from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_causal_lm import Qwen3_5CausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.litertlm.model_specs import _EXPORT_SPEC_REGISTRY
from keras_hub.src.utils.litertlm.model_specs import Gemma3nSpec
from keras_hub.src.utils.litertlm.model_specs import Gemma3Spec
from keras_hub.src.utils.litertlm.model_specs import Gemma4Spec
from keras_hub.src.utils.litertlm.model_specs import GemmaSpec
from keras_hub.src.utils.litertlm.model_specs import LiteRTLMExportSpec
from keras_hub.src.utils.litertlm.model_specs import PaliGemmaSpec
from keras_hub.src.utils.litertlm.model_specs import Llama3Spec
from keras_hub.src.utils.litertlm.model_specs import Qwen2p5FamilySpec
from keras_hub.src.utils.litertlm.model_specs import Qwen3_5Spec
from keras_hub.src.utils.litertlm.model_specs import Qwen3FamilySpec
from keras_hub.src.utils.litertlm.model_specs import resolve_export_spec


class ExportSpecRegistryIntegrityTest(TestCase):
    """Walk `_EXPORT_SPEC_REGISTRY` and verify every entry actually resolves.

    Deliberately has no torch/litert_torch dependency and no backend
    requirement: it only imports plain Keras model-definition modules (which
    build fine under any Keras backend) plus `model_specs.py` itself (which
    has no external imports at all).
    """

    def test_every_registry_entry_imports_and_resolves(self):
        """Every `(module_path, class_name, spec_factory)` entry must import.

        `resolve_export_spec` deliberately swallows `ImportError` per entry
        so an unavailable optional model class doesn't break resolution for
        every other family -- but that means a typo'd `module_path` or
        `class_name` would silently and permanently fall back to the base
        spec, with no test noticing. Import each entry directly here (not
        through `resolve_export_spec`), so a broken entry fails loudly.
        """
        self.assertTrue(_EXPORT_SPEC_REGISTRY, "Registry must not be empty.")
        for module_path, class_name, spec_factory in _EXPORT_SPEC_REGISTRY:
            with self.subTest(module_path=module_path, class_name=class_name):
                module = importlib.import_module(module_path)
                self.assertTrue(
                    hasattr(module, class_name),
                    f"{module_path!r} has no attribute {class_name!r} -- "
                    "check for a typo in _EXPORT_SPEC_REGISTRY.",
                )
                cls = getattr(module, class_name)
                self.assertTrue(
                    isinstance(cls, type),
                    f"{module_path}.{class_name} is not a class.",
                )
                spec = spec_factory()
                self.assertIsInstance(spec, LiteRTLMExportSpec)

    # -- Representative per-family resolution ------------------------------
    #
    # Tiny, randomly-initialized instances, matching the pattern every
    # `*_causal_lm_test.py` in this repo already uses for cheap model
    # construction. `resolve_export_spec` only performs `isinstance` checks,
    # so no preprocessor or real weights are needed.

    def _tiny_llama(self):
        backbone = LlamaBackbone(
            vocabulary_size=10,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
        )
        return LlamaCausalLM(backbone=backbone)

    def _tiny_llama3(self):
        backbone = Llama3Backbone(
            vocabulary_size=10,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
        )
        return Llama3CausalLM(backbone=backbone)

    def _tiny_gemma(self):
        backbone = GemmaBackbone(
            vocabulary_size=10,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            head_dim=4,
            intermediate_dim=16,
        )
        return GemmaCausalLM(backbone=backbone)

    def _tiny_qwen(self):
        backbone = QwenBackbone(
            vocabulary_size=10,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
        )
        return QwenCausalLM(backbone=backbone)

    def _tiny_qwen3(self):
        backbone = Qwen3Backbone(
            vocabulary_size=10,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            head_dim=4,
            intermediate_dim=16,
        )
        return Qwen3CausalLM(backbone=backbone)

    def _tiny_qwen3_5(self):
        backbone = Qwen3_5Backbone(
            vocabulary_size=10,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            head_dim=8,
            intermediate_dim=16,
            layer_types=["linear_attention", "full_attention"],
            partial_rotary_factor=0.25,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=4,
        )
        return Qwen3_5CausalLM(backbone=backbone)

    def test_llama_resolves_to_base_generic_spec(self):
        """Llama is explicitly registered but maps to the base spec/class
        (see the `LlamaCausalLM` entry's NOTE in `_EXPORT_SPEC_REGISTRY`)."""
        spec = resolve_export_spec(self._tiny_llama())
        self.assertIs(type(spec), LiteRTLMExportSpec)
        self.assertEqual(spec.model_type, "generic_model")
        self.assertEqual(spec.cache_structure, "single_stacked")

    def test_qwen_resolves_to_qwen2p5_family_spec(self):
        spec = resolve_export_spec(self._tiny_qwen())
        self.assertIsInstance(spec, Qwen2p5FamilySpec)
        self.assertEqual(spec.model_type, "qwen2p5")

    def test_qwen3_resolves_to_qwen3_family_spec(self):
        spec = resolve_export_spec(self._tiny_qwen3())
        self.assertIsInstance(spec, Qwen3FamilySpec)
        self.assertNotIsInstance(spec, Qwen3_5Spec)
        self.assertEqual(spec.model_type, "qwen3")
        self.assertEqual(spec.cache_structure, "single_stacked")

    def test_qwen3_5_resolves_to_hybrid_cache_spec(self):
        """Regression coverage for the `cache_structure` fix: Qwen3.5 must
        resolve to its own `Qwen3_5Spec` (not the shared `Qwen3FamilySpec`
        every other Qwen3-family model uses), so `export_to_litertlm`'s
        `cache_structure` fail-fast check actually fires for it.

        This is deliberately not a duplicate of
        `test_litertlm_model_type_detection` in `qwen3_5_causal_lm_test.py`:
        that test only checks `model_type` and requires the torch backend;
        this one also checks spec class identity and `cache_structure`, and
        needs neither torch nor any litertlm dependency.
        """
        spec = resolve_export_spec(self._tiny_qwen3_5())
        self.assertIsInstance(spec, Qwen3_5Spec)
        self.assertEqual(spec.model_type, "qwen3")
        self.assertEqual(spec.cache_structure, "hybrid")

    def test_gemma_resolves_to_gemma_spec(self):
        """Base Gemma has no dedicated `LlmModelType` subtype, but must still
        resolve to `GemmaSpec` (not the plain `LiteRTLMExportSpec` fallback)
        to get the shared Gemma-family `<end_of_turn>` chat-stop-token
        behavior."""
        spec = resolve_export_spec(self._tiny_gemma())
        self.assertIsInstance(spec, GemmaSpec)
        self.assertEqual(spec.model_type, "generic_model")

    def test_llama3_resolves_to_llama3_spec(self):
        """`Llama3CausalLM` is a subclass of `LlamaCausalLM`; it must resolve
        to `Llama3Spec` (registered earlier in `_EXPORT_SPEC_REGISTRY`), not
        fall through to the plain `LlamaCausalLM` entry."""
        spec = resolve_export_spec(self._tiny_llama3())
        self.assertIsInstance(spec, Llama3Spec)
        self.assertEqual(spec.model_type, "generic_model")

    # -- Chat-turn stop-token overrides -------------------------------------
    #
    # Dependency-free checks of `get_chat_stop_token_ids` against small fake
    # tokenizer objects -- no real KerasHub tokenizer or torch/litert
    # dependency needed, since the method only calls `token_to_id`/reads
    # plain attributes.

    def test_gemma_spec_chat_stop_token_ids_looks_up_end_of_turn(self):
        vocab = {"<end_of_turn>": 7}
        tokenizer = types.SimpleNamespace(
            token_to_id=vocab.__getitem__, _unk_token_id=0
        )
        self.assertEqual(GemmaSpec().get_chat_stop_token_ids(tokenizer), [7])

    def test_gemma_spec_chat_stop_token_ids_absent_returns_empty(self):
        tokenizer = types.SimpleNamespace(
            token_to_id=lambda token: (_ for _ in ()).throw(KeyError(token))
        )
        self.assertEqual(GemmaSpec().get_chat_stop_token_ids(tokenizer), [])

    def test_llama3_spec_chat_stop_token_ids_uses_end_token2_id(self):
        """`Llama3Tokenizer` stores `<|eot_id|>` as `end_token2_id` (see the
        "Hack" comment in `llama3_tokenizer.py`), not via `token_to_id`
        lookup -- `Llama3Spec` must read that attribute directly."""
        tokenizer = types.SimpleNamespace(end_token2_id=5)
        self.assertEqual(Llama3Spec().get_chat_stop_token_ids(tokenizer), [5])

    def test_llama3_spec_chat_stop_token_ids_absent_returns_empty(self):
        """Base Llama (no `end_token2` hack) has no `end_token2_id`
        attribute at all."""
        tokenizer = types.SimpleNamespace()
        self.assertEqual(Llama3Spec().get_chat_stop_token_ids(tokenizer), [])

    def test_qwen3_family_spec_chat_stop_token_ids_looks_up_im_end(self):
        """Qwen3's `<|im_end|>` is already `tokenizer.end_token_id`; this
        override documents that intentionally rather than leaving it as an
        accident of `end_token_id`'s value (`_build_llm_metadata`
        deduplicates the two)."""
        vocab = {"<|im_end|>": 3}
        tokenizer = types.SimpleNamespace(
            token_to_id=vocab.__getitem__, _unk_token_id=0
        )
        self.assertEqual(
            Qwen3FamilySpec().get_chat_stop_token_ids(tokenizer), [3]
        )

    def test_qwen2p5_family_spec_chat_stop_token_ids_absent_by_default(self):
        """Base Qwen (2.5) tokenizers use `<|endoftext|>`, not `<|im_end|>`;
        the override must not invent a token that isn't in vocab."""
        tokenizer = types.SimpleNamespace(
            token_to_id=lambda token: (_ for _ in ()).throw(KeyError(token))
        )
        self.assertEqual(
            Qwen2p5FamilySpec().get_chat_stop_token_ids(tokenizer), []
        )

    # -- Unsupported cache-structure messaging ------------------------------

    def test_base_spec_describes_unsupported_cache_structure_generically(self):
        """The default `describe_unsupported_cache_structure` must name the
        actual mismatched `cache_structure` value generically, without any
        family-specific (e.g. Qwen3.5) text -- that lives on the family's
        own spec (see `Qwen3_5Spec`'s override) instead of leaking into the
        shared/generic path."""

        class _CustomHybridSpec(LiteRTLMExportSpec):
            cache_structure = "custom_hybrid"

        message = _CustomHybridSpec().describe_unsupported_cache_structure()
        self.assertIn("custom_hybrid", message)
        self.assertIn("single_stacked", message)
        self.assertNotIn("Qwen3.5", message)
        self.assertNotIn("Qwen", message)

    def test_qwen3_5_spec_describes_hybrid_cache_specifically(self):
        message = Qwen3_5Spec().describe_unsupported_cache_structure()
        self.assertIn("hybrid full_attention/linear_attention", message)

    # -- Audio input style --------------------------------------------------
    #
    # Every audio-capable family (a spec that overrides
    # `populate_audio_metadata` with real audio tokens) must also declare an
    # `audio_input_style` so the adapter dispatches on the spec instead of
    # sniffing `call_with_cache`'s signature. This is the audio analogue of
    # `vision_input_style` and mirrors the migration vision already got.

    def test_audio_capable_specs_declare_audio_input_style(self):
        """Gemma3n and Gemma4 are the only audio-capable families; each must
        declare a concrete `audio_input_style` matching how its audio encoder
        consumes input (embedded-in-backbone vs standalone in-trace)."""
        self.assertEqual(Gemma3nSpec().audio_input_style, "embedded_mel")
        self.assertEqual(Gemma4Spec().audio_input_style, "standalone_mel")

    def test_non_audio_specs_have_no_audio_input_style(self):
        """The base spec and non-audio families default to `None`, so the
        registry test's audio-capability signal stays a clean non-None check
        (mirrors `vision_input_style` being `None` only via the adapter's
        `has_audio` guard -- here it is the spec-declared default)."""
        self.assertIsNone(LiteRTLMExportSpec().audio_input_style)
        self.assertIsNone(GemmaSpec().audio_input_style)
        self.assertIsNone(Gemma3Spec().audio_input_style)
        self.assertIsNone(PaliGemmaSpec().audio_input_style)

    # -- flatten_image_batch (single-image ViT declaration) ----------------

    def test_single_image_family_declares_flatten_image_batch(self):
        """PaliGemma's ViT is 4-D-only; its spec must declare
        flatten_image_batch=True so the adapter flattens the batched images
        stack instead of sniffing the encoder's Functional input rank."""
        self.assertTrue(PaliGemmaSpec().flatten_image_batch)

    def test_multi_image_families_do_not_flatten(self):
        """Every other vision family accepts the batched stack (or does not
        run the encoder standalone), so flatten_image_batch stays False."""
        self.assertFalse(LiteRTLMExportSpec().flatten_image_batch)
        self.assertFalse(Gemma3Spec().flatten_image_batch)
        self.assertFalse(Gemma3nSpec().flatten_image_batch)
        self.assertFalse(Gemma4Spec().flatten_image_batch)

    # -- get_max_images_per_prompt (explicit, no silent default) -----------

    def test_max_images_reads_preprocessor_attribute(self):
        pre = types.SimpleNamespace(max_images_per_prompt=4)
        self.assertEqual(Gemma4Spec().get_max_images_per_prompt(pre), 4)

    def test_max_images_single_image_family_defaults_to_one(self):
        """PaliGemma's preprocessor has no max_images_per_prompt; because it
        declares flatten_image_batch=True, resolving to 1 is legitimate."""
        pre = types.SimpleNamespace()  # no max_images_per_prompt attribute
        self.assertEqual(PaliGemmaSpec().get_max_images_per_prompt(pre), 1)

    def test_max_images_missing_on_multi_image_family_raises(self):
        """A multi-image (flatten_image_batch=False) family with no
        max_images_per_prompt is a misconfiguration -- must raise, not
        silently default to 1 (the old getattr(..., 1) behavior)."""
        pre = types.SimpleNamespace()  # no max_images_per_prompt attribute
        with self.assertRaisesRegex(ValueError, "flatten_image_batch=False"):
            Gemma4Spec().get_max_images_per_prompt(pre)
