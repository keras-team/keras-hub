import importlib
import types

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM
from keras_hub.src.models.gemma4.gemma4_assistant_causal_lm import (
    Gemma4AssistantCausalLM,
)
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.llama.llama_backbone import LlamaBackbone
from keras_hub.src.models.llama.llama_causal_lm import LlamaCausalLM
from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.models.llama3.llama3_causal_lm import Llama3CausalLM
from keras_hub.src.models.phi3.phi3_backbone import Phi3Backbone
from keras_hub.src.models.phi3.phi3_causal_lm import Phi3CausalLM
from keras_hub.src.models.qwen.qwen_backbone import QwenBackbone
from keras_hub.src.models.qwen.qwen_causal_lm import QwenCausalLM
from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.models.qwen3.qwen3_causal_lm import Qwen3CausalLM
from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_causal_lm import Qwen3_5CausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.litertlm.model_specs import _EXPORT_SPEC_REGISTRY
from keras_hub.src.utils.litertlm.model_specs import FunctionGemmaSpec
from keras_hub.src.utils.litertlm.model_specs import Gemma3nSpec
from keras_hub.src.utils.litertlm.model_specs import Gemma3Spec
from keras_hub.src.utils.litertlm.model_specs import Gemma4AssistantSpec
from keras_hub.src.utils.litertlm.model_specs import Gemma4Spec
from keras_hub.src.utils.litertlm.model_specs import GemmaSpec
from keras_hub.src.utils.litertlm.model_specs import LiteRTLMExportSpec
from keras_hub.src.utils.litertlm.model_specs import Llama3Spec
from keras_hub.src.utils.litertlm.model_specs import PaliGemmaSpec
from keras_hub.src.utils.litertlm.model_specs import Phi3Spec
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

        `resolve_export_spec` imports each entry lazily and deliberately does
        not catch `ImportError`, so a typo'd `module_path` or `class_name`
        breaks resolution for every model reaching that entry. Import each
        entry directly here (not through `resolve_export_spec`), so a broken
        entry is reported by name instead of surfacing as an unrelated
        family's export failure.
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

    def _tiny_phi3(self):
        backbone = Phi3Backbone(
            vocabulary_size=10,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
        )
        return Phi3CausalLM(backbone=backbone)

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

    def _tiny_gemma3(self):
        # Text-only Gemma3 (`vision_encoder=None`); `resolve_export_spec` only
        # does `isinstance`, so no preprocessor or real weights are needed.
        backbone = Gemma3Backbone(
            vocabulary_size=10,
            image_size=16,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            head_dim=4,
            intermediate_dim=16,
            vision_encoder=None,
        )
        # `Gemma3CausalLM` requires `preprocessor`, but `resolve_export_spec`
        # only does an `isinstance` check, so a null preprocessor is fine.
        return Gemma3CausalLM(preprocessor=None, backbone=backbone)

    def _tiny_gemma4_assistant(self):
        # Mirrors the tiny config in `gemma4_assistant_causal_lm_test.py`.
        backbone = Gemma4Backbone(
            vocabulary_size=256,
            num_layers=4,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            global_head_dim=8,
            image_size=16,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        return Gemma4AssistantCausalLM(
            preprocessor=None,
            backbone=backbone,
            backbone_hidden_size=16,
            num_centroids=4,
            centroid_intermediate_top_k=2,
            use_ordered_embeddings=True,
        )

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

    def test_gemma3_resolves_to_gemma3_spec_by_default(self):
        """A plain Gemma3 (no override) resolves to `Gemma3Spec` -- the
        regression guard that the `function_gemma` override never leaks into
        ordinary Gemma3 exports."""
        spec = resolve_export_spec(self._tiny_gemma3())
        self.assertIsInstance(spec, Gemma3Spec)
        self.assertNotIsInstance(spec, FunctionGemmaSpec)
        self.assertEqual(spec.model_type, "gemma3")

    def test_function_gemma_override_resolves_to_function_gemma_spec(self):
        """The explicit `llm_model_type="function_gemma"` override selects
        `FunctionGemmaSpec` (`model_type="function_gemma"`), even though the
        model is a plain `Gemma3CausalLM` that would otherwise resolve to
        `Gemma3Spec`."""
        model = self._tiny_gemma3()
        self.assertEqual(resolve_export_spec(model).model_type, "gemma3")
        spec = resolve_export_spec(model, llm_model_type="function_gemma")
        self.assertIsInstance(spec, FunctionGemmaSpec)
        self.assertEqual(spec.model_type, "function_gemma")

    def test_function_gemma_spec_not_in_isinstance_registry(self):
        """`FunctionGemmaSpec` must NOT be in `_EXPORT_SPEC_REGISTRY`: an
        `isinstance` entry would shadow `Gemma3Spec` for every Gemma3 model."""
        self.assertNotIn(
            FunctionGemmaSpec, [f for _, _, f in _EXPORT_SPEC_REGISTRY]
        )

    def test_function_gemma_auto_detected_by_tokenizer_tokens(self):
        """A `Gemma3CausalLM` whose tokenizer exposes function-calling special
        tokens (`<start_function_call>`) auto-resolves to `FunctionGemmaSpec`
        even without an explicit `llm_model_type` override."""

        class MockTokenizer:
            def token_to_id(self, token):
                if token == "<start_function_call>":
                    return 48
                return -1

            def id_to_token(self, token_id):
                if token_id == 48:
                    return "<start_function_call>"
                return "<unk>"

        class MockPreprocessor:
            tokenizer = MockTokenizer()

        model = self._tiny_gemma3()
        model.preprocessor = MockPreprocessor()
        spec = resolve_export_spec(model)
        self.assertIsInstance(spec, FunctionGemmaSpec)
        self.assertEqual(spec.model_type, "function_gemma")

    def test_unknown_llm_model_type_override_raises(self):
        """An unrecognized `llm_model_type` override is a hard error, not a
        silent fall-through to isinstance resolution."""
        model = self._tiny_gemma3()
        with self.assertRaisesRegex(ValueError, "Unknown `llm_model_type`"):
            resolve_export_spec(model, llm_model_type="not_a_real_type")

    def test_llama3_resolves_to_llama3_spec(self):
        """`Llama3CausalLM` is a subclass of `LlamaCausalLM`; it must resolve
        to `Llama3Spec` (registered earlier in `_EXPORT_SPEC_REGISTRY`), not
        fall through to the plain `LlamaCausalLM` entry."""
        spec = resolve_export_spec(self._tiny_llama3())
        self.assertIsInstance(spec, Llama3Spec)
        self.assertEqual(spec.model_type, "generic_model")

    def test_phi3_causal_lm_resolves_to_phi3_spec(self):
        """`Phi3CausalLM` must resolve to `Phi3Spec` (not fall through to
        the plain `LiteRTLMExportSpec` default), so its `<|end|>`
        chat-stop-token override actually fires."""
        spec = resolve_export_spec(self._tiny_phi3())
        self.assertIsInstance(spec, Phi3Spec)
        self.assertEqual(spec.model_type, "generic_model")

    def test_gemma4_assistant_check_exportable_raises(self):
        """`Gemma4AssistantCausalLM` resolves to `Gemma4AssistantSpec`, whose
        `check_exportable` fails fast with the MTP-draft explanation instead
        of letting the model fall through to the generic spec and crash in
        tracing. The base-class gate is a no-op for ordinary models."""
        model = self._tiny_gemma4_assistant()
        spec = resolve_export_spec(model)
        self.assertIs(type(spec), Gemma4AssistantSpec)
        with self.assertRaisesRegex(
            ValueError,
            "does not support `Gemma4AssistantCausalLM`.*"
            "multi-token-prediction",
        ):
            spec.check_exportable(model)
