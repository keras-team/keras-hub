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
