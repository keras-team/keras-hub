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
        self.assertIsNone(
            LiteRTLMExportSpec().check_exportable(self._tiny_llama())
        )

    def test_llm_model_type_override_cannot_bypass_check_exportable(self):
        """`llm_model_type` selects exported metadata, never exportability.

        The override resolves a spec by name instead of by class, so it must
        not be usable to route a non-exportable model (an MTP draft model) to
        an exportable family's spec and slip past `check_exportable`."""
        model = self._tiny_gemma4_assistant()
        with self.assertRaisesRegex(
            ValueError,
            "does not support `Gemma4AssistantCausalLM`.*"
            "multi-token-prediction",
        ):
            resolve_export_spec(model, llm_model_type="function_gemma")

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

    def test_phi3_spec_chat_stop_token_ids_looks_up_end(self):
        """Phi-3's chat template ends each turn with `<|end|>` (distinct
        from the `<|endoftext|>` EOS); the override must surface it.
        `<|end|>` is an ordinary special token looked up by string, not a
        named tokenizer attribute."""
        vocab = {"<|end|>": 18}
        tokenizer = types.SimpleNamespace(
            token_to_id=vocab.__getitem__, _unk_token_id=0
        )
        self.assertEqual(Phi3Spec().get_chat_stop_token_ids(tokenizer), [18])

    def test_phi3_spec_chat_stop_token_ids_absent_returns_empty(self):
        """Base/non-instruct Phi-3 vocabularies without `<|end|>` get no
        chat-stop token; the override must not invent one."""
        tokenizer = types.SimpleNamespace(
            token_to_id=lambda token: (_ for _ in ()).throw(KeyError(token))
        )
        self.assertEqual(Phi3Spec().get_chat_stop_token_ids(tokenizer), [])

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

    # Every audio-capable family must declare an `audio_input_style`
    # matching how its encoder consumes input.

    def test_audio_capable_specs_declare_audio_input_style(self):
        """Gemma3n and Gemma4 are the only audio-capable families; each must
        declare a concrete `audio_input_style` matching how its audio encoder
        consumes input (embedded-in-backbone vs standalone in-trace)."""
        self.assertEqual(Gemma3nSpec().audio_input_style, "embedded_mel")
        self.assertEqual(Gemma4Spec().audio_input_style, "standalone_mel")

    def test_non_audio_specs_have_no_audio_input_style(self):
        """The base spec and non-audio families default to `None`, so the
        registry test's audio-capability signal stays a clean non-None
        check."""
        self.assertIsNone(LiteRTLMExportSpec().audio_input_style)
        self.assertIsNone(GemmaSpec().audio_input_style)
        self.assertIsNone(Gemma3Spec().audio_input_style)
        self.assertIsNone(PaliGemmaSpec().audio_input_style)

    def test_gemma3n_audio_metadata_pins_gemma4_token_strings(self):
        """Gemma3n's audio metadata deliberately uses Gemma4's token strings
        (golden reference bundles); regression-guard against deriving them
        from Gemma3n's own `<start_of_audio>`/`<end_of_audio>` tokens."""
        meta = types.SimpleNamespace(
            llm_model_type=types.SimpleNamespace(
                gemma3n=types.SimpleNamespace(
                    start_of_audio_token=types.SimpleNamespace(token_str=""),
                    end_of_audio_token=types.SimpleNamespace(token_str=""),
                )
            )
        )
        Gemma3nSpec().populate_audio_metadata(meta, None)
        self.assertEqual(
            meta.llm_model_type.gemma3n.start_of_audio_token.token_str,
            "<|audio>",
        )
        self.assertEqual(
            meta.llm_model_type.gemma3n.end_of_audio_token.token_str,
            "<audio|>",
        )

    # The gemma3-family image-token population belongs to the subtypes that
    # actually have those proto fields; every other Gemma-family spec must
    # leave `LlmMetadata` untouched.

    def _fake_vision_meta(self, subtype_name=None):
        llm_model_type = types.SimpleNamespace()
        subtype = None
        if subtype_name is not None:
            subtype = types.SimpleNamespace(
                start_of_image_token=types.SimpleNamespace(token_str=""),
                end_of_image_token=types.SimpleNamespace(token_str=""),
                image_tensor_height=0,
                image_tensor_width=0,
            )
            setattr(llm_model_type, subtype_name, subtype)
        return types.SimpleNamespace(llm_model_type=llm_model_type), subtype

    def test_gemma3_family_populates_image_token_metadata(self):
        for spec_cls, subtype_name in (
            (Gemma3Spec, "gemma3"),
            (Gemma3nSpec, "gemma3n"),
        ):
            with self.subTest(subtype=subtype_name):
                meta, subtype = self._fake_vision_meta(subtype_name)
                spec_cls().populate_vision_metadata(meta, {"image_size": 48})
                self.assertEqual(
                    subtype.start_of_image_token.token_str, "<start_of_image>"
                )
                self.assertEqual(
                    subtype.end_of_image_token.token_str, "<end_of_image>"
                )
                self.assertEqual(subtype.image_tensor_height, 48)
                self.assertEqual(subtype.image_tensor_width, 48)

    def test_specs_without_image_proto_fields_populate_nothing(self):
        """These families map to a `LlmModelType` subtype with no image fields
        (`generic_model`, `function_gemma`), or have no dedicated vision
        subtype at all (PaliGemma), so populating gemma3-family image metadata
        would be an `AttributeError` on the real protobuf. Their
        `populate_vision_metadata` must be a strict no-op, touching nothing on
        `meta`."""
        for spec_cls in (
            LiteRTLMExportSpec,
            GemmaSpec,
            FunctionGemmaSpec,
            PaliGemmaSpec,
        ):
            with self.subTest(spec=spec_cls.__name__):
                meta, _ = self._fake_vision_meta()
                self.assertIsNone(
                    spec_cls().populate_vision_metadata(
                        meta, {"image_size": 48}
                    )
                )
                self.assertEqual(vars(meta.llm_model_type), {})

    def test_single_image_family_declares_flatten_image_batch(self):
        """PaliGemma's ViT is 4-D-only; its spec must declare
        flatten_image_batch=True so the adapter flattens the batched images
        stack before calling it."""
        self.assertTrue(PaliGemmaSpec().flatten_image_batch)

    def test_multi_image_families_do_not_flatten(self):
        """Every other vision family accepts the batched stack (or does not
        run the encoder standalone), so flatten_image_batch stays False."""
        self.assertFalse(LiteRTLMExportSpec().flatten_image_batch)
        self.assertFalse(Gemma3Spec().flatten_image_batch)
        self.assertFalse(Gemma3nSpec().flatten_image_batch)
        self.assertFalse(Gemma4Spec().flatten_image_batch)

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
        silently default to 1."""
        pre = types.SimpleNamespace()  # no max_images_per_prompt attribute
        with self.assertRaisesRegex(ValueError, "flatten_image_batch=False"):
            Gemma4Spec().get_max_images_per_prompt(pre)

    # `get_vision_config` / `get_audio_config` only read plain attributes, so
    # these guard tests drive them with small fakes instead of real backbones.

    def _fake_vision_model(self, **encoder_attrs):
        vision_encoder = types.SimpleNamespace(**encoder_attrs)
        backbone = types.SimpleNamespace(
            vision_encoder=vision_encoder,
            image_size=32,
            num_vision_tokens_per_image=4,
        )
        return types.SimpleNamespace(
            backbone=backbone,
            preprocessor=types.SimpleNamespace(max_images_per_prompt=2),
        )

    def _fake_audio_model(self, **preprocessor_attrs):
        return types.SimpleNamespace(
            backbone=types.SimpleNamespace(audio_encoder=object()),
            preprocessor=types.SimpleNamespace(**preprocessor_attrs),
        )

    def test_patch_values_family_requires_patch_size(self):
        """Gemma4 derives its exported `patch_values` signature and
        `max_num_patches` from `patch_size`, so a vision encoder that does not
        declare one is a misconfiguration -- it must raise rather than fall
        back to an invented default patch geometry."""
        model = self._fake_vision_model()  # no patch_size attribute
        with self.assertRaisesRegex(ValueError, "patch_size=None"):
            Gemma4Spec().get_vision_config(model)

    def test_non_patch_values_family_tolerates_missing_patch_size(self):
        """Only `patch_values` families consume `patch_size`. Gemma3n feeds
        `embedded_pixel_values` and no keras-hub Gemma3n preset can supply a
        `patch_size`, so a missing one must resolve to None, not raise."""
        model = self._fake_vision_model()  # no patch_size attribute
        vision_cfg = Gemma3nSpec().get_vision_config(model)
        self.assertIsNone(vision_cfg["patch_size"])
        self.assertEqual(vision_cfg["num_vision_tokens"], 8)

    def test_patch_size_is_read_from_the_vision_encoder(self):
        model = self._fake_vision_model(patch_size=8, pool_size=2)
        vision_cfg = Gemma4Spec().get_vision_config(model)
        self.assertEqual(vision_cfg["patch_size"], 8)
        self.assertEqual(vision_cfg["pool_size"], 2)

    def test_gemma4_vision_metadata_rejects_missing_patch_size(self):
        """`Gemma4Spec.populate_vision_metadata` computes `max_num_patches`
        from `patch_size`; a None must produce the same actionable error as
        `get_vision_config`, not a `TypeError` from the arithmetic."""
        meta = types.SimpleNamespace(llm_model_type=types.SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "patch_size=None"):
            Gemma4Spec().populate_vision_metadata(
                meta, {"image_size": 32, "patch_size": None}
            )

    def test_missing_max_clips_per_prompt_raises(self):
        model = self._fake_audio_model(
            num_audio_tokens_per_audio=3, audio_input_feat_size=16
        )
        with self.assertRaisesRegex(
            ValueError, "Could not determine `max_clips_per_prompt`"
        ):
            Gemma3nSpec().get_audio_config(model)

    def test_missing_num_audio_tokens_per_clip_raises(self):
        model = self._fake_audio_model(
            max_audios_per_prompt=2, audio_input_feat_size=16
        )
        with self.assertRaisesRegex(
            ValueError, "Could not determine `num_audio_tokens_per_clip`"
        ):
            Gemma3nSpec().get_audio_config(model)

    def test_missing_audio_input_feat_size_raises(self):
        model = self._fake_audio_model(
            max_audios_per_prompt=2, num_audio_tokens_per_audio=3
        )
        with self.assertRaisesRegex(
            ValueError, "Could not determine `audio_input_feat_size`"
        ):
            Gemma3nSpec().get_audio_config(model)

    def test_audio_config_resolves_from_preprocessor_attributes(self):
        """The happy path for the three guards above: with every attribute
        present the config resolves, so each guard test is failing on the one
        attribute it drops rather than on an incomplete fake."""
        model = self._fake_audio_model(
            max_audios_per_prompt=2,
            num_audio_tokens_per_audio=3,
            audio_input_feat_size=16,
        )
        audio_cfg = Gemma3nSpec().get_audio_config(model)
        self.assertEqual(audio_cfg["max_clips_per_prompt"], 2)
        self.assertEqual(audio_cfg["num_audio_tokens"], 6)
        self.assertEqual(audio_cfg["audio_input_feat_size"], 16)

    # These tests lock the family-wide default (False) so a per-family
    # relaxation is a deliberate, visible one-line override.

    def test_all_vision_families_disallow_bucketing_by_default(self):
        """Every vision-capable family inherits allows_vision_bucketing=False,
        so the family-wide bucketing ban stays in force until a family is
        explicitly, deliberately relaxed with a numerics-gated override."""
        self.assertFalse(Gemma3Spec().allows_vision_bucketing)
        self.assertFalse(Gemma3nSpec().allows_vision_bucketing)
        self.assertFalse(Gemma4Spec().allows_vision_bucketing)
        self.assertFalse(PaliGemmaSpec().allows_vision_bucketing)

    def test_base_and_text_specs_default_disallow_vision_bucketing(self):
        """The base spec and text-only families also default to False. The
        flag is only consulted when `get_vision_config` returns non-None (see
        `export.py`'s `has_vision` guard), so text-only families keep full
        bucketing support regardless of this value; the default is asserted
        here for completeness and to document the base-class contract."""
        self.assertFalse(LiteRTLMExportSpec().allows_vision_bucketing)
        self.assertFalse(GemmaSpec().allows_vision_bucketing)

    # These tests lock the {baked, separate} support matrix so a change is
    # deliberate.

    def test_vision_families_declare_supports_separate_vision(self):
        """Every vision-capable family declares supports_separate_vision
        explicitly, and its value matches the current support matrix: Gemma3,
        Gemma4 and PaliGemma support the separate path; Gemma3n (encoder
        inside the backbone) does not."""
        self.assertTrue(Gemma3Spec().supports_separate_vision)
        self.assertTrue(Gemma4Spec().supports_separate_vision)
        self.assertTrue(PaliGemmaSpec().supports_separate_vision)
        self.assertFalse(Gemma3nSpec().supports_separate_vision)

    def test_embedded_vision_family_disallows_separate_vision(self):
        """The `supports_separate_vision=False` families are exactly the
        `embedded_pixel_values` ones (encoder runs in-backbone). Lock the two
        facts together so the flag can't drift away from the input style it
        summarizes."""
        for spec in (Gemma3nSpec(),):
            self.assertEqual(spec.vision_input_style, "embedded_pixel_values")
            self.assertFalse(spec.supports_separate_vision)

    def test_base_and_text_specs_default_support_separate_vision(self):
        """The base spec and text-only families inherit the permissive
        default (True). The flag is only consulted when `get_vision_config`
        returns non-None (see export.py's `has_vision` guard), so its value on
        text-only families is inert; asserted here for the base-class
        contract, matching the allows_vision_bucketing default test above."""
        self.assertTrue(LiteRTLMExportSpec().supports_separate_vision)
        self.assertTrue(GemmaSpec().supports_separate_vision)
