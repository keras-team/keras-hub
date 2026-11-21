"""Unit tests for the Gemma4 KerasHub → HuggingFace export utilities.

Tests cover:
  * Text-only export  (``vision_encoder=None``, ``audio_encoder=None``)
  * Multimodal export (``vision_encoder`` provided)
  * Config fields round-trip
  * Weight-map key presence and shape consistency
  * Tokenizer config structure
  * End-to-end logit parity via ``export_to_transformers``
"""

import os

import numpy as np
import torch
from keras import ops
from transformers import AutoModelForCausalLM

from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_causal_lm import Gemma4CausalLM
from keras_hub.src.models.gemma4.gemma4_causal_lm_preprocessor import (
    Gemma4CausalLMPreprocessor,
)
from keras_hub.src.models.gemma4.gemma4_tokenizer import Gemma4Tokenizer
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.export.gemma4 import get_gemma4_config
from keras_hub.src.utils.transformers.export.gemma4 import (
    get_gemma4_tokenizer_config,
)
from keras_hub.src.utils.transformers.export.gemma4 import (
    get_gemma4_weights_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_only_backbone(vocab_size=256, num_layers=2):
    """Create a tiny text-only Gemma4Backbone for testing."""
    return Gemma4Backbone(
        vocabulary_size=vocab_size,
        image_size=None,
        num_layers=num_layers,
        num_query_heads=2,
        num_key_value_heads=1,
        hidden_dim=64,
        intermediate_dim=128,
        head_dim=32,
        use_sliding_window_attention=True,
        sliding_window_size=16,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        vision_encoder=None,
        audio_encoder=None,
        layer_norm_epsilon=1e-6,
        dropout=0,
    )


def _make_vision_encoder(image_size=16):
    """Create a tiny Gemma4VisionEncoder for testing."""
    return Gemma4VisionEncoder(
        image_size=image_size,
        patch_size=4,
        pool_size=2,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        num_key_value_heads=2,
        hidden_dim=8,
        intermediate_dim=16,
        output_dim=64,  # must match text backbone hidden_dim
    )


def _make_multimodal_backbone(vocab_size=256, image_size=16, num_layers=2):
    """Create a tiny multimodal Gemma4Backbone (text + vision) for testing."""
    vision_encoder = _make_vision_encoder(image_size)
    return Gemma4Backbone(
        vocabulary_size=vocab_size,
        image_size=image_size,
        num_layers=num_layers,
        num_query_heads=2,
        num_key_value_heads=1,
        hidden_dim=64,
        intermediate_dim=128,
        head_dim=32,
        use_sliding_window_attention=True,
        sliding_window_size=16,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        vision_encoder=vision_encoder,
        audio_encoder=None,
        layer_norm_epsilon=1e-6,
        dropout=0,
    )


def _randomize_weights(model, seed=42):
    """Fill model weights with deterministic random values."""
    rng = np.random.default_rng(seed)
    weights = model.get_weights()
    for i in range(len(weights)):
        weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
    model.set_weights(weights)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestGemma4Config(TestCase):
    def test_text_only_config_fields(self):
        backbone = _make_text_only_backbone()
        cfg = get_gemma4_config(backbone, include_lm_head=False)

        self.assertEqual(cfg["model_type"], "gemma4_text")
        self.assertEqual(cfg["vocab_size"], backbone.vocabulary_size)
        self.assertEqual(cfg["num_hidden_layers"], backbone.num_layers)
        self.assertEqual(cfg["num_attention_heads"], backbone.num_query_heads)
        self.assertEqual(
            cfg["num_key_value_heads"], backbone.num_key_value_heads
        )
        self.assertEqual(cfg["hidden_size"], backbone.hidden_dim)
        self.assertEqual(cfg["intermediate_size"], backbone.intermediate_dim)
        self.assertEqual(cfg["head_dim"], backbone.head_dim)
        self.assertEqual(cfg["rms_norm_eps"], backbone.layer_norm_epsilon)
        self.assertIn("layer_types", cfg)
        self.assertIn("rope_parameters", cfg)

    def test_text_only_backbone_architecture(self):
        backbone = _make_text_only_backbone()
        cfg = get_gemma4_config(backbone, include_lm_head=False)
        self.assertIn("Gemma4TextModel", cfg["architectures"])

    def test_text_only_causallm_architecture(self):
        backbone = _make_text_only_backbone()
        cfg = get_gemma4_config(backbone, include_lm_head=True)
        self.assertIn("Gemma4TextForCausalLM", cfg["architectures"])

    def test_multimodal_config_structure(self):
        backbone = _make_multimodal_backbone()
        cfg = get_gemma4_config(backbone, include_lm_head=True)

        self.assertEqual(cfg["model_type"], "gemma4")
        self.assertIn("text_config", cfg)
        self.assertIn("vision_config", cfg)
        self.assertNotIn("audio_config", cfg)  # no audio encoder
        self.assertIn("Gemma4ForConditionalGeneration", cfg["architectures"])

    def test_vision_config_fields(self):
        backbone = _make_multimodal_backbone()
        cfg = get_gemma4_config(backbone)
        vis_cfg = cfg["vision_config"]
        ve = backbone.vision_encoder

        self.assertEqual(vis_cfg["num_hidden_layers"], ve.num_layers)
        self.assertEqual(vis_cfg["num_attention_heads"], ve.num_heads)
        self.assertEqual(vis_cfg["hidden_size"], ve.hidden_dim)
        self.assertEqual(vis_cfg["intermediate_size"], ve.intermediate_dim)
        self.assertEqual(vis_cfg["head_dim"], ve.head_dim)
        self.assertEqual(vis_cfg["patch_size"], ve.patch_size)
        self.assertEqual(vis_cfg["pooling_kernel_size"], ve.pool_size)

    def test_rope_parameters_present(self):
        backbone = _make_text_only_backbone()
        cfg = get_gemma4_config(backbone)
        rope = cfg["rope_parameters"]
        self.assertIn("full_attention", rope)
        self.assertIn("sliding_attention", rope)
        self.assertIn("rope_theta", rope["full_attention"])
        self.assertIn("partial_rotary_factor", rope["full_attention"])


# ---------------------------------------------------------------------------
# Weights map tests
# ---------------------------------------------------------------------------


class TestGemma4WeightsMap(TestCase):
    def test_text_only_key_presence(self):
        backbone = _make_text_only_backbone(num_layers=2)
        # Force build.
        backbone(
            {
                "token_ids": np.ones((1, 4), dtype="int32"),
                "padding_mask": np.ones((1, 4), dtype="int32"),
                "position_ids": np.arange(4, dtype="int32")[None],
            }
        )
        weights = get_gemma4_weights_map(backbone, include_lm_head=False)

        self.assertIn("model.embed_tokens.weight", weights)
        self.assertIn("model.norm.weight", weights)

        for i in range(backbone.num_layers):
            lp = f"model.layers.{i}"
            for key in [
                f"{lp}.input_layernorm.weight",
                f"{lp}.post_attention_layernorm.weight",
                f"{lp}.pre_feedforward_layernorm.weight",
                f"{lp}.post_feedforward_layernorm.weight",
                f"{lp}.self_attn.q_proj.weight",
                f"{lp}.self_attn.q_norm.weight",
                f"{lp}.self_attn.o_proj.weight",
                f"{lp}.mlp.gate_proj.weight",
                f"{lp}.mlp.up_proj.weight",
                f"{lp}.mlp.down_proj.weight",
                f"{lp}.layer_scalar",
            ]:
                self.assertIn(key, weights, f"Missing key: {key}")

    def test_text_only_weight_shapes(self):
        backbone = _make_text_only_backbone(num_layers=2)
        backbone(
            {
                "token_ids": np.ones((1, 4), dtype="int32"),
                "padding_mask": np.ones((1, 4), dtype="int32"),
                "position_ids": np.arange(4, dtype="int32")[None],
            }
        )
        weights = get_gemma4_weights_map(backbone, include_lm_head=False)

        vocab = backbone.vocabulary_size
        d = backbone.hidden_dim
        n = backbone.num_query_heads
        k = backbone.num_key_value_heads
        h = backbone.head_dim
        f = backbone.intermediate_dim

        # Embedding.
        self.assertEqual(weights["model.embed_tokens.weight"].shape, (vocab, d))

        lp = "model.layers.0"
        # Q: (n*h, d)
        self.assertEqual(
            weights[f"{lp}.self_attn.q_proj.weight"].shape, (n * h, d)
        )
        # K: (k*h, d)
        self.assertEqual(
            weights[f"{lp}.self_attn.k_proj.weight"].shape, (k * h, d)
        )
        # V: (k*h, d)
        self.assertEqual(
            weights[f"{lp}.self_attn.v_proj.weight"].shape, (k * h, d)
        )
        # O: (d, n*h)
        self.assertEqual(
            weights[f"{lp}.self_attn.o_proj.weight"].shape, (d, n * h)
        )
        # Gate / up: (f, d)
        self.assertEqual(weights[f"{lp}.mlp.gate_proj.weight"].shape, (f, d))
        self.assertEqual(weights[f"{lp}.mlp.up_proj.weight"].shape, (f, d))
        # Down: (d, f)
        self.assertEqual(weights[f"{lp}.mlp.down_proj.weight"].shape, (d, f))
        # Layer scalar: (1,)
        self.assertEqual(weights[f"{lp}.layer_scalar"].shape, (1,))

    def test_multimodal_key_presence(self):
        backbone = _make_multimodal_backbone(num_layers=2)
        # Build with dummy multimodal input.
        num_patches = (16 // 4) ** 2  # 16
        patch_dim = 3 * 4 * 4  # 48
        dummy_input = {
            "token_ids": np.ones((1, 8), dtype="int32"),
            "padding_mask": np.ones((1, 8), dtype="int32"),
            "position_ids": np.arange(8, dtype="int32")[None],
            "pixel_values": np.ones(
                (1, 1, num_patches, patch_dim), dtype="float32"
            ),
            "pixel_position_ids": np.zeros(
                (1, 1, num_patches, 2), dtype="int32"
            ),
            "vision_indices": np.zeros((1, 8), dtype="int32"),
            "vision_mask": np.zeros((1, 8), dtype="int32"),
        }
        backbone(dummy_input)
        weights = get_gemma4_weights_map(backbone, include_lm_head=False)

        # Text prefix is "model.language_model.*" for multimodal models.
        self.assertIn("model.language_model.embed_tokens.weight", weights)
        self.assertIn("model.language_model.norm.weight", weights)

        # Vision tower keys.
        self.assertIn(
            "model.vision_tower.patch_embedder.input_proj.weight", weights
        )
        self.assertIn(
            "model.vision_tower.patch_embedder.position_embedding_table",
            weights,
        )
        self.assertIn("model.embed_vision.embedding_projection.weight", weights)

        # Vision encoder transformer block keys.
        ve = backbone.vision_encoder
        image_encoder = ve.get_layer("image_encoder")
        for i in range(len(image_encoder.encoder_blocks)):
            vp = f"model.vision_tower.encoder.layers.{i}"
            self.assertIn(f"{vp}.input_layernorm.weight", weights)
            self.assertIn(f"{vp}.self_attn.q_proj.linear.weight", weights)


# ---------------------------------------------------------------------------
# Full export round-trip tests
# ---------------------------------------------------------------------------


class TestGemma4Export(TestCase):
    def test_text_only_export_to_hf(self):
        """Export a text-only Gemma4 backbone and verify HF model loads."""
        proto = os.path.join(self.get_test_data_dir(), "gemma4_test_vocab.spm")
        tokenizer = Gemma4Tokenizer(
            proto=proto,
            has_vision_tokens=False,
            has_audio_tokens=False,
            has_video_tokens=False,
        )

        backbone = Gemma4Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=None,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=64,
            intermediate_dim=128,
            head_dim=32,
            use_sliding_window_attention=False,
            sliding_window_size=512,
            attention_logit_soft_cap=None,
            final_logit_soft_cap=None,
            vision_encoder=None,
            audio_encoder=None,
            layer_norm_epsilon=1e-6,
            dropout=0,
        )

        preprocessor = Gemma4CausalLMPreprocessor(tokenizer=tokenizer)
        keras_model = Gemma4CausalLM(
            backbone=backbone, preprocessor=preprocessor
        )
        _randomize_weights(keras_model)

        export_path = os.path.join(self.get_temp_dir(), "export_text_only")
        keras_model.export_to_transformers(export_path)

        # Verify files exist.
        for fname in ["config.json", "model.safetensors"]:
            self.assertTrue(
                os.path.exists(os.path.join(export_path, fname)),
                f"Missing exported file: {fname}",
            )

        # Load with HF and verify config fields.
        hf_model = AutoModelForCausalLM.from_pretrained(export_path)
        hf_cfg = hf_model.config

        self.assertEqual(hf_cfg.vocab_size, backbone.vocabulary_size)
        self.assertEqual(hf_cfg.num_hidden_layers, backbone.num_layers)
        self.assertEqual(hf_cfg.num_attention_heads, backbone.num_query_heads)
        self.assertEqual(
            hf_cfg.num_key_value_heads, backbone.num_key_value_heads
        )
        self.assertEqual(hf_cfg.hidden_size, backbone.hidden_dim)
        self.assertEqual(hf_cfg.intermediate_size, backbone.intermediate_dim)
        self.assertEqual(hf_cfg.head_dim, backbone.head_dim)

    def test_multimodal_export_to_hf(self):
        """Export a multimodal Gemma4 backbone and verify HF model loads."""
        proto = os.path.join(self.get_test_data_dir(), "gemma4_test_vocab.spm")
        tokenizer = Gemma4Tokenizer(proto=proto)

        vision_encoder = Gemma4VisionEncoder(
            image_size=16,
            patch_size=4,
            pool_size=2,
            num_layers=2,
            num_heads=2,
            head_dim=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=64,
        )
        backbone = Gemma4Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=16,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=64,
            intermediate_dim=128,
            head_dim=32,
            use_sliding_window_attention=True,
            sliding_window_size=16,
            attention_logit_soft_cap=None,
            final_logit_soft_cap=None,
            vision_encoder=vision_encoder,
            audio_encoder=None,
            layer_norm_epsilon=1e-6,
            dropout=0,
        )

        preprocessor = Gemma4CausalLMPreprocessor(tokenizer=tokenizer)
        keras_model = Gemma4CausalLM(
            backbone=backbone, preprocessor=preprocessor
        )
        _randomize_weights(keras_model)

        export_path = os.path.join(self.get_temp_dir(), "export_multimodal")
        keras_model.export_to_transformers(export_path)

        # Verify files exist.
        for fname in ["config.json", "model.safetensors"]:
            self.assertTrue(
                os.path.exists(os.path.join(export_path, fname)),
                f"Missing exported file: {fname}",
            )

        # Load and verify config.
        hf_model = AutoModelForCausalLM.from_pretrained(export_path)
        hf_cfg = hf_model.config
        text_cfg = hf_cfg.text_config

        self.assertEqual(text_cfg.vocab_size, backbone.vocabulary_size)
        self.assertEqual(text_cfg.num_hidden_layers, backbone.num_layers)
        self.assertEqual(text_cfg.num_attention_heads, backbone.num_query_heads)
        self.assertEqual(text_cfg.hidden_size, backbone.hidden_dim)

        # Verify vision config is present.
        self.assertTrue(hasattr(hf_cfg, "vision_config"))
        vis_cfg = hf_cfg.vision_config
        self.assertEqual(vis_cfg.patch_size, vision_encoder.patch_size)

    def test_logit_parity_text_only(self):
        """Exported text-only model should produce identical logits."""
        proto = os.path.join(self.get_test_data_dir(), "gemma4_test_vocab.spm")
        tokenizer = Gemma4Tokenizer(
            proto=proto,
            has_vision_tokens=False,
            has_audio_tokens=False,
            has_video_tokens=False,
        )

        backbone = Gemma4Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=None,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=64,
            intermediate_dim=128,
            head_dim=32,
            use_sliding_window_attention=False,
            sliding_window_size=512,
            attention_logit_soft_cap=None,
            final_logit_soft_cap=None,
            vision_encoder=None,
            audio_encoder=None,
            layer_norm_epsilon=1e-6,
            dropout=0,
        )
        preprocessor = Gemma4CausalLMPreprocessor(tokenizer=tokenizer)
        keras_model = Gemma4CausalLM(
            backbone=backbone, preprocessor=preprocessor
        )
        _randomize_weights(keras_model)

        export_path = os.path.join(self.get_temp_dir(), "export_logit_parity")
        keras_model.export_to_transformers(export_path)

        hf_model = AutoModelForCausalLM.from_pretrained(
            export_path, torch_dtype=torch.float32
        )
        hf_model.eval()

        # Prepare a short token sequence.
        prompt = "the quick"
        keras_inputs = preprocessor.generate_preprocess(prompt)["token_ids"]
        keras_inputs = keras_inputs[keras_inputs != 0]
        input_ids = torch.tensor(
            [ops.convert_to_numpy(keras_inputs)], dtype=torch.long
        )

        # Keras forward pass.
        keras_input_dict = {
            "token_ids": ops.convert_to_numpy(keras_inputs)[None],
            "padding_mask": np.ones((1, len(keras_inputs)), dtype="int32"),
            "position_ids": np.arange(len(keras_inputs), dtype="int32")[None],
        }
        hidden = keras_model.backbone(keras_input_dict)
        keras_logits = np.array(
            ops.convert_to_numpy(
                keras_model.backbone.token_embedding(hidden, reverse=True)
            )
        )

        # HF forward pass.
        with torch.no_grad():
            hf_out = hf_model(input_ids=input_ids)
        hf_logits = hf_out.logits.float().numpy()

        self.assertAllClose(keras_logits, hf_logits, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Tokenizer config tests
# ---------------------------------------------------------------------------


class TestGemma4TokenizerConfig(TestCase):
    def test_tokenizer_config_fields(self):
        proto = os.path.join(self.get_test_data_dir(), "gemma4_test_vocab.spm")
        tokenizer = Gemma4Tokenizer(proto=proto)
        cfg = get_gemma4_tokenizer_config(tokenizer)

        self.assertEqual(cfg["tokenizer_class"], "GemmaTokenizer")
        self.assertIn("bos_token", cfg)
        self.assertIn("eos_token", cfg)
        self.assertIn("pad_token", cfg)
        self.assertIn("added_tokens_decoder", cfg)

    def test_tokenizer_config_export(self):
        proto = os.path.join(self.get_test_data_dir(), "gemma4_test_vocab.spm")
        tokenizer = Gemma4Tokenizer(proto=proto)
        export_path = os.path.join(self.get_temp_dir(), "export_tokenizer")
        tokenizer.export_to_transformers(export_path)

        self.assertTrue(
            os.path.exists(os.path.join(export_path, "tokenizer_config.json"))
        )
