from unittest.mock import patch

import keras
import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_causal_lm import (
    MuseGlimmerCausalLM,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_muse_glimmer


class FakeLoader:
    """Minimal stand-in for `SafetensorLoader.port_weight`."""

    def __init__(self):
        self.ported_keys = []
        self.rng = np.random.default_rng(0)

    def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
        keras_shape = tuple(keras_variable.shape)
        if hook_fn is None:
            hf_shape = keras_shape
        elif len(keras_shape) == 3:
            # A 3D EinsumDense kernel is ported from a 2D HF `nn.Linear`
            # (out_features, in_features) weight — any 2D factorization of
            # the same total size round-trips correctly through the real
            # transpose+reshape hook.
            total = keras_shape[0] * keras_shape[1] * keras_shape[2]
            hf_shape = (total // keras_shape[-1], keras_shape[-1])
        else:
            hf_shape = tuple(reversed(keras_shape))
        hf_tensor = self.rng.standard_normal(hf_shape).astype("float32")
        if hook_fn:
            hf_tensor = hook_fn(hf_tensor, list(keras_shape))
        keras_variable.assign(hf_tensor)
        self.ported_keys.append(hf_weight_key)


class TestMuseGlimmerConverter(TestCase, parameterized.TestCase):
    def _text_config(self):
        return {
            "vocab_size": 100,
            "hidden_size": 32,
            "head_dim": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 48,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.196,
            "final_logit_softcapping": 20.0,
            "sliding_window": 4,
            "rope_theta": 500000.0,
        }

    def test_convert_tokenizer_loads_added_tokens(self):
        tokenizer_json = {
            "model": {
                "vocab": {"a": 0, "b": 1},
                "merges": [["a", "b"]],
            },
            "added_tokens": [
                {
                    "content": "<|begin_of_text|>",
                    "id": 200000,
                    "special": True,
                },
                {
                    "content": "<|finetune_right_pad|>",
                    "id": 200018,
                    "special": True,
                },
                {
                    "content": "<|reserved_special_token_2047|>",
                    "id": 202047,
                    "special": True,
                },
            ],
        }
        tokenizer_config = {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": "<|finetune_right_pad|>",
        }
        model_config = {
            "image_token_id": 200092,
            "video_token_id": 200091,
        }

        class FakeTokenizer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with patch.object(
            convert_muse_glimmer,
            "load_json",
            side_effect=[tokenizer_json, tokenizer_config, model_config],
        ):
            tokenizer = convert_muse_glimmer.convert_tokenizer(
                FakeTokenizer, "unused"
            )

        self.assertEqual(
            tokenizer.kwargs["vocabulary"]["<|finetune_right_pad|>"],
            200018,
        )
        self.assertEqual(
            tokenizer.kwargs["vocabulary"]["<|reserved_special_token_2047|>"],
            202047,
        )
        self.assertEqual(
            tokenizer.kwargs["pad_token"], "<|finetune_right_pad|>"
        )
        self.assertEqual(tokenizer.kwargs["merges"], ["a b"])
        self.assertEqual(len(tokenizer.kwargs["unsplittable_tokens"]), 3)

    def test_convert_backbone_config_text_only(self):
        transformers_config = {"text_config": self._text_config()}
        keras_config = convert_muse_glimmer.convert_backbone_config(
            transformers_config, MuseGlimmerBackbone
        )
        self.assertEqual(keras_config["vocabulary_size"], 100)
        self.assertEqual(keras_config["hidden_dim"], 32)
        self.assertEqual(keras_config["head_dim"], 8)
        self.assertEqual(keras_config["num_layers"], 4)
        self.assertEqual(keras_config["rope_max_wavelength"], 500000.0)
        self.assertEqual(keras_config["qk_scale_factor"], 3.87)
        self.assertNotIn("vision_encoder", keras_config)
        # Default layer_types: every 4th layer (from the end) is full.
        self.assertEqual(
            keras_config["layer_types"],
            [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

    def test_convert_backbone_config_rope_parameters_dict(self):
        text_config = self._text_config()
        del text_config["rope_theta"]
        text_config["rope_parameters"] = {"rope_theta": 250000.0}
        keras_config = convert_muse_glimmer.convert_backbone_config(
            {"text_config": text_config}, MuseGlimmerBackbone
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 250000.0)

    def test_load_image_converter_config(self):
        processor_config = {
            "image_processor": {
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "max_image_tokens": 4096,
                "merge_size": 2,
                "patch_size": 14,
                "rescale_factor": 1.0 / 255.0,
                "temporal_patch_size": 2,
            }
        }
        with patch.object(
            convert_muse_glimmer,
            "load_json",
            return_value=processor_config,
        ):
            config = convert_muse_glimmer.load_image_converter_config(
                "unused", {"vision_config": {}}
            )

        self.assertEqual(config["patch_size"], 14)
        self.assertEqual(config["patch_temporal"], 2)
        self.assertEqual(config["merge_size"], 2)
        self.assertEqual(config["max_image_tokens"], 4096)
        self.assertEqual(config["interpolation"], "lanczos3")
        self.assertTrue(config["antialias"])
        self.assertEqual(config["scale"], [2.0 / 255.0] * 3)
        self.assertEqual(config["offset"], [-1.0] * 3)

    def test_load_video_converter_config(self):
        processor_config = {
            "video_processor": {
                "fps": 2.0,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "max_video_frame_tokens": 144,
                "merge_size": 2,
                "num_frames": 96,
                "patch_size": 14,
                "rescale_factor": 1.0 / 255.0,
                "temporal_patch_size": 2,
            }
        }
        with patch.object(
            convert_muse_glimmer,
            "load_json",
            return_value=processor_config,
        ):
            config = convert_muse_glimmer.load_video_converter_config(
                "unused", {"vision_config": {}}
            )

        self.assertEqual(config["patch_size"], 14)
        self.assertEqual(config["patch_temporal"], 2)
        self.assertEqual(config["merge_size"], 2)
        self.assertEqual(config["fps"], 2.0)
        self.assertEqual(config["num_frames"], 96)
        self.assertEqual(config["max_video_frame_tokens"], 144)
        self.assertEqual(config["interpolation"], "lanczos3")
        self.assertTrue(config["antialias"])
        self.assertEqual(config["scale"], [2.0 / 255.0] * 3)
        self.assertEqual(config["offset"], [-1.0] * 3)

    def test_convert_backbone_config_with_vision(self):
        transformers_config = {
            "text_config": self._text_config(),
            "vision_config": {
                "hidden_size": 16,
                "num_attention_heads": 2,
                "intermediate_size": 32,
                "num_hidden_layers": 4,
                "patch_size": 4,
                "patch_temporal": 2,
                "merge_size": 2,
                "pos_emb_height": 8,
                "pos_emb_width": 8,
                "layer_norm_eps": 1e-5,
                "rope_parameters": {"rope_theta": 10000.0},
            },
            "projector_hidden_size": 24,
            "projector_hidden_act": "gelu",
        }
        keras_config = convert_muse_glimmer.convert_backbone_config(
            transformers_config, MuseGlimmerBackbone
        )
        self.assertIn("vision_encoder", keras_config)
        self.assertEqual(keras_config["projector_hidden_dim"], 24)
        self.assertEqual(
            keras_config["vision_encoder"].out_hidden_size, 16 * 2**2
        )

    def test_convert_weights_text_only(self):
        transformers_config = {"text_config": self._text_config()}
        keras_config = convert_muse_glimmer.convert_backbone_config(
            transformers_config, MuseGlimmerBackbone
        )
        backbone = MuseGlimmerBackbone(**keras_config)
        loader = FakeLoader()
        convert_muse_glimmer.convert_weights(
            backbone, loader, transformers_config
        )
        # Every port_weight call must have resolved without raising, and
        # the layer-0 input_layernorm weight should reflect the ported
        # (non-default) value rather than its zero initialization.
        self.assertGreater(len(loader.ported_keys), 0)
        ported_scale = keras.ops.convert_to_numpy(
            backbone.transformer_layers[0]._input_layernorm.scale
        )
        self.assertFalse(np.allclose(ported_scale, np.zeros_like(ported_scale)))

    def test_convert_weights_tied_embeddings(self):
        transformers_config = {
            "text_config": self._text_config(),
            "tie_word_embeddings": True,
        }
        keras_config = convert_muse_glimmer.convert_backbone_config(
            transformers_config, MuseGlimmerBackbone
        )
        backbone = MuseGlimmerBackbone(**keras_config)
        loader = FakeLoader()
        convert_muse_glimmer.convert_weights(
            backbone, loader, transformers_config
        )
        self.assertNotIn("lm_head.weight", loader.ported_keys)
        embedding_key = "model.language_model.embed_tokens.weight"
        self.assertEqual(loader.ported_keys.count(embedding_key), 2)

    def test_convert_weights_untied_embeddings_read_lm_head(self):
        transformers_config = {"text_config": self._text_config()}
        keras_config = convert_muse_glimmer.convert_backbone_config(
            transformers_config, MuseGlimmerBackbone
        )
        backbone = MuseGlimmerBackbone(**keras_config)
        loader = FakeLoader()
        convert_muse_glimmer.convert_weights(
            backbone, loader, transformers_config
        )
        self.assertIn("lm_head.weight", loader.ported_keys)

    def _vision_config(self):
        return {
            "hidden_size": 16,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "patch_size": 4,
            "patch_temporal": 2,
            "merge_size": 2,
            "pos_emb_height": 8,
            "pos_emb_width": 8,
            "layer_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 10000.0},
        }

    def test_unsupported_options_raise(self):
        text_overrides = [
            (
                "rope_type",
                {
                    "rope_parameters": {
                        "rope_type": "linear",
                        "rope_theta": 500000.0,
                        "factor": 2.0,
                    }
                },
            ),
            ("attention_bias", {"attention_bias": True}),
            ("hidden_activation", {"hidden_activation": "gelu"}),
            (
                "layer_rope_theta",
                {"layer_rope_theta": [500000.0, 500000.0, 500000.0, 500000.0]},
            ),
        ]
        for option, override in text_overrides:
            text_config = {**self._text_config(), **override}
            with self.assertRaisesRegex(ValueError, option):
                convert_muse_glimmer.convert_backbone_config(
                    {"text_config": text_config}, MuseGlimmerBackbone
                )
        with self.assertRaisesRegex(ValueError, "hidden_act"):
            convert_muse_glimmer.convert_backbone_config(
                {
                    "text_config": self._text_config(),
                    "vision_config": {
                        **self._vision_config(),
                        "hidden_act": "gelu_pytorch_tanh",
                    },
                    "projector_hidden_size": 24,
                },
                MuseGlimmerBackbone,
            )

    def test_layer_rope_theta_values_only_switch_rope(self):
        # Hugging Face uses one rotary embedding for all layers. A nonzero
        # `layer_rope_theta` entry only turns RoPE on.
        text_config = {
            **self._text_config(),
            "layer_rope_theta": [1.0, 2.0, 3.0, 0],
        }
        keras_config = convert_muse_glimmer.convert_backbone_config(
            {"text_config": text_config}, MuseGlimmerBackbone
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 500000.0)

    def test_default_layer_rope_theta_must_match_layer_types(self):
        # The Hugging Face default NoPE layers are every 4th layer from
        # the end. They do not follow custom `layer_types`.
        text_config = {
            **self._text_config(),
            "layer_types": ["full_attention"] * 4,
        }
        with self.assertRaisesRegex(ValueError, "layer_rope_theta"):
            convert_muse_glimmer.convert_backbone_config(
                {"text_config": text_config}, MuseGlimmerBackbone
            )

    def test_vision_layer_norm_eps_is_not_an_error(self):
        # The block norms use a fixed 1e-5, like Hugging Face. The config
        # value applies to `ln_pre` and `ln_post`.
        vision_config = {**self._vision_config(), "layer_norm_eps": 1e-2}
        keras_config = convert_muse_glimmer.convert_backbone_config(
            {
                "text_config": self._text_config(),
                "vision_config": vision_config,
                "projector_hidden_size": 24,
            },
            MuseGlimmerBackbone,
        )
        encoder = keras_config["vision_encoder"]
        self.assertEqual(encoder.ln_pre.epsilon, 1e-2)
        self.assertEqual(encoder.blocks[0].layer_norm_eps, 1e-5)

    @parameterized.named_parameters(
        ("baseline", {}, {}, {}),
        # `layer_norm_eps` applies to `ln_pre` and `ln_post` only. The block
        # norms use a fixed 1e-5.
        ("vision_layer_norm_eps", {}, {"layer_norm_eps": 1e-2}, {}),
        # Every config value differs from the `MuseGlimmerBackbone` default,
        # so a dropped or misrouted config mapping changes the logits.
        (
            "non_default_config",
            {
                "rms_norm_eps": 1e-2,
                "post_norm_eps": 1e-3,
                "qk_scale_factor": 1.7,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 777.0,
                },
            },
            {
                "rope_parameters": {"rope_type": "axial", "rope_theta": 123.0},
                "layer_types": ["full_attention", "window_attention"],
                # A 4x4 grid on a 2x2 table: the position embedding is
                # interpolated and the window layer has four windows.
                "pos_emb_height": 2,
                "pos_emb_width": 2,
            },
            {"projector_hidden_act": "silu"},
        ),
    )
    def test_logits_match_hf(
        self, text_overrides, vision_overrides, top_overrides
    ):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "MuseGlimmerForConditionalGeneration"):
            self.skipTest("transformers has no MuseGlimmer model.")

        class HFLoader:
            """Reads weights from the HF state dict."""

            def __init__(self, state_dict):
                self.state_dict = state_dict

            def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
                tensor = self.state_dict[hf_weight_key].numpy()
                if hook_fn:
                    tensor = hook_fn(tensor, list(keras_variable.shape))
                keras_variable.assign(tensor)

        text_config = {
            "vocab_size": 50,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "sliding_window": 3,
            "final_logit_softcapping": 1.0,
            "output_multiplier": 0.5,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1e4},
            **text_overrides,
        }
        vision_config = {
            "hidden_size": 16,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "patch_size": 2,
            "patch_temporal": 2,
            "merge_size": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
            **vision_overrides,
        }
        hf_config = transformers.MuseGlimmerConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=40,
            out_hidden_size=64,
            projector_hidden_size=16,
            **top_overrides,
        )
        torch.manual_seed(0)
        hf_model = transformers.MuseGlimmerForConditionalGeneration(
            hf_config
        ).eval()
        # Random weights make every mapping observable. The default
        # initializer is so small that most layers barely change the output.
        with torch.no_grad():
            for param in hf_model.parameters():
                if param.ndim == 1:
                    param.copy_(1.0 + 0.3 * torch.randn_like(param))
                else:
                    param.copy_(
                        torch.randn_like(param) / param.shape[-1] ** 0.5
                    )

        config_dict = hf_config.to_dict()
        keras_config = convert_muse_glimmer.convert_backbone_config(
            config_dict, MuseGlimmerBackbone
        )
        backbone = MuseGlimmerBackbone(**keras_config)
        state_dict = {
            key: value.detach().cpu()
            for key, value in hf_model.state_dict().items()
        }
        convert_muse_glimmer.convert_weights(
            backbone, HFLoader(state_dict), config_dict
        )
        causal_lm = MuseGlimmerCausalLM(backbone=backbone, preprocessor=None)

        rng = np.random.default_rng(0)
        # One image of 16 patches gives 4 merged vision tokens.
        token_ids = rng.integers(0, 30, size=(1, 10)).astype("int32")
        token_ids[0, 2:6] = 40
        patch_dim = 2 * 3 * 2**2
        pixel_values = rng.standard_normal((16, patch_dim)).astype("float32")
        grid_thw = np.array([[1, 4, 4]], dtype="int32")
        with torch.no_grad():
            hf_logits = hf_model(
                input_ids=torch.from_numpy(token_ids).long(),
                attention_mask=torch.ones(1, 10, dtype=torch.long),
                pixel_values=torch.from_numpy(pixel_values),
                image_grid_thw=torch.from_numpy(grid_thw).long(),
            ).logits.numpy()
        keras_logits = causal_lm(
            {
                "token_ids": token_ids,
                "padding_mask": np.ones((1, 10), "int32"),
                "pixel_values": pixel_values[None],
                "image_grid_thw": grid_thw[None],
                "vision_indices": np.array([[2, 3, 4, 5]], dtype="int32"),
            }
        )
        self.assertAllClose(
            keras.ops.convert_to_numpy(keras_logits),
            hf_logits,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_dtype_reaches_vision_encoder(self):
        keras_config = convert_muse_glimmer.convert_backbone_config(
            {
                "text_config": self._text_config(),
                "vision_config": self._vision_config(),
                "projector_hidden_size": 24,
            },
            MuseGlimmerBackbone,
            dtype="bfloat16",
        )
        self.assertEqual(keras_config["dtype"], "bfloat16")
        self.assertEqual(
            keras_config["vision_encoder"].dtype_policy.name, "bfloat16"
        )

    def test_convert_weights_with_vision(self):
        transformers_config = {
            "text_config": self._text_config(),
            "vision_config": {
                "hidden_size": 16,
                "num_attention_heads": 2,
                "intermediate_size": 32,
                "num_hidden_layers": 4,
                "patch_size": 4,
                "patch_temporal": 2,
                "merge_size": 2,
                "pos_emb_height": 8,
                "pos_emb_width": 8,
                "layer_norm_eps": 1e-5,
                "rope_parameters": {"rope_theta": 10000.0},
            },
            "projector_hidden_size": 24,
            "projector_hidden_act": "gelu",
        }
        keras_config = convert_muse_glimmer.convert_backbone_config(
            transformers_config, MuseGlimmerBackbone
        )
        backbone = MuseGlimmerBackbone(**keras_config)
        loader = FakeLoader()
        convert_muse_glimmer.convert_weights(
            backbone, loader, transformers_config
        )
        self.assertIn(
            "model.vision_tower.patch_embedder.patch_embedding.weight",
            loader.ported_keys,
        )
        self.assertIn("model.vision_adapter.fc1.weight", loader.ported_keys)
        self.assertIn("model.vision_projection.weight", loader.ported_keys)
