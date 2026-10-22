import keras
import numpy as np
import pytest

from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu
from keras_hub.src.utils.transformers import convert_muse_glimmer_assistant


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
            total = keras_shape[0] * keras_shape[1] * keras_shape[2]
            hf_shape = (total // keras_shape[-1], keras_shape[-1])
        else:
            hf_shape = tuple(reversed(keras_shape))
        hf_tensor = self.rng.standard_normal(hf_shape).astype("float32")
        if hook_fn:
            hf_tensor = hook_fn(hf_tensor, list(keras_shape))
        keras_variable.assign(hf_tensor)
        self.ported_keys.append(hf_weight_key)


class TestMuseGlimmerAssistantConverter(TestCase):
    def _config(self):
        return {
            "hidden_size": 32,
            "head_dim": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 48,
            "rms_norm_eps": 1e-5,
            "sliding_window": 4,
            "rope_parameters": {"rope_theta": 500000.0},
            "layer_types": ["sliding_attention", "sliding_attention"],
            "target_layer_ids": [1, 3],
            "block_size": 16,
        }

    def test_convert_backbone_config(self):
        keras_config = convert_muse_glimmer_assistant.convert_backbone_config(
            self._config()
        )
        self.assertEqual(keras_config["hidden_dim"], 32)
        self.assertEqual(keras_config["head_dim"], 8)
        self.assertEqual(keras_config["num_layers"], 2)
        self.assertEqual(keras_config["rope_max_wavelength"], 500000.0)
        self.assertEqual(keras_config["context_projection_layer_ids"], [1, 3])
        self.assertTrue(keras_config["use_bidirectional_attention"])
        self.assertTrue(keras_config["use_external_embeddings"])
        self.assertFalse(keras_config["enable_qk_scale_and_gate"])
        self.assertTrue(keras_config["qk_norm_with_scale"])
        self.assertFalse(keras_config["use_sandwich_norm"])
        self.assertFalse(keras_config["use_centered_norm"])

    def test_convert_backbone_config_default_layer_types(self):
        config = self._config()
        del config["layer_types"]
        config["num_hidden_layers"] = 5
        keras_config = convert_muse_glimmer_assistant.convert_backbone_config(
            config
        )
        self.assertEqual(keras_config["layer_types"], ["sliding_attention"] * 5)

    def test_convert_backbone_config_rope_theta_fallback(self):
        config = self._config()
        del config["rope_parameters"]
        config["rope_theta"] = 250000.0
        keras_config = convert_muse_glimmer_assistant.convert_backbone_config(
            config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 250000.0)

    def test_convert_backbone_config_rejects_rope_type(self):
        for rope_type in ("linear", "yarn"):
            config = self._config()
            config["rope_parameters"]["rope_type"] = rope_type
            with self.assertRaisesRegex(ValueError, "rope_type"):
                convert_muse_glimmer_assistant.convert_backbone_config(config)

    def test_convert_backbone_config_rejects_hidden_act(self):
        config = self._config()
        config["hidden_act"] = "gelu"
        with self.assertRaisesRegex(ValueError, "hidden_act"):
            convert_muse_glimmer_assistant.convert_backbone_config(config)

    def test_convert_backbone_config_accepts_default_options(self):
        config = self._config()
        config["rope_parameters"]["rope_type"] = "default"
        config["hidden_act"] = "silu"
        convert_muse_glimmer_assistant.convert_backbone_config(config)

    def test_convert_task_config(self):
        kwargs = convert_muse_glimmer_assistant.convert_task_config(
            self._config()
        )
        self.assertEqual(kwargs["block_size"], 16)

    def test_convert_weights(self):
        transformers_config = self._config()
        keras_config = convert_muse_glimmer_assistant.convert_backbone_config(
            transformers_config
        )
        backbone = MuseGlimmerBackbone(**keras_config)
        loader = FakeLoader()
        convert_muse_glimmer_assistant.convert_weights(
            backbone, loader, transformers_config
        )
        self.assertIn("norm.weight", loader.ported_keys)
        self.assertIn("encoder.fc.weight", loader.ported_keys)
        self.assertIn("encoder.output_norm_enc.weight", loader.ported_keys)
        self.assertIn("layers.0.self_attn.q_proj.weight", loader.ported_keys)
        self.assertIn("layers.0.self_attn.q_norm.weight", loader.ported_keys)
        self.assertIn("layers.0.self_attn.k_norm.weight", loader.ported_keys)
        # HF's `post_attention_layernorm` maps to `_pre_feedforward_layernorm`.
        ported_scale = keras.ops.convert_to_numpy(
            backbone.transformer_layers[0]._pre_feedforward_layernorm.scale
        )
        self.assertFalse(np.allclose(ported_scale, np.zeros_like(ported_scale)))

    def test_convert_weights_matches_hf(self):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "MuseGlimmerAssistantModel"):
            self.skipTest("transformers has no MuseGlimmerAssistantModel.")

        class HFLoader:
            """Reads weights from the HF state dict."""

            def __init__(self, state_dict):
                self.state_dict = state_dict

            def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
                tensor = self.state_dict[hf_weight_key].numpy()
                if hook_fn:
                    tensor = hook_fn(tensor, list(keras_variable.shape))
                keras_variable.assign(tensor)

        batch_size, block_size, context_len = 2, 4, 3
        # A small window makes the sliding mask observable.
        hf_config = transformers.MuseGlimmerAssistantConfig(
            vocab_size=10,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=3,
            target_layer_ids=[1, 3],
            block_size=block_size,
            rope_parameters={"rope_theta": 777.0, "rope_type": "default"},
        )
        torch.manual_seed(0)
        hf_model = transformers.MuseGlimmerAssistantModel(hf_config).eval()
        # Random weights make every mapping observable. The default
        # initializer is so small that attention barely changes the output.
        with torch.no_grad():
            for param in hf_model.parameters():
                if param.ndim == 1:
                    param.copy_(1.0 + 0.3 * torch.randn_like(param))
                else:
                    param.copy_(
                        torch.randn_like(param) / param.shape[-1] ** 0.5
                    )

        config_dict = hf_config.to_dict()
        backbone = MuseGlimmerBackbone(
            **convert_muse_glimmer_assistant.convert_backbone_config(
                config_dict
            )
        )
        state_dict = {
            key: value.detach().cpu()
            for key, value in hf_model.state_dict().items()
        }
        convert_muse_glimmer_assistant.convert_weights(
            backbone, HFLoader(state_dict), config_dict
        )

        rng = np.random.default_rng(0)
        noise_embeds = rng.standard_normal((batch_size, block_size, 32)).astype(
            "float32"
        )
        context = rng.standard_normal((batch_size, context_len, 64)).astype(
            "float32"
        )
        with torch.no_grad():
            hf_output = hf_model(
                noise_embeds=torch.from_numpy(noise_embeds),
                context_hidden_states=torch.from_numpy(context),
                attention_mask=torch.ones(
                    batch_size, context_len + block_size, dtype=torch.long
                ),
            ).last_hidden_state.numpy()
        keras_output = backbone(
            {
                "noise_embeds": noise_embeds,
                "context_hidden_states": context,
                "padding_mask": np.ones((batch_size, block_size), "int32"),
            }
        )
        # TF32 matmuls on GPU leave up to 3e-3 against HF. CPU keeps 1e-4.
        tol = 1e-2 if running_on_gpu() else 1e-4
        self.assertAllClose(
            hf_output,
            keras.ops.convert_to_numpy(keras_output),
            atol=tol,
            rtol=1e-6,
        )
