import keras
import numpy as np

from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.tests.test_case import TestCase
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

    def test_convert_backbone_config_rope_theta_fallback(self):
        config = self._config()
        del config["rope_parameters"]
        config["rope_theta"] = 250000.0
        keras_config = convert_muse_glimmer_assistant.convert_backbone_config(
            config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 250000.0)

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
