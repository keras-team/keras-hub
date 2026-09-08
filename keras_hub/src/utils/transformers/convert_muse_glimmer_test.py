from unittest.mock import patch

import keras
import numpy as np

from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_muse_glimmer


class FakeLoader:
    """Minimal stand-in for `SafetensorLoader.port_weight`.

    Generates a random tensor shaped like the real on-disk HF weight (a 2D
    `nn.Linear`-style `(out, in)` shape when `hook_fn` is given, since real
    HF checkpoints never store 3D kernels directly) and lets `hook_fn`
    reshape it into KerasHub's convention, then assigns it and records
    which HF keys were requested so tests can assert every expected weight
    was actually ported.
    """

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
            # `_multi_head_transpose` hook's transpose+reshape.
            total = keras_shape[0] * keras_shape[1] * keras_shape[2]
            hf_shape = (total // keras_shape[-1], keras_shape[-1])
        else:
            hf_shape = tuple(reversed(keras_shape))
        hf_tensor = self.rng.standard_normal(hf_shape).astype("float32")
        if hook_fn:
            hf_tensor = hook_fn(hf_tensor, list(keras_shape))
        keras_variable.assign(hf_tensor)
        self.ported_keys.append(hf_weight_key)


class TestMuseGlimmerConverter(TestCase):
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
            transformers_config
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
            {"text_config": text_config}
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 250000.0)

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
            transformers_config
        )
        self.assertIn("vision_encoder", keras_config)
        self.assertEqual(keras_config["projector_hidden_dim"], 24)
        self.assertEqual(
            keras_config["vision_encoder"].out_hidden_size, 16 * 2**2
        )

    def test_convert_weights_text_only(self):
        transformers_config = {"text_config": self._text_config()}
        keras_config = convert_muse_glimmer.convert_backbone_config(
            transformers_config
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
            transformers_config
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
