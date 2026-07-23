import json
import os
import tempfile

import keras
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.models.mistral.mistral_causal_lm import MistralCausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_mistral
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = MistralCausalLM.from_preset("hf://cosmo3769/tiny-mistral-test")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://cosmo3769/tiny-mistral-test",
            load_weights=False,
        )
        self.assertIsInstance(model, MistralCausalLM)
        model = Backbone.from_preset(
            "hf://cosmo3769/tiny-mistral-test",
            load_weights=False,
        )
        self.assertIsInstance(model, MistralBackbone)

    def test_convert_backbone_config_rope_theta(self):
        # transformers < 5 format
        transformers_config = {
            "vocab_size": 100,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_key_value_heads": 2,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "sliding_window": 4096,
        }
        keras_config = convert_mistral.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)

        # transformers >= 5 format
        transformers_config = {
            "vocab_size": 100,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_key_value_heads": 2,
            "rope_parameters": {"rope_theta": 20000.0},
            "rms_norm_eps": 1e-5,
            "sliding_window": 4096,
        }
        # In the real transformers >= 5, rope_theta might still be present at
        # top level for some models, but the source of truth moved to
        # rope_parameters.
        keras_config = convert_mistral.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)

    def test_fast_safetensor_loading_matches_numpy(self):
        # The fast (backend-native) loading path must produce identical weights
        # to the default numpy path. Only the torch backend has a fast path.
        if keras.config.backend() != "torch":
            self.skipTest("Fast safetensor loading is only enabled on torch.")
        transformers = pytest.importorskip("transformers")
        torch = pytest.importorskip("torch")

        config = transformers.MistralConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            sliding_window=None,
            rope_theta=1e6,
            rms_norm_eps=1e-5,
        )
        torch.manual_seed(0)
        hf_model = (
            transformers.MistralForCausalLM(config).to(torch.bfloat16).eval()
        )

        with tempfile.TemporaryDirectory() as preset:
            hf_model.save_pretrained(preset)
            keras_config = convert_mistral.convert_backbone_config(
                json.load(open(os.path.join(preset, "config.json")))
            )
            # Default numpy path.
            backbone_np = MistralBackbone(**keras_config)
            with SafetensorLoader(preset, framework="np") as loader:
                convert_mistral.convert_weights(backbone_np, loader, None)
            # Fast torch path.
            backbone_pt = MistralBackbone(**keras_config)
            with SafetensorLoader(
                preset, framework="pt", device="cpu"
            ) as loader:
                convert_mistral.convert_weights(backbone_pt, loader, None)

        for w_np, w_pt in zip(backbone_np.weights, backbone_pt.weights):
            a = keras.ops.convert_to_numpy(w_np.value).astype("float32")
            b = keras.ops.convert_to_numpy(w_pt.value).astype("float32")
            self.assertAllClose(a, b, atol=1e-5)

    # TODO: compare numerics with huggingface model
