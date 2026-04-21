import tempfile

import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.models.mistral.mistral_causal_lm import MistralCausalLM
from keras_hub.src.tests.test_case import TestCase


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

    @pytest.mark.large
    def test_explicit_head_dim_matches_hf(self):
        # Magistral-style config: `head_dim` is set explicitly and does not
        # equal `hidden_size // num_attention_heads`, and sliding window is
        # disabled. Build a small HF Mistral, convert, and check that the
        # keras-hub forward pass matches the HF reference to within the
        # precision of the fp16 hops used by the converter.
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        cfg = transformers.MistralConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=12,
            sliding_window=None,
            rope_theta=1_000_000.0,
            rms_norm_eps=1e-5,
        )
        self.assertNotEqual(
            cfg.head_dim, cfg.hidden_size // cfg.num_attention_heads
        )
        torch.manual_seed(0)
        hf_model = transformers.MistralForCausalLM(cfg).eval()

        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model.save_pretrained(preset_dir)
            keras_backbone = MistralBackbone.from_preset(preset_dir)

        input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="int32")
        padding = np.ones_like(input_ids)
        keras_out = np.asarray(
            keras_backbone({"token_ids": input_ids, "padding_mask": padding})
        )
        with torch.no_grad():
            hf_out = (
                hf_model.model(torch.tensor(input_ids))
                .last_hidden_state.detach()
                .cpu()
                .numpy()
            )
        self.assertEqual(keras_out.shape, hf_out.shape)
        # The converter stores weights in float16, so the parity bound is
        # dominated by fp16 quantization rather than implementation error.
        self.assertAllClose(keras_out, hf_out, atol=1e-2)
