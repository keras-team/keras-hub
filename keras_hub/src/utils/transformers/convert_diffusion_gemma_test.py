import pytest

from keras_hub.src.models.diffusion_gemma.diffusion_gemma_backbone import (
    DiffusionGemmaBackbone,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_diffusion_gemma


class ConvertDiffusionGemmaTest(TestCase):
    def test_convert_backbone_config(self):
        transformers_config = {
            "model_type": "diffusion_gemma_text",
            "vocab_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "head_dim": 16,
            "global_head_dim": 32,
            "num_global_key_value_heads": 1,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": None,
            "sliding_window": 512,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {"rope_theta": 1000000.0},
                "sliding_attention": {"rope_theta": 10000.0},
            },
            "layer_types": ["full_attention", "sliding_attention"],
        }
        kwargs = convert_diffusion_gemma.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(kwargs["num_layers"], 2)
        self.assertEqual(kwargs["hidden_dim"], 64)
        self.assertEqual(kwargs["vocabulary_size"], 256)
        self.assertEqual(kwargs["global_head_dim"], 32)
        self.assertEqual(kwargs["attention_logit_soft_cap"], 50.0)

    def test_convert_task_config(self):
        transformers_config = {
            "canvas_length": 128,
            "max_denoising_steps": 10,
        }
        kwargs = convert_diffusion_gemma.convert_task_config(
            transformers_config
        )
        self.assertEqual(kwargs["canvas_length"], 128)
        self.assertEqual(kwargs["max_denoising_steps"], 10)

    def test_load_preprocessor_config_defaults(self):
        temp_dir = self.get_temp_dir()
        kwargs = convert_diffusion_gemma.load_preprocessor_config(temp_dir, {})
        self.assertFalse(kwargs["add_start_token"])
        self.assertFalse(kwargs["add_end_token"])

    @pytest.mark.extra_large
    def test_backbone_from_hf_preset(self):
        model = DiffusionGemmaBackbone.from_preset(
            "hf://google/diffusiongemma-26B-A4B-it",
            load_weights=False,
        )
        self.assertEqual(model.num_layers, 30)
