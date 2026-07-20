from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_diffusion_gemma


class ConvertDiffusionGemmaTest(TestCase):
    def test_convert_backbone_config(self):
        transformers_config = {
            "model_type": "diffusion_gemma_text",
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "head_dim": 16,
            "layer_types": ["full_attention", "sliding_attention"],
        }
        kwargs = convert_diffusion_gemma.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(kwargs["num_layers"], 2)
        self.assertEqual(kwargs["hidden_dim"], 64)
        self.assertTrue(kwargs["has_encoder_layer_scalar"])
        self.assertTrue(kwargs["has_diffusion_self_conditioning"])

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
