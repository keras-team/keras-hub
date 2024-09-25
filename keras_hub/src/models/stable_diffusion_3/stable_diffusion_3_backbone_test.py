import pytest
from keras import ops

from keras_hub.src.models.clip.clip_text_encoder import CLIPTextEncoder
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (
    StableDiffusion3Backbone,
)
from keras_hub.src.tests.test_case import TestCase


class StableDiffusion3BackboneTest(TestCase):
    def setUp(self):
        clip_l = CLIPTextEncoder(
            20, 32, 32, 2, 2, 64, "quick_gelu", -2, name="clip_l"
        )
        clip_g = CLIPTextEncoder(
            20, 64, 64, 2, 2, 128, "gelu", -2, name="clip_g"
        )
        self.init_kwargs = {
            "mmdit_patch_size": 2,
            "mmdit_hidden_dim": 16 * 2,
            "mmdit_num_layers": 2,
            "mmdit_num_heads": 2,
            "mmdit_position_size": 192,
            "vae_stackwise_num_filters": [32, 32, 32, 32],
            "vae_stackwise_num_blocks": [1, 1, 1, 1],
            "clip_l": clip_l,
            "clip_g": clip_g,
            "height": 64,
            "width": 64,
        }
        self.input_data = {
            "latents": ops.ones((2, 8, 8, 16)),
            "clip_l_token_ids": ops.ones((2, 5), dtype="int32"),
            "clip_l_negative_token_ids": ops.ones((2, 5), dtype="int32"),
            "clip_g_token_ids": ops.ones((2, 5), dtype="int32"),
            "clip_g_negative_token_ids": ops.ones((2, 5), dtype="int32"),
            "num_steps": ops.ones((2,), dtype="int32"),
            "guidance_scale": ops.ones((2,)),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=StableDiffusion3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 64, 64, 3),
            # Since `clip_l` and `clip_g` were instantiated outside of
            # `StableDiffusion3Backbone`, the mixed precision and
            # quantization checks will fail.
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=StableDiffusion3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
