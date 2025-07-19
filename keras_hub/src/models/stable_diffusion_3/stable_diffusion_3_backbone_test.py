import pytest
from keras import ops

from keras_hub.src.models.clip.clip_text_encoder import CLIPTextEncoder
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (  # noqa: E501
    StableDiffusion3Backbone,
)
from keras_hub.src.models.vae.vae_backbone import VAEBackbone
from keras_hub.src.tests.test_case import TestCase


class StableDiffusion3BackboneTest(TestCase):
    def setUp(self):
        image_shape = (64, 64, 3)
        height, width = image_shape[0], image_shape[1]
        vae = VAEBackbone(
            [32, 32, 32, 32],
            [1, 1, 1, 1],
            [32, 32, 32, 32],
            [1, 1, 1, 1],
            # Use `mode` generate a deterministic output.
            sampler_method="mode",
            name="vae",
        )
        clip_l = CLIPTextEncoder(
            20,
            32,
            32,
            2,
            2,
            64,
            "quick_gelu",
            -2,
            # TODO: JAX CPU doesn't support float16 for
            # `nn.dot_product_attention`. We set dtype to float32 despite the
            # model defaulting to float16.
            dtype="float32",
            name="clip_l",
        )
        clip_g = CLIPTextEncoder(
            20,
            64,
            64,
            2,
            2,
            128,
            "gelu",
            -2,
            # TODO: JAX CPU doesn't support float16 for
            # `nn.dot_product_attention`. We set dtype to float32 despite the
            # model defaulting to float16.
            dtype="float32",
            name="clip_g",
        )
        self.init_kwargs = {
            "mmdit_patch_size": 2,
            "mmdit_hidden_dim": 16 * 2,
            "mmdit_num_layers": 2,
            "mmdit_num_heads": 2,
            "mmdit_position_size": 192,
            "mmdit_qk_norm": None,
            "mmdit_dual_attention_indices": None,
            "vae": vae,
            "clip_l": clip_l,
            "clip_g": clip_g,
            "image_shape": image_shape,
        }
        self.input_data = {
            "images": ops.ones((2, height, width, 3)),
            "latents": ops.ones((2, height // 8, width // 8, 16)),
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
            expected_output_shape={
                "images": (2, 64, 64, 3),
                "latents": (2, 8, 8, 16),
            },
            # Since `clip_l` and `clip_g` were instantiated outside of
            # `StableDiffusion3Backbone`, the mixed precision and
            # quantization checks will fail.
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    def test_backbone_basics_mmditx(self):
        # MMDiT-X includes `mmdit_qk_norm` and `mmdit_dual_attention_indices`.
        self.run_backbone_test(
            cls=StableDiffusion3Backbone,
            init_kwargs={
                **self.init_kwargs,
                "mmdit_qk_norm": "rms_norm",
                "mmdit_dual_attention_indices": (0,),
            },
            input_data=self.input_data,
            expected_output_shape={
                "images": (2, 64, 64, 3),
                "latents": (2, 8, 8, 16),
            },
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

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in StableDiffusion3Backbone.presets:
            self.run_preset_test(
                cls=StableDiffusion3Backbone,
                preset=preset,
                input_data=self.input_data,
                init_kwargs={
                    "image_shape": self.init_kwargs["image_shape"],
                },
            )
