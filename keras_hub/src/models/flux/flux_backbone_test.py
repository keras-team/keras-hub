import pytest
from keras import ops

from keras_hub.src.models.clip.clip_text_encoder import CLIPTextEncoder
from keras_hub.src.models.flux.flux_model import FluxBackbone
from keras_hub.src.models.vae.vae_backbone import VAEBackbone
from keras_hub.src.tests.test_case import TestCase


class FluxBackboneTest(TestCase):
    def setUp(self):
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
            20, 32, 32, 2, 2, 64, "quick_gelu", -2, name="clip_l"
        )
        self.init_kwargs = {
            "input_channels": 256,
            "hidden_size": 1024,
            "mlp_ratio": 2.0,
            "num_heads": 8,
            "depth": 4,
            "depth_single_blocks": 8,
            "axes_dim": [16, 56, 56],
            "theta": 10_000,
            "use_bias": True,
            "guidance_embed": True,
            "image_shape": (32, 256),
            "text_shape": (32, 256),
            "image_ids_shape": (32, 3),
            "text_ids_shape": (32, 3),
            "y_shape": (256,),
        }

        self.pipeline_models = {
            "vae": vae,
            "clip_l": clip_l,
        }

        self.input_data = {
            "image": ops.ones((1, 32, 256)),
            "image_ids": ops.ones((1, 32, 3)),
            "text": ops.ones((1, 32, 256)),
            "text_ids": ops.ones((1, 32, 3)),
            "y": ops.ones((1, 256)),
            "timesteps": ops.ones((1)),
            "guidance": ops.ones((1)),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=FluxBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=[1, 32, 256],
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=FluxBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
