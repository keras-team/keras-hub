import pytest
from keras import backend
from keras import ops

from keras_hub.src.models.clip.clip_text_encoder import CLIPTextEncoder
from keras_hub.src.models.flux.flux_backbone import FluxBackbone
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
            10,
            16,
            16,
            2,
            2,
            32,
            "quick_gelu",
            -2,
            name="clip_l",
        )
        self.init_kwargs = {
            "input_channels": 64,
            "hidden_size": 256,
            "mlp_ratio": 2.0,
            "num_heads": 4,
            "depth": 2,
            "depth_single_blocks": 4,
            "axes_dim": [8, 28, 28],
            "theta": 10_000,
            "use_bias": True,
            "guidance_embed": True,
            "image_shape": (16, 64),
            "text_shape": (16, 64),
            "image_ids_shape": (16, 3),
            "text_ids_shape": (16, 3),
            "y_shape": (64,),
        }

        self.pipeline_models = {
            "vae": vae,
            "clip_l": clip_l,
        }

        self.input_data = {
            "image": ops.ones((1, 16, 64)),
            "image_ids": ops.ones((1, 16, 3)),
            "text": ops.ones((1, 16, 64)),
            "text_ids": ops.ones((1, 16, 3)),
            "y": ops.ones((1, 64)),
            "timesteps": ops.ones((1)),
            "guidance": ops.ones((1)),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=FluxBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                1,
                16,
                64,
            ),
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

    @pytest.mark.xfail(
        condition=backend.backend() == "torch",
        reason="torch.export guard from Flux's dynamic num_heads reshape.",
    )
    def test_litert_export(self):
        self.run_litert_export_test(
            cls=FluxBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-4, "mean": 1e-5}},
        )


class FluxDynamicShapeTest(TestCase):
    """The published preset has `None` sequence axes, so it must be exercised.

    With static axes the final text/image split happens to work. With
    dynamic axes `text.shape[1]` is `None` during graph construction, and
    `image[:, None:, ...]` is a silent no-op that leaves the text tokens in
    the output -- the model still runs and still produces a plausible
    tensor, just with the wrong sequence length.
    """

    def test_output_contains_only_image_tokens(self):
        backbone = FluxBackbone(
            input_channels=64,
            hidden_size=256,
            mlp_ratio=2.0,
            num_heads=4,
            depth=2,
            depth_single_blocks=2,
            axes_dim=[8, 28, 28],
            theta=10_000,
            use_bias=True,
            guidance_embed=False,
            image_shape=(None, 64),
            text_shape=(None, 64),
            image_ids_shape=(None, 3),
            text_ids_shape=(None, 3),
            y_shape=(64,),
        )
        # Deliberately unequal: if the text tokens were not stripped the
        # output length would be 17 rather than 12.
        image_length, text_length = 12, 5
        output = backbone(
            {
                "image": ops.ones((1, image_length, 64)),
                "image_ids": ops.ones((1, image_length, 3)),
                "text": ops.ones((1, text_length, 64)),
                "text_ids": ops.ones((1, text_length, 3)),
                "y": ops.ones((1, 64)),
                "timesteps": ops.ones((1,)),
            }
        )
        self.assertEqual(tuple(ops.shape(output)), (1, image_length, 64))
