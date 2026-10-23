import pytest
from keras import backend
from keras import ops

from keras_hub.src.models.flux2.flux2_backbone import Flux2Backbone
from keras_hub.src.tests.test_case import TestCase


class Flux2BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "input_channels": 16,
            "hidden_size": 64,
            "mlp_ratio": 3.0,
            "num_heads": 2,
            "depth": 2,
            "depth_single_blocks": 2,
            "axes_dim": [8, 8, 8, 8],
            "theta": 2000,
            "guidance_embed": False,
            "image_shape": (12, 16),
            "text_shape": (5, 24),
            "image_ids_shape": (12, 4),
            "text_ids_shape": (5, 4),
        }
        self.input_data = {
            "image": ops.ones((2, 12, 16)),
            "image_ids": ops.ones((2, 12, 4)),
            "text": ops.ones((2, 5, 24)),
            "text_ids": ops.ones((2, 5, 4)),
            "timesteps": ops.ones((2,)),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Flux2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 12, 16),
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    def test_guidance_embed(self):
        backbone = Flux2Backbone(**{**self.init_kwargs, "guidance_embed": True})
        self.assertIn("guidance", backbone.input)
        output = backbone({**self.input_data, "guidance": ops.ones((2,))})
        self.assertEqual(tuple(ops.shape(output)), (2, 12, 16))

    def test_no_biases(self):
        """No FLUX.2 linear layer has a bias; the checkpoint has none."""
        backbone = Flux2Backbone(**self.init_kwargs)
        biases = [w.path for w in backbone.weights if "bias" in w.path]
        self.assertEqual(biases, [])

    def test_modulation_is_shared(self):
        """One modulation projection per stream type, not one per block."""
        backbone = Flux2Backbone(**self.init_kwargs)
        modulation_weights = [
            w.path for w in backbone.weights if "modulation" in w.path
        ]
        # Image, text and single-stream projections, plus the final layer's
        # adaLN projection. Independent of `depth` and `depth_single_blocks`.
        self.assertEqual(len(modulation_weights), 4)

    def test_rejects_head_dim_mismatch(self):
        with self.assertRaisesRegex(ValueError, "axes_dim"):
            Flux2Backbone(**{**self.init_kwargs, "axes_dim": [8, 8, 8]})

    def test_rejects_indivisible_heads(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            Flux2Backbone(**{**self.init_kwargs, "num_heads": 3})

    def test_rejects_ids_axes_mismatch(self):
        with self.assertRaisesRegex(ValueError, "image_ids_shape"):
            Flux2Backbone(**{**self.init_kwargs, "image_ids_shape": (12, 3)})

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Flux2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.xfail(
        condition=backend.backend() == "torch",
        reason="torch.export guard from the dynamic num_heads reshape.",
    )
    def test_litert_export(self):
        self.run_litert_export_test(
            cls=Flux2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-4, "mean": 1e-5}},
        )


class Flux2DynamicShapeTest(TestCase):
    """Presets keep the sequence axes dynamic, so exercise that path.

    With dynamic axes `text.shape[1]` is `None` while the functional graph
    is built; the text tokens must still be stripped from the output.
    """

    def test_output_contains_only_image_tokens(self):
        backbone = Flux2Backbone(
            input_channels=16,
            hidden_size=64,
            mlp_ratio=3.0,
            num_heads=2,
            depth=1,
            depth_single_blocks=2,
            axes_dim=[8, 8, 8, 8],
            theta=2000,
            image_shape=(None, 16),
            text_shape=(None, 24),
        )
        # Deliberately unequal, and different from any earlier call.
        for image_length, text_length in ((12, 5), (7, 9)):
            output = backbone(
                {
                    "image": ops.ones((1, image_length, 16)),
                    "image_ids": ops.ones((1, image_length, 4)),
                    "text": ops.ones((1, text_length, 24)),
                    "text_ids": ops.ones((1, text_length, 4)),
                    "timesteps": ops.ones((1,)),
                }
            )
            self.assertEqual(tuple(ops.shape(output)), (1, image_length, 16))
