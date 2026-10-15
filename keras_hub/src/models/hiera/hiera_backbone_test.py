import keras
import pytest
from keras import ops

from keras_hub.src.models.hiera.hiera_backbone import HieraBackbone
from keras_hub.src.tests.test_case import TestCase


class HieraBackboneTest(TestCase):
    def setUp(self):
        # A tiny Hiera-tiny-shaped config that fits in a unit-test budget.
        # Using patch_stride=4 with image_shape=(64, 64, 3) means the
        # stage-0 feature map is 16x16 — cleanly divisible by the window
        # sizes and by three 2x2 pooling stages.
        self.init_kwargs = {
            "embed_dim": 8,
            "num_heads": 1,
            "stages": (1, 2, 3, 2),
            "global_attention_blocks": (4, 6),
            "window_spec": (8, 4, 4, 2),
            "window_pos_embed_bkg_spatial_size": (4, 4),
            "patch_kernel_size": (7, 7),
            "patch_stride": (4, 4),
            "q_stride": (2, 2),
            "mlp_ratio": 4.0,
            "image_shape": (64, 64, 3),
        }
        self.input_data = ops.ones((2, 64, 64, 3))

    def test_backbone_basics(self):
        # Final stage output is stride 4 * 2**3 = 32, with embed_dim
        # doubling three times: 8 -> 16 -> 32 -> 64.
        self.run_backbone_test(
            cls=HieraBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 2, 2, 64),
            # Mixed-precision dtype sweep is deferred; the softmax and
            # bicubic `ops.image.resize` hops leak fp32 in a few backend
            # configurations that would need a dedicated follow-up to
            # stabilize.
            run_mixed_precision_check=False,
            # Quantization support for this backbone is out of scope for
            # the initial PR; revisit once the SAM2 task class is in.
            run_quantization_check=False,
        )

    def test_feature_pyramid_outputs(self):
        backbone = HieraBackbone(**self.init_kwargs)
        pyramid = keras.Model(
            inputs=backbone.inputs, outputs=backbone.pyramid_outputs
        )
        outputs = pyramid(self.input_data)
        self.assertEqual(list(outputs.keys()), ["P2", "P3", "P4", "P5"])
        self.assertEqual(tuple(ops.shape(outputs["P2"])), (2, 16, 16, 8))
        self.assertEqual(tuple(ops.shape(outputs["P3"])), (2, 8, 8, 16))
        self.assertEqual(tuple(ops.shape(outputs["P4"])), (2, 4, 4, 32))
        self.assertEqual(tuple(ops.shape(outputs["P5"])), (2, 2, 2, 64))

    def test_global_attention_blocks_disable_windowing(self):
        # Spot-check that blocks in `global_attention_blocks` end up with
        # `window_size=0` (i.e. no windowed partition) while their peers keep
        # the stage's per-spec window size.
        backbone = HieraBackbone(**self.init_kwargs)
        for block_index, block in enumerate(backbone.blocks):
            expected_global = (
                block_index in self.init_kwargs["global_attention_blocks"]
            )
            if expected_global:
                self.assertEqual(block.window_size, 0)
            else:
                self.assertGreater(block.window_size, 0)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=HieraBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
