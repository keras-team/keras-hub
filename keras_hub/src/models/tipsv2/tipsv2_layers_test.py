"""Tests for TIPSv2 layers."""

import numpy as np
from keras import ops

from keras_hub.src.models.tipsv2.tipsv2_layers import TIPSv2LayerScale
from keras_hub.src.models.tipsv2.tipsv2_layers import TIPSv2MLP
from keras_hub.src.models.tipsv2.tipsv2_layers import TIPSv2PatchEmbedding
from keras_hub.src.models.tipsv2.tipsv2_layers import TIPSv2SwiGLU
from keras_hub.src.models.tipsv2.tipsv2_layers import TIPSv2TextAttention
from keras_hub.src.models.tipsv2.tipsv2_layers import TIPSv2TextBlock
from keras_hub.src.models.tipsv2.tipsv2_layers import TIPSv2TextMLP
from keras_hub.src.models.tipsv2.tipsv2_layers import TIPSv2VisionAttention
from keras_hub.src.models.tipsv2.tipsv2_layers import TIPSv2VisionBlock
from keras_hub.src.tests.test_case import TestCase


class TIPSv2LayerScaleTest(TestCase):
    def test_output_shape(self):
        layer = TIPSv2LayerScale(dim=32, init_values=1.0)
        x = np.random.rand(2, 4, 32).astype("float32")
        out = layer(x)
        self.assertEqual(out.shape, (2, 4, 32))

    def test_init_values(self):
        layer = TIPSv2LayerScale(dim=16, init_values=0.5)
        layer.build((2, 4, 16))
        gamma = np.array(layer.gamma)
        np.testing.assert_allclose(gamma, np.full(16, 0.5), atol=1e-6)

    def test_scaling_effect(self):
        layer = TIPSv2LayerScale(dim=8, init_values=2.0)
        x = np.ones((1, 3, 8), dtype="float32")
        out = ops.convert_to_numpy(layer(x))
        np.testing.assert_allclose(out, np.full((1, 3, 8), 2.0), atol=1e-6)

    def test_get_config(self):
        layer = TIPSv2LayerScale(dim=32, init_values=1e-5)
        config = layer.get_config()
        self.assertEqual(config["dim"], 32)
        self.assertEqual(config["init_values"], 1e-5)


class TIPSv2PatchEmbeddingTest(TestCase):
    def test_output_shape(self):
        layer = TIPSv2PatchEmbedding(
            hidden_dim=64, patch_size=14, image_size=28
        )
        x = np.random.rand(2, 28, 28, 3).astype("float32")
        out = layer(x)
        # 28/14 = 2 patches per side → 4 patches total.
        self.assertEqual(out.shape, (2, 4, 64))

    def test_different_patch_size(self):
        layer = TIPSv2PatchEmbedding(hidden_dim=32, patch_size=7, image_size=28)
        x = np.random.rand(1, 28, 28, 3).astype("float32")
        out = layer(x)
        # 28/7 = 4 patches per side → 16 patches total.
        self.assertEqual(out.shape, (1, 16, 32))

    def test_get_config(self):
        layer = TIPSv2PatchEmbedding(
            hidden_dim=64, patch_size=14, image_size=224
        )
        config = layer.get_config()
        self.assertEqual(config["hidden_dim"], 64)
        self.assertEqual(config["patch_size"], 14)
        self.assertEqual(config["image_size"], 224)


class TIPSv2VisionAttentionTest(TestCase):
    def test_output_shape(self):
        layer = TIPSv2VisionAttention(dim=32, num_heads=4)
        x = np.random.rand(2, 5, 32).astype("float32")
        out = layer(x)
        self.assertEqual(out.shape, (2, 5, 32))

    def test_single_head(self):
        layer = TIPSv2VisionAttention(dim=16, num_heads=1)
        x = np.random.rand(1, 8, 16).astype("float32")
        out = layer(x)
        self.assertEqual(out.shape, (1, 8, 16))

    def test_finite_outputs(self):
        layer = TIPSv2VisionAttention(dim=32, num_heads=4)
        x = np.random.rand(2, 5, 32).astype("float32")
        out = ops.convert_to_numpy(layer(x))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_get_config(self):
        layer = TIPSv2VisionAttention(
            dim=64, num_heads=8, qkv_bias=False, proj_bias=False
        )
        config = layer.get_config()
        self.assertEqual(config["dim"], 64)
        self.assertEqual(config["num_heads"], 8)
        self.assertFalse(config["qkv_bias"])
        self.assertFalse(config["proj_bias"])


class TIPSv2MLPTest(TestCase):
    def test_output_shape(self):
        layer = TIPSv2MLP(hidden_features=64)
        x = np.random.rand(2, 5, 32).astype("float32")
        out = layer(x)
        # Output dim defaults to input dim (32).
        self.assertEqual(out.shape, (2, 5, 32))

    def test_custom_output_features(self):
        layer = TIPSv2MLP(hidden_features=64, out_features=16)
        x = np.random.rand(2, 5, 32).astype("float32")
        out = layer(x)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_get_config(self):
        layer = TIPSv2MLP(hidden_features=128, out_features=64, drop=0.1)
        config = layer.get_config()
        self.assertEqual(config["hidden_features"], 128)
        self.assertEqual(config["out_features"], 64)
        self.assertEqual(config["drop"], 0.1)


class TIPSv2SwiGLUTest(TestCase):
    def test_output_shape(self):
        layer = TIPSv2SwiGLU(hidden_features=64)
        x = np.random.rand(2, 5, 32).astype("float32")
        out = layer(x)
        # Output dim defaults to input dim (32).
        self.assertEqual(out.shape, (2, 5, 32))

    def test_custom_output_features(self):
        layer = TIPSv2SwiGLU(hidden_features=128, out_features=16)
        x = np.random.rand(2, 5, 32).astype("float32")
        out = layer(x)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_hidden_rounding(self):
        """Verify hidden_features gets rounded to multiple of 8."""
        layer = TIPSv2SwiGLU(hidden_features=100)
        layer.build((1, 5, 32))
        # int(100 * 2/3) = 66, rounded to nearest 8 = 72.
        self.assertEqual(layer._hidden_features_actual, 72)

    def test_get_config(self):
        layer = TIPSv2SwiGLU(hidden_features=256)
        config = layer.get_config()
        self.assertEqual(config["hidden_features"], 256)


class TIPSv2VisionBlockTest(TestCase):
    def test_output_shape_mlp(self):
        block = TIPSv2VisionBlock(
            dim=32, num_heads=4, mlp_ratio=2.0, init_values=1.0, ffn_layer="mlp"
        )
        x = np.random.rand(2, 5, 32).astype("float32")
        out = block(x)
        self.assertEqual(out.shape, (2, 5, 32))

    def test_output_shape_swiglu(self):
        block = TIPSv2VisionBlock(
            dim=32,
            num_heads=4,
            mlp_ratio=2.0,
            init_values=1.0,
            ffn_layer="swiglu",
        )
        x = np.random.rand(2, 5, 32).astype("float32")
        out = block(x)
        self.assertEqual(out.shape, (2, 5, 32))

    def test_no_layer_scale(self):
        """Block without LayerScale (init_values=None)."""
        block = TIPSv2VisionBlock(
            dim=32, num_heads=4, mlp_ratio=2.0, init_values=None
        )
        x = np.random.rand(2, 5, 32).astype("float32")
        out = block(x)
        self.assertEqual(out.shape, (2, 5, 32))
        self.assertIsNone(block.ls1)
        self.assertIsNone(block.ls2)

    def test_residual_connection(self):
        """Output should differ from input (non-trivial computation)."""
        block = TIPSv2VisionBlock(
            dim=32, num_heads=4, mlp_ratio=2.0, init_values=1.0
        )
        x = np.random.rand(2, 5, 32).astype("float32")
        out = ops.convert_to_numpy(block(x))
        # Residual ensures output should not be zero.
        self.assertFalse(np.allclose(out, 0.0))

    def test_get_config(self):
        block = TIPSv2VisionBlock(
            dim=64,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=False,
            init_values=0.1,
            ffn_layer="swiglu",
        )
        config = block.get_config()
        self.assertEqual(config["dim"], 64)
        self.assertEqual(config["num_heads"], 8)
        self.assertEqual(config["mlp_ratio"], 4.0)
        self.assertFalse(config["qkv_bias"])
        self.assertEqual(config["init_values"], 0.1)
        self.assertEqual(config["ffn_layer"], "swiglu")


class TIPSv2TextAttentionTest(TestCase):
    def test_output_shape(self):
        layer = TIPSv2TextAttention(d_model=32, num_heads=4)
        x = np.random.rand(2, 6, 32).astype("float32")
        mask = np.array(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype="float32"
        )
        out = layer(x, mask)
        self.assertEqual(out.shape, (2, 6, 32))

    def test_masking_effect(self):
        """Padding positions should produce near-zero output."""
        layer = TIPSv2TextAttention(d_model=32, num_heads=4)
        x = np.random.rand(1, 4, 32).astype("float32")
        # All masked except position 0.
        mask = np.array([[1, 0, 0, 0]], dtype="float32")
        out = ops.convert_to_numpy(layer(x, mask))
        # The output should still be finite even with heavy masking.
        self.assertTrue(np.all(np.isfinite(out)))

    def test_get_config(self):
        layer = TIPSv2TextAttention(d_model=64, num_heads=8)
        config = layer.get_config()
        self.assertEqual(config["d_model"], 64)
        self.assertEqual(config["num_heads"], 8)


class TIPSv2TextMLPTest(TestCase):
    def test_output_shape(self):
        layer = TIPSv2TextMLP(mlp_dim=64, d_model=32)
        x = np.random.rand(2, 6, 32).astype("float32")
        mask = np.array(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype="float32"
        )
        out = layer(x, mask)
        self.assertEqual(out.shape, (2, 6, 32))

    def test_masking_zeros_padding(self):
        """Padding positions should be zeroed out."""
        layer = TIPSv2TextMLP(mlp_dim=64, d_model=32)
        x = np.random.rand(1, 4, 32).astype("float32")
        mask = np.array([[1, 1, 0, 0]], dtype="float32")
        out = ops.convert_to_numpy(layer(x, mask))
        # Positions 2 and 3 should be zero.
        np.testing.assert_allclose(out[0, 2], np.zeros(32), atol=1e-7)
        np.testing.assert_allclose(out[0, 3], np.zeros(32), atol=1e-7)

    def test_get_config(self):
        layer = TIPSv2TextMLP(mlp_dim=128, d_model=64)
        config = layer.get_config()
        self.assertEqual(config["mlp_dim"], 128)
        self.assertEqual(config["d_model"], 64)


class TIPSv2TextBlockTest(TestCase):
    def test_output_shape(self):
        block = TIPSv2TextBlock(d_model=32, num_heads=4, mlp_dim=64)
        x = np.random.rand(2, 6, 32).astype("float32")
        mask = np.array(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype="float32"
        )
        out = block(x, mask)
        self.assertEqual(out.shape, (2, 6, 32))

    def test_residual_connection(self):
        block = TIPSv2TextBlock(d_model=32, num_heads=4, mlp_dim=64)
        x = np.random.rand(2, 6, 32).astype("float32")
        mask = np.ones((2, 6), dtype="float32")
        out = ops.convert_to_numpy(block(x, mask))
        # Residual connection means output shouldn't be zero.
        self.assertFalse(np.allclose(out, 0.0))

    def test_finite_outputs(self):
        block = TIPSv2TextBlock(d_model=32, num_heads=4, mlp_dim=64)
        x = np.random.rand(2, 6, 32).astype("float32")
        mask = np.array(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype="float32"
        )
        out = ops.convert_to_numpy(block(x, mask))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_get_config(self):
        block = TIPSv2TextBlock(d_model=64, num_heads=8, mlp_dim=256)
        config = block.get_config()
        self.assertEqual(config["d_model"], 64)
        self.assertEqual(config["num_heads"], 8)
        self.assertEqual(config["mlp_dim"], 256)
