import os

import keras
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.unet.unet_backbone import UNetBackbone
from keras_hub.src.tests.test_case import TestCase


class UNetBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "depth": 3,
            "filters": 32,
            "image_shape": (None, None, 3),
            "data_format": "channels_last",
        }
        self.input_size = 128
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    def test_backbone_basics(self):
        expected_output_shape = (2, self.input_size, self.input_size, 32)
        self.run_vision_backbone_test(
            cls=UNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=expected_output_shape,
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    @parameterized.named_parameters(
        ("depth_2", 2, 16),
        ("depth_3", 3, 16),
        ("depth_4", 4, 32),
        ("depth_5", 5, 32),
    )
    def test_different_configs(self, depth, filters):
        """Test UNet with different depths and filters."""
        init_kwargs = {
            "depth": depth,
            "filters": filters,
            "image_shape": (None, None, 3),
            "data_format": "channels_last",
        }
        model = UNetBackbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(
            output.shape, (2, self.input_size, self.input_size, filters)
        )

    def test_dynamic_input_shapes(self):
        """Test that the model can handle different input sizes."""
        model = UNetBackbone(**self.init_kwargs)

        # Test with different input sizes
        input_128 = ops.ones((1, 128, 128, 3))
        output_128 = model(input_128)
        self.assertEqual(output_128.shape, (1, 128, 128, 32))

        input_256 = ops.ones((1, 256, 256, 3))
        output_256 = model(input_256)
        self.assertEqual(output_256.shape, (1, 256, 256, 32))

    def test_batch_norm(self):
        """Test UNet with batch normalization."""
        init_kwargs = {
            **self.init_kwargs,
            "use_batch_norm": True,
        }
        model = UNetBackbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(
            output.shape, (2, self.input_size, self.input_size, 32)
        )

    def test_dropout(self):
        """Test UNet with dropout."""
        init_kwargs = {
            **self.init_kwargs,
            "use_dropout": True,
            "dropout_rate": 0.5,
        }
        model = UNetBackbone(**init_kwargs)
        output = model(self.input_data, training=True)
        self.assertEqual(
            output.shape, (2, self.input_size, self.input_size, 32)
        )

    def test_kernel_initializer(self):
        """Test UNet with different kernel initializers."""
        init_kwargs = {
            **self.init_kwargs,
            "kernel_initializer": "glorot_uniform",
        }
        model = UNetBackbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(
            output.shape, (2, self.input_size, self.input_size, 32)
        )

    def test_dtype(self):
        """Test UNet with different dtypes."""
        init_kwargs = {
            **self.init_kwargs,
            "dtype": "bfloat16",
        }
        model = UNetBackbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(
            output.shape, (2, self.input_size, self.input_size, 32)
        )
        self.assertEqual(model.dtype_policy.name, "bfloat16")

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=UNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=UNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 5e-3, "mean": 5e-4}},
        )

    @pytest.mark.large
    def test_save_to_preset(self):
        save_dir = self.get_temp_dir()
        backbone = UNetBackbone(**self.init_kwargs)
        backbone.save_to_preset(save_dir)

        # Check existence of files.
        self.assertTrue(os.path.exists(os.path.join(save_dir, "config.json")))
        self.assertTrue(
            os.path.exists(os.path.join(save_dir, "model.weights.h5"))
        )
        self.assertTrue(os.path.exists(os.path.join(save_dir, "metadata.json")))

        # Try loading the model from preset directory.
        restored_backbone = UNetBackbone.from_preset(save_dir)

        # Check the model output.
        ref_out = backbone(self.input_data)
        new_out = restored_backbone(self.input_data)
        self.assertAllClose(ref_out, new_out)

    def test_upsampling_strategy_transpose(self):
        """Test UNet with Conv2DTranspose upsampling (default)."""
        init_kwargs = {
            **self.init_kwargs,
            "upsampling_strategy": "transpose",
        }
        model = UNetBackbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(
            output.shape, (2, self.input_size, self.input_size, 32)
        )

    def test_upsampling_strategy_interpolation(self):
        """Test UNet with interpolation upsampling to avoid checkerboard."""
        init_kwargs = {
            **self.init_kwargs,
            "upsampling_strategy": "interpolation",
        }
        model = UNetBackbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(
            output.shape, (2, self.input_size, self.input_size, 32)
        )

    def test_invalid_upsampling_strategy(self):
        """Test that invalid upsampling strategy raises error."""
        init_kwargs = {
            **self.init_kwargs,
            "upsampling_strategy": "invalid",
        }
        with self.assertRaisesRegex(ValueError, "upsampling_strategy"):
            UNetBackbone(**init_kwargs)

    def test_resunet(self):
        """Test ResUNet variant with residual connections."""
        init_kwargs = {
            **self.init_kwargs,
            "use_residual": True,
            "use_batch_norm": True,
        }
        model = UNetBackbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(
            output.shape, (2, self.input_size, self.input_size, 32)
        )
        # Verify ResUNet configuration is set correctly
        self.assertTrue(model.use_residual)
        self.assertTrue(model.use_batch_norm)
        self.assertTrue(model.encoder.use_residual)
        self.assertTrue(model.decoder.use_residual)

    def test_attention_unet(self):
        """Test Attention U-Net variant with attention gates."""
        init_kwargs = {
            **self.init_kwargs,
            "use_attention": True,
            "use_batch_norm": True,
        }
        model = UNetBackbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(
            output.shape, (2, self.input_size, self.input_size, 32)
        )
        # Verify Attention U-Net configuration is set correctly
        self.assertTrue(model.use_attention)
        self.assertTrue(model.use_batch_norm)
        self.assertTrue(model.decoder.use_attention)

    def test_advanced_unet_all_features(self):
        """Test UNet with all advanced features enabled."""
        init_kwargs = {
            **self.init_kwargs,
            "use_batch_norm": True,
            "use_dropout": True,
            "dropout_rate": 0.5,
            "upsampling_strategy": "interpolation",
            "use_residual": True,
            "use_attention": True,
        }
        model = UNetBackbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(
            output.shape, (2, self.input_size, self.input_size, 32)
        )

    def test_pretrained_backbone_resnet(self):
        """Test UNet with pretrained ResNet50 as encoder."""
        # Ensure channels_last format for backbone creation
        keras.config.set_image_data_format("channels_last")

        # Create a small ResNet-like backbone for testing
        backbone = keras.applications.ResNet50(
            include_top=False,
            weights=None,  # Random weights for testing
            input_shape=(None, None, 3),
        )

        init_kwargs = {
            "backbone": backbone,
            "use_batch_norm": True,
            "upsampling_strategy": "interpolation",
            "data_format": "channels_last",
        }
        model = UNetBackbone(**init_kwargs)
        output = model(self.input_data)
        # Output shape will match input spatial dimensions
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], self.input_size)
        self.assertEqual(output.shape[2], self.input_size)

    def test_pretrained_backbone_mobilenet(self):
        """Test UNet with pretrained MobileNetV2 as encoder."""
        # Ensure channels_last format for backbone creation
        keras.config.set_image_data_format("channels_last")

        backbone = keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=(None, None, 3),
        )

        init_kwargs = {
            "backbone": backbone,
            "use_batch_norm": True,
            "use_attention": True,
            "data_format": "channels_last",
        }
        model = UNetBackbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], self.input_size)
        self.assertEqual(output.shape[2], self.input_size)

    def test_config_with_pretrained_backbone(self):
        """Test get_config and from_config with pretrained backbone."""
        # Ensure channels_last format for backbone creation
        keras.config.set_image_data_format("channels_last")

        backbone = keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=(128, 128, 3),
        )

        init_kwargs = {
            "backbone": backbone,
            "use_batch_norm": True,
            "upsampling_strategy": "interpolation",
            "data_format": "channels_last",
        }
        model = UNetBackbone(**init_kwargs)
        config = model.get_config()

        # Verify config contains backbone
        self.assertIn("backbone", config)
        self.assertIn("upsampling_strategy", config)

        # Test from_config
        restored_model = UNetBackbone.from_config(config)
        output = restored_model(self.input_data)
        self.assertEqual(output.shape[0], 2)
