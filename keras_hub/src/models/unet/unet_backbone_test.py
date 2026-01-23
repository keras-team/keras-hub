import pytest
from keras import ops

from keras_hub.src.models.unet.unet_backbone import UNetBackbone
from keras_hub.src.tests.test_case import TestCase


class UNetBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "depth": 3,
            "filters": 32,
            "image_shape": (None, None, 3),
        }
        self.input_size = 128
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=UNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, self.input_size, self.input_size, 32),
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

    def test_different_depths(self):
        """Test UNet with different depths."""
        for depth in [2, 3, 4, 5]:
            init_kwargs = {
                **self.init_kwargs,
                "depth": depth,
            }
            model = UNetBackbone(**init_kwargs)
            output = model(self.input_data)
            self.assertEqual(
                output.shape, (2, self.input_size, self.input_size, 32)
            )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=UNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in UNetBackbone.presets:
            self.run_preset_test(
                cls=UNetBackbone,
                preset=preset,
                input_data=self.input_data,
            )
