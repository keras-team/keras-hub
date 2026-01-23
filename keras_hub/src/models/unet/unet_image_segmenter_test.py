import pytest
from keras import ops

from keras_hub.src.models.unet.unet_backbone import UNetBackbone
from keras_hub.src.models.unet.unet_image_segmenter import UNetImageSegmenter
from keras_hub.src.tests.test_case import TestCase


class UNetImageSegmenterTest(TestCase):
    def setUp(self):
        self.backbone = UNetBackbone(
            depth=3,
            filters=32,
            image_shape=(None, None, 3),
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 2,
        }
        self.input_size = 128
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    def test_segmenter_basics(self):
        self.run_task_test(
            cls=UNetImageSegmenter,
            init_kwargs=self.init_kwargs,
            train_data=self.input_data,
            expected_output_shape=(2, self.input_size, self.input_size, 2),
        )

    def test_dynamic_input_shapes(self):
        """Test that the segmenter can handle different input sizes."""
        model = UNetImageSegmenter(**self.init_kwargs)

        # Test with different input sizes
        input_128 = ops.ones((1, 128, 128, 3))
        output_128 = model(input_128)
        self.assertEqual(output_128.shape, (1, 128, 128, 2))

        input_256 = ops.ones((1, 256, 256, 3))
        output_256 = model(input_256)
        self.assertEqual(output_256.shape, (1, 256, 256, 2))

        input_512 = ops.ones((1, 512, 512, 3))
        output_512 = model(input_512)
        self.assertEqual(output_512.shape, (1, 512, 512, 2))

    def test_num_classes(self):
        """Test with different number of classes."""
        for num_classes in [2, 5, 21]:
            init_kwargs = {
                "backbone": self.backbone,
                "num_classes": num_classes,
            }
            model = UNetImageSegmenter(**init_kwargs)
            output = model(self.input_data)
            self.assertEqual(
                output.shape,
                (2, self.input_size, self.input_size, num_classes),
            )

    def test_activation_none(self):
        """Test segmenter with no activation (logits output)."""
        init_kwargs = {
            **self.init_kwargs,
            "activation": None,
        }
        model = UNetImageSegmenter(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(output.shape, (2, self.input_size, self.input_size, 2))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=UNetImageSegmenter,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
