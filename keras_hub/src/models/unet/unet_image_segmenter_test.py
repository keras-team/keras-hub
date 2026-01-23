import os

import keras
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
        # Create dummy segmentation labels (one-hot encoded)
        self.labels = ops.one_hot(
            ops.zeros((2, self.input_size, self.input_size), dtype="int32"),
            num_classes=2,
        )
        self.train_data = (self.input_data, self.labels)

    @pytest.mark.skipif(
        keras.backend.backend() != "tensorflow",
        reason="run_task_test uses TensorFlow-specific tf.data.Dataset",
    )
    def test_segmenter_basics(self):
        self.run_task_test(
            cls=UNetImageSegmenter,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
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

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=UNetImageSegmenter,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 5e-5, "mean": 1e-5}},
        )

    def test_dtype(self):
        """Test segmenter with different dtypes."""
        backbone = UNetBackbone(
            depth=3,
            filters=32,
            dtype="bfloat16",
        )
        init_kwargs = {
            "backbone": backbone,
            "num_classes": 2,
        }
        model = UNetImageSegmenter(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(output.shape, (2, self.input_size, self.input_size, 2))
        # Check that the backbone uses the correct dtype
        self.assertEqual(model.backbone.dtype_policy.name, "bfloat16")

    @pytest.mark.large
    def test_save_to_preset(self):
        save_dir = self.get_temp_dir()
        segmenter = UNetImageSegmenter(**self.init_kwargs)
        segmenter.save_to_preset(save_dir)

        # Check existence of files.
        self.assertTrue(os.path.exists(os.path.join(save_dir, "config.json")))
        self.assertTrue(
            os.path.exists(os.path.join(save_dir, "model.weights.h5"))
        )
        self.assertTrue(os.path.exists(os.path.join(save_dir, "metadata.json")))

        # Try loading the model from preset directory.
        restored_segmenter = UNetImageSegmenter.from_preset(save_dir)

        # Check the model output.
        ref_out = segmenter(self.input_data)
        new_out = restored_segmenter(self.input_data)
        self.assertAllClose(ref_out, new_out)
