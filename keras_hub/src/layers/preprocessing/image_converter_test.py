import os
import pathlib

import numpy as np
import pytest
from absl.testing import parameterized
from keras import backend
from keras import ops

from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.pali_gemma.pali_gemma_image_converter import (
    PaliGemmaImageConverter,
)
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class ImageConverterTest(TestCase):
    def test_resize_simple(self):
        converter = ImageConverter(height=4, width=4, scale=1 / 255.0)
        inputs = np.ones((10, 10, 3)) * 255.0
        outputs = converter(inputs)
        self.assertAllClose(outputs, ops.ones((4, 4, 3)))

    def test_unbatched(self):
        converter = ImageConverter(
            image_size=(4, 4),
            scale=(1.0 / 255.0, 0.8 / 255.0, 1.2 / 255.0),
            offset=(0.2, -0.1, 0.25),
        )
        inputs = np.ones((10, 10, 3)) * 128
        outputs = converter(inputs)
        self.assertEqual(ops.shape(outputs), (4, 4, 3))
        self.assertAllClose(outputs[:, :, 0], np.ones((4, 4)) * 0.701961)
        self.assertAllClose(outputs[:, :, 1], np.ones((4, 4)) * 0.301569)
        self.assertAllClose(outputs[:, :, 2], np.ones((4, 4)) * 0.852353)

    @parameterized.parameters(
        (True, False),
        (False, True),
    )
    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="disabled until resize is fixed for torch backend",
    )  # TODO: remove skip after new release with fix of https://github.com/keras-team/keras/pull/20797
    def test_resize_batch(self, crop_to_aspect_ratio, pad_to_aspect_ratio):
        converter = ImageConverter(
            image_size=(4, 4),
            scale=(1.0 / 255.0, 0.8 / 255.0, 1.2 / 255.0),
            offset=(0.2, -0.1, 0.25),
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
        )
        inputs = np.ones((2, 10, 10, 3)) * 128
        outputs = converter(inputs)
        self.assertEqual(ops.shape(outputs), (2, 4, 4, 3))
        self.assertAllClose(outputs[:, :, :, 0], np.ones((2, 4, 4)) * 0.701961)
        self.assertAllClose(outputs[:, :, :, 1], np.ones((2, 4, 4)) * 0.301569)
        self.assertAllClose(outputs[:, :, :, 2], np.ones((2, 4, 4)) * 0.852353)

    def test_pad_and_crop_to_aspect_ratio(self):
        with self.assertRaisesRegex(ValueError, "Only one of"):
            _ = ImageConverter(
                image_size=(4, 4),
                scale=1 / 255.0,
                crop_to_aspect_ratio=True,
                pad_to_aspect_ratio=True,
            )

    def test_config(self):
        converter = ImageConverter(
            image_size=(12, 20),
            scale=(0.25 / 255.0, 0.1 / 255.0, 0.5 / 255.0),
            offset=(0.2, -0.1, 0.25),
            crop_to_aspect_ratio=False,
            interpolation="nearest",
        )
        clone = ImageConverter.from_config(converter.get_config())
        test_batch = np.random.rand(4, 10, 20, 3) * 255
        self.assertAllClose(converter(test_batch), clone(test_batch))

    def test_preset_accessors(self):
        pali_gemma_presets = set(PaliGemmaImageConverter.presets.keys())
        all_presets = set(ImageConverter.presets.keys())
        self.assertContainsSubset(pali_gemma_presets, all_presets)
        self.assertIn("pali_gemma_3b_mix_224", pali_gemma_presets)
        self.assertIn("pali_gemma_3b_mix_224", all_presets)

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            ImageConverter.from_preset("pali_gemma_3b_mix_224"),
            PaliGemmaImageConverter,
        )

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            ImageConverter.from_preset("bert_tiny_en_uncased")
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            ImageConverter.from_preset("hf://spacy/en_core_web_sm")

    @pytest.mark.large
    def test_save_to_preset(self):
        save_dir = self.get_temp_dir()
        converter = ImageConverter.from_preset(
            "resnet_50_imagenet",
            interpolation="nearest",
        )
        converter.save_to_preset(save_dir)
        # Save a tiny backbone so the preset is valid.
        backbone = ResNetBackbone(
            input_conv_filters=[64],
            input_conv_kernel_sizes=[7],
            stackwise_num_filters=[64, 64, 64],
            stackwise_num_blocks=[2, 2, 2],
            stackwise_num_strides=[1, 2, 2],
            block_type="basic_block",
            use_pre_activation=True,
        )
        backbone.save_to_preset(save_dir)

        # Check existence of files.
        path = pathlib.Path(save_dir)
        self.assertTrue(os.path.exists(path / "image_converter.json"))

        # Check loading.
        restored = ImageConverter.from_preset(save_dir)
        test_image = np.random.rand(100, 100, 3) * 255
        self.assertAllClose(restored(test_image), converter(test_image))
