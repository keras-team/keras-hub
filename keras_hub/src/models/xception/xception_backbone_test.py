import keras
import pytest
from keras import ops

from keras_hub.src.models.xception.xception_backbone import XceptionBackbone
from keras_hub.src.tests.test_case import TestCase


class XceptionBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stackwise_conv_filters": [[32, 64], [128, 128], [256, 256]],
            "stackwise_pooling": [False, True, False],
            "image_shape": (None, None, 3),
        }
        self.input_size = 64
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    def test_data_format_is_serialized(self):
        # `run_serialization_test` compares config to config, so a key that is
        # missing from `get_config` is missing on both sides and matches. The
        # value only matters once the global image data format differs from
        # the one the model was built with.
        backbone = XceptionBackbone(**self.init_kwargs)
        built = backbone.data_format
        config = backbone.get_config()
        self.assertEqual(config["data_format"], built)

        original = keras.config.image_data_format()
        other = (
            "channels_first" if built == "channels_last" else "channels_last"
        )
        try:
            keras.config.set_image_data_format(other)
            revived = XceptionBackbone.from_config(config)
            self.assertEqual(revived.data_format, built)
        finally:
            keras.config.set_image_data_format(original)

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=XceptionBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 15, 15, 256),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=XceptionBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in XceptionBackbone.presets:
            self.run_preset_test(
                cls=XceptionBackbone,
                preset=preset,
                input_data=self.input_data,
            )
