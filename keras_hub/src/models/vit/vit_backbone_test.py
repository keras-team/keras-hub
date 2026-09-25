import keras
import pytest
from keras import ops

from keras_hub.src.models.vit.vit_backbone import ViTBackbone
from keras_hub.src.tests.test_case import TestCase


class ViTBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "image_shape": (28, 28, 3),
            "patch_size": (4, 4),
            "num_layers": 3,
            "hidden_dim": 48,
            "num_heads": 6,
            "mlp_dim": 48 * 4,
            "use_mha_bias": True,
        }
        self.input_size = 28
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    def test_data_format_is_serialized(self):
        # `run_serialization_test` compares config to config, so a key that is
        # missing from `get_config` is missing on both sides and matches. The
        # value only matters once the global image data format differs from
        # the one the model was built with.
        backbone = ViTBackbone(**self.init_kwargs)
        built = backbone.data_format
        config = backbone.get_config()
        self.assertEqual(config["data_format"], built)

        original = keras.config.image_data_format()
        other = (
            "channels_first" if built == "channels_last" else "channels_last"
        )
        try:
            keras.config.set_image_data_format(other)
            revived = ViTBackbone.from_config(config)
            self.assertEqual(revived.data_format, built)
        finally:
            keras.config.set_image_data_format(original)

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=ViTBackbone,
            init_kwargs={**self.init_kwargs},
            input_data=self.input_data,
            expected_output_shape=(2, 50, 48),
        )

    def test_backbone_basics_without_class_token(self):
        self.init_kwargs["use_class_token"] = False
        self.run_backbone_test(
            cls=ViTBackbone,
            init_kwargs={**self.init_kwargs},
            input_data=self.input_data,
            expected_output_shape=(2, 49, 48),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ViTBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
