import os

import pytest
from keras import ops

from keras_hub.src.models.dinov2.dinov2_backbone import DINOV2Backbone
from keras_hub.src.tests.test_case import TestCase


class DINOV2BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "patch_size": 14,
            "num_layers": 2,
            "hidden_dim": 32,
            "num_heads": 2,
            "layer_scale_init_value": 1.0,
            "mlp_ratio": 4.0,
            "num_register_tokens": 0,
            "use_swiglu_ffn": False,
            "image_shape": (224, 224, 3),
        }
        self.input_data = {
            "images": ops.ones((2, 224, 224, 3)),
        }

    def test_backbone_basics(self):
        patch_size = self.init_kwargs["patch_size"]
        image_size = self.init_kwargs["image_shape"][0]
        hidden_dim = self.init_kwargs["hidden_dim"]
        sequence_length = (image_size // patch_size) ** 2 + 1
        self.run_backbone_test(
            cls=DINOV2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, sequence_length, hidden_dim),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DINOV2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_embedding_interpolation(self):
        model = DINOV2Backbone(**self.init_kwargs)
        path = os.path.join(self.get_temp_dir(), "model")
        model.save_to_preset(path)
        model = DINOV2Backbone.from_preset(
            path,
            image_shape=(518, 518, 3),  # From 224 to 518.
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.skipTest("Presets are not uploaded yet.")
        self.run_preset_test(
            cls=DINOV2Backbone,
            preset="dinov2_base",
            input_data=self.input_data,
            expected_output_shape=(2, 1374, 768),
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        self.skipTest("Presets are not uploaded yet.")
        for preset in DINOV2Backbone.presets:
            self.run_preset_test(
                cls=DINOV2Backbone,
                preset=preset,
                input_data=self.input_data,
            )


class DINOV2BackboneWithRegistersTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "patch_size": 14,
            "num_layers": 2,
            "hidden_dim": 32,
            "num_heads": 2,
            "layer_scale_init_value": 1.0,
            "mlp_ratio": 4.0,
            "num_register_tokens": 4,
            "use_swiglu_ffn": True,
            "image_shape": (224, 224, 3),
        }
        self.input_data = {
            "images": ops.ones((2, 224, 224, 3)),
        }

    def test_backbone_basics(self):
        patch_size = self.init_kwargs["patch_size"]
        image_size = self.init_kwargs["image_shape"][0]
        hidden_dim = self.init_kwargs["hidden_dim"]
        num_register_tokens = self.init_kwargs["num_register_tokens"]
        sequence_length = (
            (image_size // patch_size) ** 2 + 1 + num_register_tokens
        )
        self.run_backbone_test(
            cls=DINOV2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, sequence_length, hidden_dim),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DINOV2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.skipTest("Presets are not uploaded yet.")
        self.run_preset_test(
            cls=DINOV2Backbone,
            preset="dinov2_with_registers_base",
            input_data=self.input_data,
            expected_output_shape=(2, 1374, 768),
        )
