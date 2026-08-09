import os

import keras
import pytest
from keras import ops

from keras_hub.src.models.dinov2.dinov2_backbone import DINOV2Backbone
from keras_hub.src.tests.test_case import TestCase


class DINOV2BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "patch_size": 14,
            "num_layers": 2,
            "hidden_dim": 16,
            "num_heads": 2,
            "intermediate_dim": 16 * 4,
            "layer_scale_init_value": 1.0,
            "num_register_tokens": 0,
            "use_swiglu_ffn": False,
            "image_shape": (70, 70, 3),
            "name": "dinov2_backbone",
        }
        self.input_data = {
            "images": ops.ones((2, 70, 70, 3)),
        }

    def test_backbone_basics(self):
        patch_size = self.init_kwargs["patch_size"]
        image_size = self.init_kwargs["image_shape"][0]
        hidden_dim = self.init_kwargs["hidden_dim"]
        sequence_length = (image_size // patch_size) ** 2 + 1
        self.run_vision_backbone_test(
            cls=DINOV2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, sequence_length, hidden_dim),
            expected_pyramid_output_keys=["stem", "stage1", "stage2"],
            expected_pyramid_image_sizes=[(sequence_length, hidden_dim)] * 3,
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DINOV2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=DINOV2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-4, "mean": 1e-5}},
        )

    @pytest.mark.large
    def test_position_embedding_interpolation(self):
        model = DINOV2Backbone(**self.init_kwargs)
        model_output = model(self.input_data)

        # Test not using interpolation in `save` and `load_model`.
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        restored_model = keras.models.load_model(path)
        restored_output = restored_model(self.input_data)
        self.assertAllClose(
            model_output, restored_output, atol=0.000001, rtol=0.000001
        )

        # Test using interpolation in `save_to_preset` and `from_preset` if
        # image_shape is different.
        path = os.path.join(self.get_temp_dir(), "model")
        model.save_to_preset(path)
        restored_model = DINOV2Backbone.from_preset(
            path,
            image_shape=(128, 128, 3),  # From 64 to 128.
        )
        input_data = {
            "images": ops.ones((2, 128, 128, 3)),
        }
        restored_output = restored_model(input_data)
        self.assertNotEqual(model_output.shape, restored_output.shape)

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
            "hidden_dim": 16,
            "num_heads": 2,
            "intermediate_dim": 16 * 4,
            "layer_scale_init_value": 1.0,
            "num_register_tokens": 4,
            "use_swiglu_ffn": True,
            "image_shape": (70, 70, 3),
            "name": "dinov2_backbone",
        }
        self.input_data = {
            "images": ops.ones((2, 70, 70, 3)),
        }

    def test_backbone_basics(self):
        patch_size = self.init_kwargs["patch_size"]
        image_size = self.init_kwargs["image_shape"][0]
        hidden_dim = self.init_kwargs["hidden_dim"]
        num_register_tokens = self.init_kwargs["num_register_tokens"]
        sequence_length = (
            (image_size // patch_size) ** 2 + 1 + num_register_tokens
        )
        self.run_vision_backbone_test(
            cls=DINOV2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, sequence_length, hidden_dim),
            expected_pyramid_output_keys=["stem", "stage1", "stage2"],
            expected_pyramid_image_sizes=[(sequence_length, hidden_dim)] * 3,
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DINOV2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=DINOV2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-4, "mean": 1e-5}},
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
