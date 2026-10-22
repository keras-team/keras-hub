import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3_VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_vision_encoder import (
    Qwen3_VLVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase

TEXT_KWARGS = {
    "vocabulary_size": 10,
    "num_layers": 4,
    "num_query_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "hidden_dim": 16,
    "intermediate_dim": 32,
    "mrope_section": [2, 1, 1],
}


class Qwen3_VLBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = dict(TEXT_KWARGS)
        self.input_data = {
            "token_ids": np.arange(10, dtype="int32").reshape(2, 5),
            "padding_mask": np.ones((2, 5), dtype="int32"),
            "position_ids": np.tile(
                np.arange(5, dtype="int32")[:, None], (2, 1, 3)
            ),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3_VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )

    def test_position_ids_default_to_1d(self):
        model = Qwen3_VLBackbone(**self.init_kwargs)
        explicit = model(self.input_data)
        default_inputs = {
            k: v for k, v in self.input_data.items() if k != "position_ids"
        }
        self.assertAllClose(model(default_inputs), explicit)

    def test_position_ids_change_output(self):
        model = Qwen3_VLBackbone(**self.init_kwargs)
        # Default weights are small enough to make attention near uniform.
        rng = np.random.RandomState(0)
        model.set_weights([rng.randn(*w.shape) for w in model.get_weights()])
        shifted = dict(self.input_data)
        shifted["position_ids"] = self.input_data["position_ids"].copy()
        shifted["position_ids"][:, :, 1] *= 2
        base = ops.convert_to_numpy(model(self.input_data))
        out = ops.convert_to_numpy(model(shifted))
        self.assertGreater(np.abs(base - out).max(), 1e-6)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3_VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )


class Qwen3_VLMultimodalBackboneTest(TestCase):
    def setUp(self):
        self.vision_encoder = Qwen3_VLVisionEncoder(
            depth=3,
            hidden_size=16,
            num_heads=2,
            intermediate_size=32,
            out_hidden_size=16,
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=64,
            deepstack_visual_indexes=(0, 1),
        )
        self.init_kwargs = dict(TEXT_KWARGS, vision_encoder=self.vision_encoder)
        # One 4x4 patch grid merges into 4 vision tokens at positions 2..5.
        self.input_data = {
            "token_ids": np.arange(10, dtype="int32")[None, :],
            "padding_mask": np.ones((1, 10), dtype="int32"),
            "position_ids": np.tile(
                np.arange(10, dtype="int32")[:, None], (1, 1, 3)
            ),
            "pixel_values": np.random.RandomState(0)
            .randn(1, 16, 2, 4, 4, 3)
            .astype("float32"),
            "image_grid_thw": np.array([[[1, 4, 4]]], dtype="int32"),
            "vision_indices": np.array([[2, 3, 4, 5]], dtype="int32"),
        }

    def test_multimodal_forward_shape(self):
        model = Qwen3_VLBackbone(**self.init_kwargs)
        self.assertEqual(
            ops.shape(model(self.input_data)),
            (1, 10, 16),
        )

    def test_text_only_call_on_multimodal_backbone(self):
        model = Qwen3_VLBackbone(**self.init_kwargs)
        text_inputs = {
            k: self.input_data[k] for k in ("token_ids", "padding_mask")
        }
        self.assertEqual(ops.shape(model(text_inputs)), (1, 10, 16))

    def test_vision_inputs_change_output(self):
        model = Qwen3_VLBackbone(**self.init_kwargs)
        other = dict(self.input_data)
        other["pixel_values"] = self.input_data["pixel_values"] + 1.0
        base = ops.convert_to_numpy(model(self.input_data))
        out = ops.convert_to_numpy(model(other))
        self.assertGreater(np.abs(base - out).max(), 1e-6)

    def test_deepstack_depends_on_vision_indices(self):
        model = Qwen3_VLBackbone(**self.init_kwargs)
        moved = dict(self.input_data)
        moved["vision_indices"] = np.array([[3, 4, 5, 6]], dtype="int32")
        base = ops.convert_to_numpy(model(self.input_data))
        out = ops.convert_to_numpy(model(moved))
        self.assertGreater(np.abs(base - out).max(), 1e-6)

    def test_mismatched_vision_width(self):
        with self.assertRaises(ValueError):
            Qwen3_VLBackbone(**dict(self.init_kwargs, hidden_dim=32))

    def test_config_roundtrip(self):
        model = Qwen3_VLBackbone(**self.init_kwargs)
        clone = Qwen3_VLBackbone.from_config(model.get_config())
        self.assertEqual(
            clone.vision_encoder.deepstack_visual_indexes,
            self.vision_encoder.deepstack_visual_indexes,
        )
        self.assertEqual(clone.mrope_section, [2, 1, 1])
