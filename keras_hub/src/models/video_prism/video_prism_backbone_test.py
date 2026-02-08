import pytest
from keras import ops

from keras_hub.src.models.video_prism.video_prism_backbone import (
    VideoPrismBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class VideoPrismBackboneVideoOnlyTest(TestCase):
    def setUp(self):
        self.image_size = 28
        self.num_frames = 4
        self.init_kwargs = {
            "image_shape": (self.image_size, self.image_size, 3),
            "num_frames": self.num_frames,
            "patch_size": 4,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "num_heads": 2,
            "num_spatial_layers": 1,
            "num_temporal_layers": 1,
            "num_auxiliary_layers": 1,
            "vocabulary_size": 0,
            "num_text_layers": 0,
        }
        self.input_data = ops.ones(
            (2, self.num_frames, self.image_size, self.image_size, 3),
            dtype="float32",
        )

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=VideoPrismBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, self.num_frames, 49, 16),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=VideoPrismBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=VideoPrismBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-3, "mean": 1e-5}},
        )


class VideoPrismBackboneTest(TestCase):
    def setUp(self):
        self.image_size = 28
        self.num_frames = 4
        self.init_kwargs = {
            "num_frames": self.num_frames,
            "patch_size": 4,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "num_heads": 2,
            "num_spatial_layers": 1,
            "num_temporal_layers": 1,
            "num_auxiliary_layers": 1,
            "vocabulary_size": 100,
            "num_text_layers": 1,
            "attention_logit_soft_cap": 50.0,
            "image_shape": (self.image_size, self.image_size, 3),
        }
        self.input_data = {
            "pixel_values": ops.ones(
                (2, self.num_frames, self.image_size, self.image_size, 3),
                dtype="float32",
            ),
            "token_ids": ops.ones((2, 12), dtype="int32"),
            "padding_mask": ops.ones((2, 12), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=VideoPrismBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "video_embeddings": (2, 16),
                "text_embeddings": (2, 16),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=VideoPrismBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=VideoPrismBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-3, "mean": 1e-5}},
        )
