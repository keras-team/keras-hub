import pytest
from keras import ops

from keras_hub.src.models.bge.bge_backbone import BgeBackbone
from keras_hub.src.tests.test_case import TestCase


class BgeBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 4,
            "intermediate_dim": 8,
            "max_sequence_length": 5,
            "num_segments": 2,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "segment_ids": ops.zeros((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=BgeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "sequence_output": (2, 5, 4),
                "pooled_output": (2, 4),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BgeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=BgeBackbone,
            preset="bge_small_en_v1.5",
            input_data={
                "token_ids": ops.array(
                    [[101, 1045, 2293, 3698, 4083, 102]], dtype="int32"
                ),
                "segment_ids": ops.zeros((1, 6), dtype="int32"),
                "padding_mask": ops.ones((1, 6), dtype="int32"),
            },
            expected_output_shape={
                "sequence_output": (1, 6, 384),
                "pooled_output": (1, 384),
            },
            expected_partial_output={
                "sequence_output": ops.array(
                    [
                        -0.24263829,
                        -0.25523514,
                        0.04677887,
                        -0.29345095,
                        0.48614883,
                    ]
                ),
                "pooled_output": ops.array(
                    [
                        -0.04671879,
                        0.15290071,
                        -0.04332585,
                        0.00902156,
                        -0.13474296,
                    ]
                ),
            },
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BgeBackbone.presets:
            self.run_preset_test(
                cls=BgeBackbone,
                preset=preset,
                input_data=self.input_data,
            )
