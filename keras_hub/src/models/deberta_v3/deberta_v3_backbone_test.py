import pytest
from keras import ops

from keras_hub.src.models.deberta_v3.deberta_v3_backbone import (
    DebertaV3Backbone,
)
from keras_hub.src.tests.test_case import TestCase


class DebertaV3BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_sequence_length": 5,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=DebertaV3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DebertaV3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=DebertaV3Backbone,
            preset="deberta_v3_extra_small_en",
            input_data={
                "token_ids": ops.array([[0, 581, 63773, 2]], dtype="int32"),
                "padding_mask": ops.ones((1, 4), dtype="int32"),
            },
            expected_output_shape=(1, 4, 384),
            # The forward pass from a preset should be stable!
            expected_partial_output=ops.array(
                [0.418, -0.116, -0.122, -1.847, -0.035]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DebertaV3Backbone.presets:
            self.run_preset_test(
                cls=DebertaV3Backbone,
                preset=preset,
                input_data=self.input_data,
            )
