import pytest
from keras import ops

from keras_hub.src.models.opt.opt_backbone import OPTBackbone
from keras_hub.src.tests.test_case import TestCase


class OPTBackboneTest(TestCase):
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
            cls=OPTBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=OPTBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=OPTBackbone,
            preset="opt_125m_en",
            input_data={
                "token_ids": ops.array([[133, 2119, 6219, 23602, 4]]),
                "padding_mask": ops.ones((1, 5), dtype="int32"),
            },
            expected_output_shape=(1, 5, 768),
            # The forward pass from a preset should be stable!
            expected_partial_output=ops.array(
                [-0.246, -1.004, -0.072, 0.097, 0.533]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in OPTBackbone.presets:
            self.run_preset_test(
                cls=OPTBackbone,
                preset=preset,
                input_data=self.input_data,
            )
