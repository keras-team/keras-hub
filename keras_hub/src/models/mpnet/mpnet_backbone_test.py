import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.mpnet.mpnet_backbone import MPNetBackbone
from keras_hub.src.tests.test_case import TestCase


class MPNetBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 4,
            "intermediate_dim": 8,
            "max_sequence_length": 5,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MPNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 4),
        )

    def test_padding_does_not_change_outputs(self):
        backbone = MPNetBackbone(**self.init_kwargs)
        output = backbone(
            {
                "token_ids": np.array([[0, 5, 6, 2]], dtype="int32"),
                "padding_mask": np.array([[1, 1, 1, 1]], dtype="int32"),
            }
        )
        padded_output = backbone(
            {
                "token_ids": np.array([[0, 5, 6, 2, 1]], dtype="int32"),
                "padding_mask": np.array([[1, 1, 1, 1, 0]], dtype="int32"),
            }
        )
        self.assertAllClose(output, padded_output[:, :4, :], atol=1e-5)

    def test_shared_relative_position_bias(self):
        backbone = MPNetBackbone(**self.init_kwargs)
        # A single bias table is shared by all transformer layers.
        bias_weights = [
            weight
            for weight in backbone.weights
            if "relative_position_bias" in weight.path
        ]
        self.assertLen(bias_weights, 1)
        self.assertEqual(bias_weights[0].shape, (32, 2))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MPNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MPNetBackbone,
            preset="all_mpnet_base_v2_en",
            input_data={
                "token_ids": ops.array([[0, 2000, 4252, 2833, 2]], "int32"),
                "padding_mask": ops.ones((1, 5), dtype="int32"),
            },
            expected_output_shape=(1, 5, 768),
            # The forward pass from a preset should be stable!
            expected_partial_output=ops.array(
                [0.016, -0.159, 0.090, -0.106, -0.029]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MPNetBackbone.presets:
            self.run_preset_test(
                cls=MPNetBackbone,
                preset=preset,
                input_data=self.input_data,
            )
