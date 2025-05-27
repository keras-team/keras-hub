import pytest
from keras import ops

from keras_hub.src.models.mixtral.mixtral_backbone import MixtralBackbone
from keras_hub.src.tests.test_case import TestCase


class MixtralBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 8,
            "num_key_value_heads": 4,
            "hidden_dim": 16,
            "intermediate_dim": 8,
            "num_experts": 2,
            "top_k": 2,
            "sliding_window": 2,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MixtralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MixtralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = MixtralBackbone(**self.init_kwargs)
        # Calculated based on the model architecture:
        # - Token embedding: vocabulary_size * hidden_dim + hidden_dim *
        # vocabulary_size (tie_weights=False)
        # - Transformer layers: 2 * (attention + MoE block + layer norms)
        # - Attention: query + key + value + output
        # - MoE: experts (gate + intermediate + output) + router
        # - Layer norms: hidden_dim each
        head_dim = 16 // 8  # hidden_dim / num_query_heads
        expected_params = (
            10 * 16
            + 16 * 10  # Token embedding (embedding + output projection)
            + 2
            * (  # Two layers
                (  # Attention
                    16 * head_dim * 8  # Query
                    + 16 * head_dim * 4  # Key
                    + 16 * head_dim * 4  # Value
                    + 8 * head_dim * 16  # Output
                )
                + (  # MoE
                    2 * (16 * 8 + 16 * 8 + 8 * 16) + 16 * 2
                )
                + 2 * 16  # Two layer norms (self_attention + feedforward)
            )
            + 16  # Final layer norm
        )
        self.assertEqual(model.count_params(), expected_params)
