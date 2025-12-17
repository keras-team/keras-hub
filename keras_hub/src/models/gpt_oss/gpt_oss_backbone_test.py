import pytest
from keras import ops

from keras_hub.src.models.gpt_oss.gpt_oss_backbone import GptOssBackbone
from keras_hub.src.tests.test_case import TestCase


class GptOssBackboneTest(TestCase):
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
            cls=GptOssBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
            run_quantization_check=True,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=GptOssBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = GptOssBackbone(**self.init_kwargs)
        # Calculated based on the model architecture:
        # - Token embedding: vocabulary_size * hidden_dim
        # - Output projection: hidden_dim * vocabulary_size
        # - Transformer layers: num_layers * (attention + MoE block + LNs)
        # - Attention: q, k, v, o projections + sinks
        # - MoE: router (w+b) + experts (gate_up_proj (w+b), down_proj (w+b))
        # - Layer norms: hidden_dim each
        self.assertEqual(model.count_params(), 3780)
