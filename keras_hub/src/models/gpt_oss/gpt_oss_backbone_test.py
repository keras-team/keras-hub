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
            "rope_max_wavelength": 10000,
            "rope_scaling_factor": 1.0,
            "layer_norm_epsilon": 1e-6,
            "dropout": 0.0,
            "use_bias": False,
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
            expected_output_shape=(
                2,
                5,
                16,
            ),  # (batch_size, sequence_length, hidden_dim)
            run_quantization_check=False,
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
        # - Final Layer Norm: hidden_dim
        # - Per Decoder Layer (num_layers times):
        #   - Input Layer Norm: hidden_dim
        #   - Post-Attention Layer Norm: hidden_dim
        #   - Attention (GptOssAttention):
        #     - q_proj: hidden_dim * (num_query_heads * head_dim)
        #     - k_proj: hidden_dim * (num_key_value_heads * head_dim)
        #     - v_proj: hidden_dim * (num_key_value_heads * head_dim)
        #     - o_proj: (num_query_heads * head_dim) * hidden_dim
        #     - sinks: num_query_heads
        #   - MLP (GptOssMLP):
        #     - Router (GptOssTopKRouter):
        #       - weight: num_experts * hidden_dim
        #       - bias: num_experts
        #     - Experts (GptOssExperts):
        #       - gate_up_proj: num_experts * hidden_dim *(2 *intermediate_dim)
        #       - gate_up_proj_bias: num_experts * (2 * intermediate_dim)
        #       - down_proj: num_experts * intermediate_dim * hidden_dim
        #       - down_proj_bias: num_experts * hidden_dim

        vocabulary_size = self.init_kwargs["vocabulary_size"]
        num_layers = self.init_kwargs["num_layers"]
        num_query_heads = self.init_kwargs["num_query_heads"]
        num_key_value_heads = self.init_kwargs["num_key_value_heads"]
        hidden_dim = self.init_kwargs["hidden_dim"]
        intermediate_dim = self.init_kwargs["intermediate_dim"]
        num_experts = self.init_kwargs["num_experts"]
        use_bias = self.init_kwargs["use_bias"]

        head_dim = hidden_dim // num_query_heads  # 16 // 8 = 2

        # Token Embedding
        token_embedding_params = vocabulary_size * hidden_dim  # 10 * 16 = 160

        # Final Layer Norm
        final_norm_params = hidden_dim  # 16

        # Per Decoder Layer
        layer_params = 0
        # Input Layer Norm
        layer_params += hidden_dim  # 16
        # Post-Attention Layer Norm
        layer_params += hidden_dim  # 16

        # Attention (GptOssAttention)
        attention_params = 0
        attention_params += hidden_dim * (
            num_query_heads * head_dim
        )  # q_proj: 16 * (8 * 2) = 256
        attention_params += hidden_dim * (
            num_key_value_heads * head_dim
        )  # k_proj: 16 * (4 * 2) = 128
        attention_params += hidden_dim * (
            num_key_value_heads * head_dim
        )  # v_proj: 16 * (4 * 2) = 128
        attention_params += (
            num_query_heads * head_dim
        ) * hidden_dim  # o_proj: (8 * 2) * 16 = 256
        if use_bias:
            attention_params += num_query_heads * head_dim  # q_proj bias
            attention_params += num_key_value_heads * head_dim  # k_proj bias
            attention_params += num_key_value_heads * head_dim  # v_proj bias
            attention_params += hidden_dim  # o_proj bias
        attention_params += num_query_heads  # sinks: 8
        # Total Attention: 256 + 128 + 128 + 256 + 8 = 776
        layer_params += attention_params

        # MLP (GptOssMLP)
        mlp_params = 0
        # Router (GptOssTopKRouter)
        router_params = 0
        router_params += num_experts * hidden_dim  # weight: 2 * 16 = 32
        router_params += num_experts  # bias: 2
        # Total Router: 32 + 2 = 34
        mlp_params += router_params

        # Experts (GptOssExperts)
        experts_params = 0
        experts_params += (
            num_experts * hidden_dim * (2 * intermediate_dim)
        )  # gate_up_proj: 2 * 16 * (2 * 8) = 512
        experts_params += num_experts * (
            2 * intermediate_dim
        )  # gate_up_proj_bias: 2 * (2 * 8) = 32
        experts_params += (
            num_experts * intermediate_dim * hidden_dim
        )  # down_proj: 2 * 8 * 16 = 256
        experts_params += (
            num_experts * hidden_dim
        )  # down_proj_bias: 2 * 16 = 32
        # Total Experts: 512 + 32 + 256 + 32 = 832
        mlp_params += experts_params
        # Total MLP: 34 + 832 = 866
        layer_params += mlp_params

        # Total expected parameters
        expected_params = (
            token_embedding_params
            + final_norm_params
            + num_layers * layer_params
        )

        self.assertEqual(model.count_params(), expected_params)
