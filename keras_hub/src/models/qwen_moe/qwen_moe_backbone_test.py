import pytest
from keras import ops

from keras_hub.src.models.qwen_moe.qwen_moe_backbone import QwenMoeBackbone
from keras_hub.src.tests.test_case import TestCase


class QwenMoeBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 20,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "moe_intermediate_dim": 16,
            "shared_expert_intermediate_dim": 32,
            "num_experts": 4,
            "top_k": 2,
            "norm_topk_prob": True,
            "decoder_sparse_step": 1,
            "layer_norm_epsilon": 1e-6,
            "rope_max_wavelength": 10000,
            "rope_scaling_factor": 1.0,
            "dropout": 0.0,
            "use_sliding_window_attention": False,
            "sliding_window_size": 4096,
            "router_aux_loss_coef": 0.01,
            "tie_word_embeddings": False,
            "output_router_logits": False,
            "mlp_only_layers": [],
            "dtype": "float32",  # Explicitly set dtype to avoid mixed precision
        }
        self.input_data = {
            "token_ids": ops.ones((2, 7), dtype="int32"),
            "padding_mask": ops.ones((2, 7), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=QwenMoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 7, 16),
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=QwenMoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_architecture_characteristics(self):
        model = QwenMoeBackbone(**self.init_kwargs)
        expected_params = (
            # Token Embedding (forward and reverse, since
            # tie_word_embeddings=False)
            20 * 16 * 2  # 640
            # Transformer Layers
            + 2
            * (
                # Self-Attention
                (16 * 4 * 4 + 4 * 4)  # Query + Bias = 256 + 16
                + (16 * 2 * 4 + 2 * 4)  # Key + Bias = 128 + 8
                + (16 * 2 * 4 + 2 * 4)  # Value + Bias = 128 + 8
                + (4 * 4 * 16)  # Output = 256
                + 16  # Self-Attention LayerNorm
                # MoE
                + (16 * 4)  # Router = 64
                + 4 * (16 * 2 * 16)  # Experts Gate+Up = 2048
                + 4 * (16 * 16)  # Experts Output = 1024
                + (16 * 32)  # Shared Expert Gate = 512
                + (16 * 32)  # Shared Expert Intermediate = 512
                + (32 * 16)  # Shared Expert Output = 512
                + (16 * 1)  # Shared Expert Gate = 16
                + 16  # Feedforward LayerNorm
            )
            # Final LayerNorm
            + 16
        )
        # Should be 11696
        self.assertEqual(model.count_params(), expected_params)
        # token_embedding + 2 transformer layers + final norm + 2 inputs
        expected_layers = 6
        self.assertEqual(len(model.layers), expected_layers)

    def test_auxiliary_loss(self):
        model = QwenMoeBackbone(**self.init_kwargs)
        _ = model(self.input_data, training=True)
        self.assertTrue(
            len(model.losses) > 0, "Auxiliary losses should be present"
        )
        for loss in model.losses:
            self.assertGreater(loss, 0.0, "Auxiliary loss should be positive")
