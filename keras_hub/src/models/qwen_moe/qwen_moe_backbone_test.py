import keras
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
            "norm_top_k_prob": True,
            "decoder_sparse_step": 1,
            "layer_norm_epsilon": 1e-6,
            "rope_max_wavelength": 10000,
            "rope_scaling_factor": 1.0,
            "dropout": 0.0,
            "use_sliding_window_attention": False,
            "sliding_window_size": 4096,
            "router_aux_loss_coefficient": 0.01,
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

    def test_distribution(self):
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")
        devices = keras.distribution.list_devices("CPU")
        if len(devices) == 1:
            self.skipTest("`ModelParallel` testing requires multiple devices.")
        devices = devices[:2]
        device_mesh = keras.distribution.DeviceMesh(
            shape=(1, 2),
            axis_names=("batch", "model"),
            devices=devices,
        )

        layout_map = QwenMoeBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(layout_map=layout_map)
        # `mlp_only_layers=[1]` makes layer 1 use the dense FFN fallback
        # (`qwen_moe_mlp`) while layer 0 stays sparse, so both the dense
        # fallback and the routed-expert layout rules are exercised here.
        init_kwargs = dict(self.init_kwargs, mlp_only_layers=[1])
        with distribution.scope():
            model = QwenMoeBackbone(**init_kwargs)

        for w in model.weights:
            if "token_embedding/embeddings" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch")
                )
            if "token_embedding/reverse_embeddings" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch")
                )
            if "self_attention/query/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch", None)
                )
            if "self_attention/key/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", None, None)
                )
            if "self_attention/value/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", None, None)
                )
            if "self_attention/attention_output/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", None, "batch")
                )
            if "qwen_moe_mlp/feedforward_intermediate_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "qwen_moe_mlp/feedforward_gate_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "qwen_moe_mlp/feedforward_output_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch")
                )
            if "experts/expert_feedforward_gate_dense" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec),
                    (None, "batch", "model"),
                )
            if "experts/expert_feedforward_output_dense" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec),
                    (None, "model", "batch"),
                )
            if "sparse_feedforward_gate_dense/kernel" in w.path:
                self.assertEqual(tuple(w.value.sharding.spec), ("batch", None))
            if "shared_expert_gate_dense/kernel" in w.path:
                self.assertEqual(tuple(w.value.sharding.spec), ("batch", None))
            if (
                "shared_expert_dense/feedforward_intermediate_dense/kernel"
                in w.path
            ):
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "shared_expert_dense/feedforward_gate_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "shared_expert_dense/feedforward_output_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch")
                )

    def test_auxiliary_loss(self):
        model = QwenMoeBackbone(**self.init_kwargs)
        _ = model(self.input_data, training=True)
        self.assertTrue(
            len(model.losses) > 0, "Auxiliary losses should be present"
        )
        for loss in model.losses:
            self.assertGreater(loss, 0.0, "Auxiliary loss should be positive")
