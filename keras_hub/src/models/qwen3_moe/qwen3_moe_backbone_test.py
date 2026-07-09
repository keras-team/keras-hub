import keras
import pytest
from keras import ops

from keras_hub.src.models.qwen3_moe.qwen3_moe_backbone import Qwen3MoeBackbone
from keras_hub.src.tests.test_case import TestCase


class Qwen3MoeBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 20,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "head_dim": 2,
            "moe_intermediate_dim": 16,
            "num_experts": 4,
            "top_k": 2,
            "norm_top_k_prob": True,
            "decoder_sparse_step": 1,
            "layer_norm_epsilon": 1e-6,
            "rope_max_wavelength": 10000,
            "rope_scaling_factor": 1.0,
            "dropout": 0.0,
            "sliding_window_size": 4096,
            "router_aux_loss_coefficient": 0.01,
            "tie_word_embeddings": False,
            "mlp_only_layers": [],
            "dtype": "float32",  # Explicitly set dtype to avoid mixed precision
        }
        self.input_data = {
            "token_ids": ops.ones((2, 7), dtype="int32"),
            "padding_mask": ops.ones((2, 7), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3MoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 7, 16),
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3MoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_architecture_characteristics(self):
        model = Qwen3MoeBackbone(**self.init_kwargs)
        expected_params = 7768
        self.assertEqual(model.count_params(), expected_params)
        expected_layers = 6
        self.assertEqual(len(model.layers), expected_layers)

    def test_auxiliary_loss(self):
        model = Qwen3MoeBackbone(**self.init_kwargs)
        _ = model(self.input_data, training=True)
        self.assertTrue(
            len(model.losses) > 0, "Auxiliary losses should be present"
        )
        for loss in model.losses:
            self.assertGreater(loss, 0.0, "Auxiliary loss should be positive")

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

        layout_map = Qwen3MoeBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(layout_map=layout_map)
        # `mlp_only_layers=[1]` makes layer 1 use the dense FFN fallback
        # (`qwen3_moe_mlp`) while layer 0 stays sparse, so both the dense
        # fallback and the routed-expert layout rules are exercised here.
        init_kwargs = dict(self.init_kwargs, mlp_only_layers=[1])
        with distribution.scope():
            model = Qwen3MoeBackbone(**init_kwargs)

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
                    tuple(w.value.sharding.spec), ("model", "batch", None)
                )
            if "self_attention/value/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch", None)
                )
            if "self_attention/attention_output/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", None, "batch")
                )
            if "qwen3_moe_mlp/feedforward_intermediate_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "qwen3_moe_mlp/feedforward_gate_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "qwen3_moe_mlp/feedforward_output_dense/kernel" in w.path:
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
