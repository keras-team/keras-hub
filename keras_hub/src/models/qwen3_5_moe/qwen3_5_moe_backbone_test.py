import keras
import pytest
from keras import ops

from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_backbone import (
    Qwen3_5MoeBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3_5MoeBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 4,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_dim": 16,
            "moe_intermediate_dim": 8,
            "shared_expert_intermediate_size": 8,
            "num_experts": 4,
            "top_k": 2,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "router_aux_loss_coefficient": 0.01,
            "dtype": "float32",
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3_5MoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
            run_quantization_check=True,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3_5MoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = Qwen3_5MoeBackbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)

    def test_auxiliary_loss(self):
        model = Qwen3_5MoeBackbone(**self.init_kwargs)
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
        # Pinned to exactly 2 devices (not len(devices)): the default test
        # config's num_key_value_heads=2 is intentionally left as-is (not
        # divisible by every host's device count) to regression-test that
        # key/value kernels are left replicated rather than sharded -- see
        # get_layout_map's comment.
        devices = devices[:2]
        device_mesh = keras.distribution.DeviceMesh(
            shape=(1, 2),
            axis_names=("batch", "model"),
            devices=devices,
        )

        layout_map = Qwen3_5MoeBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(layout_map=layout_map)
        with distribution.scope():
            model = Qwen3_5MoeBackbone(**self.init_kwargs)

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
            if "shared_expert/feedforward_intermediate_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "shared_expert/feedforward_gate_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "shared_expert/feedforward_output_dense" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch")
                )
            if (
                "experts/expert_feedforward_gate_dense" in w.path
                and "shared_expert" not in w.path
            ):
                self.assertEqual(
                    tuple(w.value.sharding.spec), (None, "batch", "model")
                )
            if (
                "experts/expert_feedforward_output_dense" in w.path
                and "shared_expert" not in w.path
            ):
                self.assertEqual(
                    tuple(w.value.sharding.spec), (None, "model", "batch")
                )
            if "router_gate/kernel" in w.path:
                self.assertEqual(tuple(w.value.sharding.spec), ("batch", None))
