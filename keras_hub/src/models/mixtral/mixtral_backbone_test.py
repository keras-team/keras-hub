import keras
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

        layout_map = MixtralBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(layout_map=layout_map)
        with distribution.scope():
            model = MixtralBackbone(**self.init_kwargs)

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
            if "experts/expert_feedforward_gate_dense" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec),
                    (None, "batch", "model"),
                )
            if "experts/expert_feedforward_intermediate_dense" in w.path:
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
