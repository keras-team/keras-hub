import keras
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

    def test_distribution(self):
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")
        devices = keras.distribution.list_devices("CPU")
        if len(devices) == 1:
            self.skipTest("`ModelParallel` testing requires multiple devices.")
        # Pinned to exactly 2 devices (not len(devices)): the default test
        # config's num_key_value_heads=4 is intentionally left as-is (not
        # divisible by every host's device count) to regression-test that
        # key/value kernels are left replicated rather than sharded -- see
        # get_layout_map's comment.
        devices = devices[:2]
        device_mesh = keras.distribution.DeviceMesh(
            shape=(1, 2),
            axis_names=("batch", "model"),
            devices=devices,
        )

        layout_map = GptOssBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(layout_map=layout_map)
        with distribution.scope():
            model = GptOssBackbone(**self.init_kwargs)

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
            if "experts/gate_up_proj" in w.path and "bias" not in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), (None, "batch", "model")
                )
            if "experts/down_proj" in w.path and "bias" not in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), (None, "model", "batch")
                )
            if "router_dense/kernel" in w.path:
                self.assertEqual(tuple(w.value.sharding.spec), ("batch", None))
