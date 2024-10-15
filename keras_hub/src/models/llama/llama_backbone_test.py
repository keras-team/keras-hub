import pytest
import keras
from keras import ops

from keras_hub.src.models.llama.llama_backbone import LlamaBackbone
from keras_hub.src.tests.test_case import TestCase


class LlamaTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 8,
            "intermediate_dim": 8,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=LlamaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=LlamaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = LlamaBackbone(**self.init_kwargs)
        # Reference value calculated using the PyTorch model
        self.assertEqual(model.count_params(), 968)

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=LlamaBackbone,
            preset="llama2_7b_en",
            input_data={
                "token_ids": ops.array([[1, 1824, 349, 524, 11234, 28804]]),
                "padding_mask": ops.ones((1, 6), dtype="int32"),
            },
            expected_output_shape=(1, 6, 4096),
            # The forward pass from a preset should be stable!
            # Reference values computed using PyTorch HF model.
            expected_partial_output=ops.array(
                [0.0153, 1.1657, 2.2452, -2.0192, -0.5801]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in LlamaBackbone.presets:
            self.run_preset_test(
                cls=LlamaBackbone,
                preset=preset,
                input_data=self.input_data,
            )

    def test_distribution(self):
        if keras.backend.backend() != "jax":
            return
        devices = keras.distribution.list_devices("CPU")
        if len(devices) == 1:
            # Need more than 1 device for distribution testing.
            return
        device_mesh = keras.distribution.DeviceMesh(
            shape=(1, len(devices)),
            axis_names=("batch", "model"),
            devices=devices,
        )

        layout_map = LlamaBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(layout_map=layout_map)
        with distribution.scope():
            model = LlamaBackbone(**self.init_kwargs)

        for w in model.weights:
            if "token_embedding/embeddings" in w.path:
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
            if "feedforward_intermediate_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "feedforward_gate_dense/kernel" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("batch", "model")
                )
            if "feedforward_output_dense" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), ("model", "batch")
                )

    def test_distribution_with_lora(self):
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")
        devices = keras.distribution.list_devices("CPU")
        if len(devices) == 1:
            # Need more than 1 device for distribution testing.
            self.skipTest("`ModelParallel` testing requires multiple devices.")
        device_mesh = keras.distribution.DeviceMesh(
            shape=(1, len(devices)),
            axis_names=("batch", "model"),
            devices=devices,
        )

        layout_map = LlamaBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(layout_map=layout_map)
        with distribution.scope():
            model = LlamaBackbone(**self.init_kwargs)
            model.enable_lora(rank=4)

        for w in model.weights:
            if "self_attention/query/lora_kernel_a" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), (None, None, None)
                )
            if "self_attention/query/lora_kernel_b" in w.path:
                self.assertEqual(tuple(w.value.sharding.spec), (None, None))
            if "self_attention/value/lora_kernel_a" in w.path:
                self.assertEqual(
                    tuple(w.value.sharding.spec), (None, None, None)
                )
            if "self_attention/value/lora_kernel_b" in w.path:
                self.assertEqual(tuple(w.value.sharding.spec), (None, None))
