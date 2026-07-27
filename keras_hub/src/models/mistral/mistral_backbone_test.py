import keras
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.tests.test_case import TestCase

# Scaled ~16x down from the width class every Mistral preset shares
# (vocab 32000, hidden 4096, intermediate 14336, 32 query / 8 kv heads).
# The 4:1 query:kv ratio is kept, since divisibility depends on it.
MISTRAL_7B_DIMS = {
    "name": "mistral_7b",
    "vocabulary_size": 2000,
    "num_layers": 1,
    "num_query_heads": 8,
    "num_key_value_heads": 2,
    "hidden_dim": 256,
    "intermediate_dim": 896,
    "sliding_window": 128,
}
# Not a real preset: a wider extrapolation, to cover an 8:1 query:kv ratio.
MISTRAL_LARGE_DIMS = {
    "name": "mistral_large",
    "vocabulary_size": 4000,
    "num_layers": 1,
    "num_query_heads": 16,
    "num_key_value_heads": 2,
    "hidden_dim": 512,
    "intermediate_dim": 1792,
    "sliding_window": 128,
}

# 3D shapes name their extra axis "seq"; no layout rule targets it, so it
# leaves weights whole and only the "model" axis size affects divisibility.
MESH_SHAPES = [(2, 4), (1, 8), (2, 2, 2)]

EXPECTED_SHARDINGS = {
    "token_embedding/embeddings": ("model", "batch"),
    "token_embedding/reverse_embeddings": ("batch", "model"),
    "self_attention.*query.kernel": ("batch", "model", None),
    "self_attention.*(key|value).kernel": ("batch", None, None),
    "self_attention.*attention_output.kernel": ("model", None, "batch"),
    "feedforward_intermediate_dense.kernel": ("batch", "model"),
    "feedforward_gate_dense.kernel": ("batch", "model"),
    "feedforward_output_dense.kernel": ("model", "batch"),
}


class MistralBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 8,
            "num_key_value_heads": 4,
            "hidden_dim": 16,
            "intermediate_dim": 8,
            "sliding_window": 2,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MistralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MistralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = MistralBackbone(**self.init_kwargs)
        # Reference value calculated using the PyTorch model
        self.assertEqual(model.count_params(), 2704)

    def test_explicit_head_dim(self):
        # Magistral-style config: `head_dim` is set explicitly and does not
        # equal `hidden_dim // num_query_heads`. `sliding_window=None` is
        # also exercised here. Run the full backbone test so the new path
        # gets serialization and precision coverage.
        init_kwargs = {
            **self.init_kwargs,
            "sliding_window": None,
            "head_dim": 4,
        }
        self.run_backbone_test(
            cls=MistralBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )
        model = MistralBackbone(**init_kwargs)
        attention = model.transformer_layers[0]._self_attention_layer
        self.assertEqual(attention._head_dim, 4)

    @pytest.mark.multi_device
    def test_distribution(self):
        self.run_distribution_test(
            cls=MistralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_shardings=EXPECTED_SHARDINGS,
        )

    @pytest.mark.multi_device
    def test_distribution_parity(self):
        self.run_distribution_parity_test(
            cls=MistralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @parameterized.named_parameters(
        (f"{dims['name']}_mesh_{'x'.join(str(s) for s in shape)}", dims, shape)
        for dims in (MISTRAL_7B_DIMS, MISTRAL_LARGE_DIMS)
        for shape in MESH_SHAPES
    )
    @pytest.mark.multi_device
    def test_layout_map_mesh_shapes(self, dims, mesh_shape):
        model_axis_size = mesh_shape[-1]
        num_devices = 1
        for size in mesh_shape:
            num_devices *= size
        devices = self._skip_unless_distribution(num_devices)[:num_devices]
        if dims["num_query_heads"] % model_axis_size:
            # An inherent tensor-parallelism ceiling, not a defect: the model
            # axis cannot exceed the query heads it splits.
            self.skipTest(
                f"num_query_heads={dims['num_query_heads']} is not divisible "
                f"by model-axis size {model_axis_size}."
            )

        axis_names = (
            ("batch", "model")
            if len(mesh_shape) == 2
            else ("batch", "seq", "model")
        )
        device_mesh = keras.distribution.DeviceMesh(
            shape=mesh_shape, axis_names=axis_names, devices=devices
        )
        layout_map = MistralBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(layout_map=layout_map)
        init_kwargs = {k: v for k, v in dims.items() if k != "name"}
        with distribution.scope():
            model = MistralBackbone(dtype="bfloat16", **init_kwargs)
            self.assert_sharding_specs(model, EXPECTED_SHARDINGS)
            self.assert_sharding_coverage(model, layout_map)

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MistralBackbone,
            preset="mistral_7b_en",
            input_data={
                "token_ids": ops.array([[1, 1824, 349, 524, 11234, 28804]]),
                "padding_mask": ops.ones((1, 6), dtype="int32"),
            },
            expected_output_shape=(1, 6, 4096),
            # The forward pass from a preset should be stable!
            # Reference values computed using PyTorch HF model.
            expected_partial_output=ops.array(
                [-1.6875, 0.5117, -1.7188, 2.3125, -0.0996]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MistralBackbone.presets:
            self.run_preset_test(
                cls=MistralBackbone,
                preset=preset,
                input_data=self.input_data,
            )
