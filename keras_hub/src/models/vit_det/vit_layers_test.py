import numpy as np
from keras import ops

from keras_hub.src.models.vit_det.vit_layers import (
    AddRelativePositionalEmbedding,
)
from keras_hub.src.models.vit_det.vit_layers import WindowedTransformerEncoder
from keras_hub.src.tests.test_case import TestCase


class AddRelativePositionalEmbeddingTest(TestCase):
    def setUp(self):
        # Four (query, key) pairs of the position grid.
        self.rows = [0, 5, 20, 39]
        self.cols = [0, 30, 20, 0]
        self.table = np.random.RandomState(0).normal(size=(127, 4))
        self.table = self.table.astype("float32")

    def test_get_rel_pos_when_table_resizes(self):
        # A 64x64 table has 127 rows. A 40x40 grid needs 79 rows.
        layer = AddRelativePositionalEmbedding(input_size=(64, 64), key_dim=4)
        rel_pos = ops.convert_to_numpy(layer._get_rel_pos(40, 40, self.table))
        self.assertEqual(rel_pos.shape, (40, 40, 4))
        # Values from `get_rel_pos` of the `transformers` SAM attention.
        expected = [
            [0.1420, -0.3193, 0.6915, 0.6947],
            [-0.0346, 0.9914, 0.2121, 0.8045],
            [0.1420, -0.3193, 0.6915, 0.6947],
            [-0.1256, -0.4151, -0.8036, -1.1626],
        ]
        self.assertAllClose(
            rel_pos[self.rows, self.cols], expected, atol=1e-4, rtol=1e-4
        )

    def test_get_rel_pos_when_table_matches(self):
        layer = AddRelativePositionalEmbedding(input_size=(64, 64), key_dim=4)
        rel_pos = ops.convert_to_numpy(layer._get_rel_pos(64, 64, self.table))
        self.assertEqual(rel_pos.shape, (64, 64, 4))
        expected = [
            [0.1420, -0.3193, 0.6915, 0.6947],
            [-0.7448, -0.8264, -0.0985, -0.6635],
            [0.1420, -0.3193, 0.6915, 0.6947],
            [-0.1851, -0.8076, -1.4465, 0.8003],
        ]
        self.assertAllClose(
            rel_pos[self.rows, self.cols], expected, atol=1e-4, rtol=1e-4
        )

    def test_attention_map_when_table_resizes(self):
        layer = AddRelativePositionalEmbedding(input_size=(16, 16), key_dim=4)
        rng = np.random.RandomState(0)
        height, width = 10, 7
        layer.rel_pos_h.assign(rng.normal(size=(31, 4)).astype("float32"))
        layer.rel_pos_w.assign(rng.normal(size=(31, 4)).astype("float32"))
        queries = rng.normal(size=(2, height * width, 4)).astype("float32")
        attention_map = rng.normal(
            size=(2, height * width, height * width)
        ).astype("float32")
        output = ops.convert_to_numpy(
            layer(
                attention_map,
                queries=queries,
                query_size=(height, width),
                key_size=(height, width),
            )
        )
        self.assertEqual(output.shape, (2, 70, 70))
        # Values from `get_decomposed_rel_pos` of the `transformers` SAM
        # attention. The same inputs and tables went in.
        expected = [
            [
                [1.1915, 1.6540, -1.4263, -4.0603],
                [2.1546, 3.5253, 0.3769, -2.0417],
                [2.6731, -2.3215, -0.7673, 0.6695],
                [2.7452, 1.7091, -0.5453, -2.4414],
            ],
            [
                [2.4937, 9.1975, 3.3479, -5.3344],
                [1.0745, 2.4979, -1.0230, 1.9574],
                [-2.2961, 0.7833, -1.3311, -2.4978],
                [1.2388, 0.0737, -0.3316, -1.7643],
            ],
        ]
        self.assertAllClose(
            output[:, ::23, ::23], expected, atol=1e-3, rtol=1e-4
        )

    def test_bias_equals_the_terms_added_to_a_zero_map(self):
        layer = AddRelativePositionalEmbedding(input_size=(16, 16), key_dim=4)
        rng = np.random.RandomState(0)
        height, width = 10, 7
        layer.rel_pos_h.assign(rng.normal(size=(31, 4)).astype("float32"))
        layer.rel_pos_w.assign(rng.normal(size=(31, 4)).astype("float32"))
        queries = rng.normal(size=(2, height * width, 4)).astype("float32")
        kwargs = {
            "queries": queries,
            "query_size": (height, width),
            "key_size": (height, width),
        }
        zeros = np.zeros((2, height * width, height * width), "float32")
        self.assertAllClose(
            layer(None, **kwargs), layer(zeros, **kwargs), atol=1e-6
        )


class WindowedTransformerEncoderTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "project_dim": 8,
            "intermediate_dim": 16,
            "num_heads": 2,
            "use_rel_pos": True,
        }

    def test_global_layer(self):
        self.run_layer_test(
            cls=WindowedTransformerEncoder,
            init_kwargs={**self.init_kwargs, "input_size": (6, 6)},
            input_data=ops.convert_to_tensor(
                np.random.uniform(size=(2, 6, 6, 8)).astype("float32")
            ),
            expected_output_shape=(2, 6, 6, 8),
            # Two norms (4), attention (qkv and projection: 4) with two
            # relative position tables (2), and the MLP (4).
            expected_num_trainable_weights=14,
        )

    def test_windowed_layer(self):
        self.run_layer_test(
            cls=WindowedTransformerEncoder,
            init_kwargs={
                **self.init_kwargs,
                "window_size": 3,
                "input_size": (6, 6),
            },
            input_data=ops.convert_to_tensor(
                np.random.uniform(size=(2, 6, 6, 8)).astype("float32")
            ),
            expected_output_shape=(2, 6, 6, 8),
            expected_num_trainable_weights=14,
        )

    def test_global_layer_runs_on_a_smaller_grid(self):
        # The relative position tables have 11 rows. A 4x4 grid needs 7.
        layer = WindowedTransformerEncoder(
            **self.init_kwargs, input_size=(6, 6)
        )
        output = layer(np.random.uniform(size=(2, 4, 4, 8)).astype("float32"))
        self.assertEqual(tuple(output.shape), (2, 4, 4, 8))
