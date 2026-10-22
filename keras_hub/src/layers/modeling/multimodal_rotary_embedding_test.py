import keras
import numpy as np
from keras import ops

from keras_hub.src.layers.modeling.multimodal_rotary_embedding import (
    MultimodalRotaryEmbedding,
)
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.tests.test_case import TestCase

HEAD_DIM = 16
SECTION = (4, 2, 2)


class MultimodalRotaryEmbeddingTest(TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.x = rng.normal(size=(2, 6, 4, HEAD_DIM)).astype("float32")
        self.pos = rng.randint(0, 50, size=(2, 3, 6)).astype("int32")

    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=MultimodalRotaryEmbedding,
            init_kwargs={
                "mrope_section": SECTION,
                "max_wavelength": 1000,
                "attention_scaling": 0.5,
            },
            input_data=self.x,
            expected_output_shape=(2, 6, 4, HEAD_DIM),
        )

    def test_text_positions_match_rotary_embedding(self):
        layer = MultimodalRotaryEmbedding(SECTION, max_wavelength=1000)
        rope = RotaryEmbedding(max_wavelength=1000)
        self.assertAllClose(layer(self.x), rope(self.x), atol=1e-5)
        self.assertAllClose(
            layer(self.x, start_index=3), rope(self.x, start_index=3)
        )
        # Equal channels are the same as no `position_ids`.
        pos = np.tile(np.arange(6, dtype="int32"), (2, 3, 1))
        self.assertAllClose(layer(self.x), layer(self.x, position_ids=pos))

    def test_yarn_matches_rotary_embedding(self):
        kwargs = dict(max_wavelength=1000, rope_type="yarn", scaling_factor=4.0)
        layer = MultimodalRotaryEmbedding(SECTION, **kwargs)
        rope = RotaryEmbedding(**kwargs)
        self.assertAllClose(layer(self.x), rope(self.x), atol=1e-5)
        kwargs["attention_factor"] = 1.5
        layer = MultimodalRotaryEmbedding(SECTION, **kwargs)
        rope = RotaryEmbedding(**kwargs)
        self.assertAllClose(layer(self.x), rope(self.x), atol=1e-5)

    def test_linear_scaling_matches_rotary_embedding(self):
        kwargs = dict(max_wavelength=1000, scaling_factor=4.0)
        layer = MultimodalRotaryEmbedding(SECTION, **kwargs)
        rope = RotaryEmbedding(**kwargs)
        self.assertAllClose(layer(self.x), rope(self.x), atol=1e-5)
        self.assertNotAllClose(
            layer(self.x), MultimodalRotaryEmbedding(SECTION)(self.x)
        )

    def test_symbolic_inputs(self):
        layer = MultimodalRotaryEmbedding(SECTION)
        x = keras.Input((None, 4, HEAD_DIM))
        position_ids = keras.Input((3, None), dtype="int32")
        model = keras.Model(
            [x, position_ids], layer(x, position_ids=position_ids)
        )
        self.assertEqual(tuple(model.output.shape), (None, None, 4, HEAD_DIM))
        self.assertAllClose(
            model([self.x, self.pos]), layer(self.x, position_ids=self.pos)
        )

    def test_pinned_reference(self):
        # `head_dim=8` with `mrope_section=(2, 1, 1)` assigns the
        # frequencies to T, H, W, T. The token sits at t=1, h=2, w=3, so the
        # angles are `[1, 2 * 100**-0.25, 3 * 0.1, 100**-0.75]`.
        layer = MultimodalRotaryEmbedding((2, 1, 1), max_wavelength=100)
        x = np.arange(1, 9, dtype="float32").reshape(1, 1, 8)
        position_ids = np.array([[[1], [2], [3]]], dtype="int32")
        self.assertAllClose(
            layer(x, position_ids=position_ids)[0, 0],
            [
                -3.66705,
                -1.93361,
                0.79737,
                3.74506,
                3.54298,
                6.02172,
                7.57392,
                8.12247,
            ],
            atol=1e-4,
        )

    def test_channels_are_interleaved(self):
        layer = MultimodalRotaryEmbedding(SECTION, max_wavelength=1000)
        cos, sin = layer.compute_embeddings(self.pos)
        self.assertEqual(tuple(cos.shape), (2, 6, 1, HEAD_DIM))
        inv_freq = 1.0 / 1000 ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM)
        # T at 0, 3, 6, ...; H at 1, 4; W at 2, 5; the rest T.
        channel = [0, 1, 2, 0, 1, 2, 0, 0]
        freqs = np.stack(
            [self.pos[:, c] * inv_freq[i] for i, c in enumerate(channel)],
            axis=-1,
        )
        emb = np.concatenate([freqs, freqs], axis=-1)
        self.assertAllClose(cos[:, :, 0], np.cos(emb))
        self.assertAllClose(sin[:, :, 0], np.sin(emb))

    def test_precomputed_tables_match(self):
        layer = MultimodalRotaryEmbedding(SECTION)
        tables = layer.compute_embeddings(self.pos)
        self.assertAllClose(
            layer(self.x, position_ids=self.pos),
            layer(self.x, rotary_embeddings=tables),
        )

    def test_partial_rotary_passes_extra_features(self):
        layer = MultimodalRotaryEmbedding(SECTION)
        extra = np.ones((2, 6, 4, 8), dtype="float32")
        x = np.concatenate([self.x, extra], axis=-1)
        output = layer(x, position_ids=self.pos)
        self.assertAllClose(
            output[..., :HEAD_DIM], layer(self.x, position_ids=self.pos)
        )
        self.assertAllClose(output[..., HEAD_DIM:], extra)

    def test_three_dimensional_inputs(self):
        layer = MultimodalRotaryEmbedding(SECTION)
        x = self.x[:, :, 0]
        self.assertAllClose(
            layer(x, position_ids=self.pos),
            layer(self.x, position_ids=self.pos)[:, :, 0],
        )

    def test_five_dimensional_inputs(self):
        layer = MultimodalRotaryEmbedding(SECTION)
        x = np.stack([self.x, 2 * self.x], axis=2)
        output = layer(x, position_ids=self.pos)
        self.assertAllClose(
            output[:, :, 0], layer(self.x, position_ids=self.pos)
        )
        self.assertAllClose(
            output[:, :, 1], layer(2 * self.x, position_ids=self.pos)
        )

    def test_start_index_is_second_argument(self):
        layer = MultimodalRotaryEmbedding(SECTION)
        # A scalar `cache_update_index` passed positionally is a start index.
        start_index = ops.convert_to_tensor(3, "int32")
        self.assertAllClose(
            layer(self.x, start_index), layer(self.x, start_index=3)
        )

    def test_invalid_shapes(self):
        layer = MultimodalRotaryEmbedding(SECTION)
        with self.assertRaisesRegex(ValueError, "sequence_length, ..."):
            layer(self.x[0, :, 0])
        # `(batch, seq, 3)` instead of `(batch, 3, seq)`.
        with self.assertRaisesRegex(ValueError, "position_ids"):
            layer(self.x, position_ids=np.transpose(self.pos, (0, 2, 1)))
        with self.assertRaisesRegex(ValueError, "position_ids"):
            layer(self.x, position_ids=self.pos[:, 0])

    def test_compute_dtype(self):
        layer = MultimodalRotaryEmbedding(SECTION, dtype="bfloat16")
        x = ops.cast(self.x, "bfloat16")
        output = layer(x, position_ids=self.pos)
        self.assertEqual(
            keras.backend.standardize_dtype(output.dtype), "bfloat16"
        )

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            MultimodalRotaryEmbedding((4, 4))
        with self.assertRaises(ValueError):
            MultimodalRotaryEmbedding(SECTION, sequence_axis=2)
        with self.assertRaises(ValueError):
            MultimodalRotaryEmbedding(SECTION).compute_embeddings()
        with self.assertRaisesRegex(ValueError, "positive integers"):
            MultimodalRotaryEmbedding((0, 1, 1))
        # Interleaving gives W only one frequency out of five, not two.
        with self.assertRaisesRegex(ValueError, "interleaved"):
            MultimodalRotaryEmbedding((1, 2, 2))
        layer = MultimodalRotaryEmbedding(SECTION)
        narrow = np.ones((1, 2, layer.rotary_dim - 2), "float32")
        with self.assertRaisesRegex(ValueError, "fewer than"):
            layer(narrow)
