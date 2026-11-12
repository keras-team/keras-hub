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
