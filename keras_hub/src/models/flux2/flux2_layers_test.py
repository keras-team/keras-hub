import numpy as np
from keras import ops

from keras_hub.src.models.flux.flux_layers import EmbedND
from keras_hub.src.models.flux2.flux2_layers import Flux2DoubleStreamBlock
from keras_hub.src.models.flux2.flux2_layers import Flux2FeedForward
from keras_hub.src.models.flux2.flux2_layers import Flux2LastLayer
from keras_hub.src.models.flux2.flux2_layers import Flux2MLPEmbedder
from keras_hub.src.models.flux2.flux2_layers import Flux2Modulation
from keras_hub.src.models.flux2.flux2_layers import Flux2SingleStreamBlock
from keras_hub.src.models.flux2.flux2_layers import Flux2SwiGLU
from keras_hub.src.models.flux2.flux2_layers import split_modulation
from keras_hub.src.tests.test_case import TestCase

# `sum(AXES_DIM)` must equal the head dimension for RoPE to line up.
HIDDEN_SIZE = 32
NUM_HEADS = 2
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
AXES_DIM = [4, 4, 4, 4]
MLP_RATIO = 3.0
MLP_HIDDEN = int(HIDDEN_SIZE * MLP_RATIO)
BATCH = 2
IMG_LEN = 5
TXT_LEN = 3


def _random(*shape, seed=0):
    rng = np.random.default_rng(seed)
    return ops.convert_to_tensor(rng.normal(size=shape).astype("float32"))


def _positional_encoding(length=TXT_LEN + IMG_LEN, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 16, size=(BATCH, length, len(AXES_DIM)))
    return EmbedND(theta=2000, axes_dim=AXES_DIM)(ops.cast(ids, "float32"))


class Flux2SwiGLUTest(TestCase):
    def test_gate_is_the_first_half(self):
        """`silu(x[..., :d]) * x[..., d:]`; swapping halves is silent."""
        x = np.random.default_rng(0).normal(size=(2, 3, 8)).astype("float32")
        gate, value = x[..., :4], x[..., 4:]
        expected = gate / (1.0 + np.exp(-gate)) * value

        actual = Flux2SwiGLU()(ops.convert_to_tensor(x))
        self.assertAllClose(actual, expected, atol=1e-6)
        self.assertEqual(tuple(actual.shape), (2, 3, 4))


class Flux2FeedForwardTest(TestCase):
    def test_matches_manual_computation(self):
        layer = Flux2FeedForward(HIDDEN_SIZE, MLP_HIDDEN)
        x = _random(BATCH, IMG_LEN, HIDDEN_SIZE)
        actual = layer(x)

        hidden = ops.matmul(x, layer.input_layer.kernel)
        gate, value = ops.split(hidden, 2, axis=-1)
        expected = ops.matmul(ops.silu(gate) * value, layer.output_layer.kernel)
        self.assertAllClose(actual, expected, atol=1e-5)
        self.assertEqual(
            tuple(layer.input_layer.kernel.shape),
            (HIDDEN_SIZE, 2 * MLP_HIDDEN),
        )

    def test_serialization(self):
        self.run_serialization_test(Flux2FeedForward(HIDDEN_SIZE, MLP_HIDDEN))


class Flux2MLPEmbedderTest(TestCase):
    def test_output_shape_and_no_bias(self):
        layer = Flux2MLPEmbedder(HIDDEN_SIZE)
        output = layer(_random(BATCH, 256))
        self.assertEqual(tuple(output.shape), (BATCH, HIDDEN_SIZE))
        self.assertEqual(len(layer.weights), 2)

    def test_serialization(self):
        self.run_serialization_test(Flux2MLPEmbedder(HIDDEN_SIZE))


class Flux2ModulationTest(TestCase):
    def test_double_and_single_widths(self):
        vec = _random(BATCH, HIDDEN_SIZE)
        double = Flux2Modulation(HIDDEN_SIZE, double=True)(vec)
        single = Flux2Modulation(HIDDEN_SIZE, double=False)(vec)
        self.assertEqual(tuple(double.shape), (BATCH, 6 * HIDDEN_SIZE))
        self.assertEqual(tuple(single.shape), (BATCH, 3 * HIDDEN_SIZE))

    def test_applies_silu_before_projection(self):
        layer = Flux2Modulation(HIDDEN_SIZE, double=False)
        vec = _random(BATCH, HIDDEN_SIZE)
        actual = layer(vec)
        expected = ops.matmul(ops.silu(vec), layer.linear_projection.kernel)
        self.assertAllClose(actual, expected, atol=1e-5)

    def test_split_order(self):
        """Chunks come out in `shift, scale, gate` order, with a token axis."""
        modulation = ops.reshape(
            ops.arange(BATCH * 3 * HIDDEN_SIZE, dtype="float32"),
            (BATCH, 3 * HIDDEN_SIZE),
        )
        shift, scale, gate = split_modulation(modulation, 3)
        self.assertEqual(tuple(shift.shape), (BATCH, 1, HIDDEN_SIZE))
        self.assertAllClose(shift[:, 0], modulation[:, :HIDDEN_SIZE])
        self.assertAllClose(
            scale[:, 0], modulation[:, HIDDEN_SIZE : 2 * HIDDEN_SIZE]
        )
        self.assertAllClose(gate[:, 0], modulation[:, 2 * HIDDEN_SIZE :])

    def test_serialization(self):
        self.run_serialization_test(Flux2Modulation(HIDDEN_SIZE, double=True))


class Flux2DoubleStreamBlockTest(TestCase):
    def _inputs(self):
        return {
            "image": _random(BATCH, IMG_LEN, HIDDEN_SIZE, seed=1),
            "text": _random(BATCH, TXT_LEN, HIDDEN_SIZE, seed=2),
            "image_modulation": _random(BATCH, 6 * HIDDEN_SIZE, seed=3),
            "text_modulation": _random(BATCH, 6 * HIDDEN_SIZE, seed=4),
            "positional_encoding": _positional_encoding(),
        }

    def test_output_shapes(self):
        block = Flux2DoubleStreamBlock(HIDDEN_SIZE, NUM_HEADS, MLP_RATIO)
        image, text = block(**self._inputs())
        self.assertEqual(tuple(image.shape), (BATCH, IMG_LEN, HIDDEN_SIZE))
        self.assertEqual(tuple(text.shape), (BATCH, TXT_LEN, HIDDEN_SIZE))

    def test_zero_gates_are_identity(self):
        """With every gate at zero, both residual streams pass through."""
        inputs = self._inputs()
        zero_gates = np.ones((6,), dtype="float32")
        zero_gates[[2, 5]] = 0.0  # gate1 and gate2.
        mask = ops.convert_to_tensor(np.repeat(zero_gates, HIDDEN_SIZE)[None])
        inputs["image_modulation"] = inputs["image_modulation"] * mask
        inputs["text_modulation"] = inputs["text_modulation"] * mask

        block = Flux2DoubleStreamBlock(HIDDEN_SIZE, NUM_HEADS, MLP_RATIO)
        image, text = block(**inputs)
        self.assertAllClose(image, inputs["image"], atol=1e-6)
        self.assertAllClose(text, inputs["text"], atol=1e-6)

    def test_has_no_own_modulation_weights(self):
        block = Flux2DoubleStreamBlock(HIDDEN_SIZE, NUM_HEADS, MLP_RATIO)
        block(**self._inputs())
        self.assertFalse(any("mod" in w.path for w in block.weights))
        self.assertFalse(any("bias" in w.path for w in block.weights))

    def test_serialization(self):
        self.run_serialization_test(
            Flux2DoubleStreamBlock(HIDDEN_SIZE, NUM_HEADS, MLP_RATIO)
        )


class Flux2SingleStreamBlockTest(TestCase):
    def _inputs(self):
        return {
            "x": _random(BATCH, TXT_LEN + IMG_LEN, HIDDEN_SIZE, seed=1),
            "modulation": _random(BATCH, 3 * HIDDEN_SIZE, seed=2),
            "positional_encoding": _positional_encoding(),
        }

    def test_output_shape_and_fused_widths(self):
        block = Flux2SingleStreamBlock(HIDDEN_SIZE, NUM_HEADS, MLP_RATIO)
        output = block(**self._inputs())
        self.assertEqual(
            tuple(output.shape), (BATCH, TXT_LEN + IMG_LEN, HIDDEN_SIZE)
        )
        # [q, k, v, mlp_gate, mlp_up] and [attention, mlp] -> hidden.
        self.assertEqual(
            tuple(block.linear1.kernel.shape),
            (HIDDEN_SIZE, 3 * HIDDEN_SIZE + 2 * MLP_HIDDEN),
        )
        self.assertEqual(
            tuple(block.linear2.kernel.shape),
            (HIDDEN_SIZE + MLP_HIDDEN, HIDDEN_SIZE),
        )

    def test_zero_gate_is_identity(self):
        inputs = self._inputs()
        mask = np.repeat(np.array([1.0, 1.0, 0.0], "float32"), HIDDEN_SIZE)
        mask = ops.convert_to_tensor(mask[None])
        inputs["modulation"] = inputs["modulation"] * mask
        block = Flux2SingleStreamBlock(HIDDEN_SIZE, NUM_HEADS, MLP_RATIO)
        self.assertAllClose(block(**inputs), inputs["x"], atol=1e-6)

    def test_serialization(self):
        self.run_serialization_test(
            Flux2SingleStreamBlock(HIDDEN_SIZE, NUM_HEADS, MLP_RATIO)
        )


class Flux2LastLayerTest(TestCase):
    def test_shift_scale_order(self):
        """The first half of `adaLN_modulation` is the shift."""
        layer = Flux2LastLayer(HIDDEN_SIZE, 8)
        x = _random(BATCH, IMG_LEN, HIDDEN_SIZE, seed=1)
        vec = _random(BATCH, HIDDEN_SIZE, seed=2)
        actual = layer(x, vec)

        modulation = ops.matmul(ops.silu(vec), layer.adaLN_modulation.kernel)
        shift, scale = ops.split(modulation, 2, axis=-1)
        mean = ops.mean(x, axis=-1, keepdims=True)
        var = ops.var(x, axis=-1, keepdims=True)
        normed = (x - mean) / ops.sqrt(var + 1e-6)
        expected = ops.matmul(
            (1 + scale[:, None]) * normed + shift[:, None],
            layer.linear.kernel,
        )
        self.assertAllClose(actual, expected, atol=1e-4)
        self.assertEqual(tuple(actual.shape), (BATCH, IMG_LEN, 8))

    def test_serialization(self):
        self.run_serialization_test(Flux2LastLayer(HIDDEN_SIZE, 8))
