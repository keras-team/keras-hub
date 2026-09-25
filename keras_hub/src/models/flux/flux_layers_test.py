import keras
import numpy as np
from keras import ops

from keras_hub.src.models.flux.flux_layers import ApproximateGELU
from keras_hub.src.models.flux.flux_layers import DoubleStreamBlock
from keras_hub.src.models.flux.flux_layers import EmbedND
from keras_hub.src.models.flux.flux_layers import MLPEmbedder
from keras_hub.src.models.flux.flux_layers import Modulation
from keras_hub.src.models.flux.flux_layers import QKNorm
from keras_hub.src.models.flux.flux_layers import SingleStreamBlock
from keras_hub.src.models.flux.flux_maths import TimestepEmbedding
from keras_hub.src.models.flux.flux_maths import rearrange_symbolic_tensors
from keras_hub.src.tests.test_case import TestCase

# Small but non-degenerate config. `sum(AXES_DIM)` must equal the head
# dimension (HIDDEN_SIZE // NUM_HEADS) for RoPE to line up.
HIDDEN_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
AXES_DIM = [4, 6, 6]
BATCH = 2
IMG_LEN = 5
TXT_LEN = 3


def _positional_encoding(txt_len=TXT_LEN, img_len=IMG_LEN, seed=0):
    """Build a RoPE encoding the way `FluxBackbone` does: text ids first."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 16, size=(BATCH, txt_len + img_len, 3))
    embedder = EmbedND(theta=10_000, axes_dim=AXES_DIM)
    return embedder(ops.cast(ids, "float32"))


def _block_inputs(seed=0):
    rng = np.random.default_rng(seed)
    image = ops.cast(rng.normal(size=(BATCH, IMG_LEN, HIDDEN_SIZE)), "float32")
    text = ops.cast(rng.normal(size=(BATCH, TXT_LEN, HIDDEN_SIZE)), "float32")
    modulation = ops.cast(rng.normal(size=(BATCH, HIDDEN_SIZE)), "float32")
    return image, text, modulation, _positional_encoding(seed=seed)


class TimestepEmbeddingTest(TestCase):
    def test_matches_reference_formula(self):
        """Pins `time_factor` scaling and the `[cos, sin]` feature order.

        Both are load-bearing: the pretrained `time_in` weights are ordered
        to match, so a flip or a missing scale is silently wrong rather than
        an error.
        """
        dim = 8
        t = np.array([0.0, 0.25, 1.0], dtype="float32")

        half = dim // 2
        scaled = t * 1000.0
        freqs = np.exp(
            -np.log(10000.0) * np.arange(half, dtype="float32") / half
        )
        args = scaled[:, None] * freqs[None]
        expected = np.concatenate([np.cos(args), np.sin(args)], axis=-1)

        actual = TimestepEmbedding()(ops.convert_to_tensor(t), dim=dim)
        self.assertAllClose(actual, expected, atol=1e-5)

    def test_cos_occupies_leading_half(self):
        dim = 8
        half = dim // 2
        t = ops.convert_to_tensor(np.array([0.3], dtype="float32"))

        out = ops.convert_to_numpy(TimestepEmbedding()(t, dim=dim))

        freqs = np.exp(
            -np.log(10000.0) * np.arange(half, dtype="float32") / half
        )
        args = (0.3 * 1000.0) * freqs
        self.assertAllClose(out[0, :half], np.cos(args), atol=1e-5)
        self.assertAllClose(out[0, half:], np.sin(args), atol=1e-5)

    def test_time_factor_is_applied(self):
        """`t` and `t * time_factor` must not produce the same embedding."""
        layer = TimestepEmbedding()
        t = ops.convert_to_tensor(np.array([0.5], dtype="float32"))

        default = ops.convert_to_numpy(layer(t, dim=8))
        unscaled = ops.convert_to_numpy(layer(t, dim=8, time_factor=1.0))

        self.assertNotAllClose(default, unscaled)

    def test_odd_dim_is_zero_padded(self):
        t = ops.convert_to_tensor(np.array([0.5], dtype="float32"))
        out = ops.convert_to_numpy(TimestepEmbedding()(t, dim=7))
        self.assertEqual(out.shape, (1, 7))
        self.assertAllClose(out[:, -1], np.zeros((1,)), atol=1e-6)


class RearrangeTest(TestCase):
    def test_matches_einops_semantics(self):
        """`B L (K H D) -> K B H L D`, computed explicitly with numpy."""
        rng = np.random.default_rng(0)
        k, h, d = 3, NUM_HEADS, HEAD_DIM
        qkv = rng.normal(size=(BATCH, IMG_LEN, k * h * d)).astype("float32")

        expected = qkv.reshape(BATCH, IMG_LEN, k, h, d).transpose(2, 0, 3, 1, 4)

        q, key, v = rearrange_symbolic_tensors(ops.convert_to_tensor(qkv), k, h)
        self.assertAllClose(q, expected[0], atol=1e-6)
        self.assertAllClose(key, expected[1], atol=1e-6)
        self.assertAllClose(v, expected[2], atol=1e-6)

    def test_output_shape(self):
        qkv = ops.ones((BATCH, IMG_LEN, 3 * NUM_HEADS * HEAD_DIM))
        q, k, v = rearrange_symbolic_tensors(qkv, 3, NUM_HEADS)
        for tensor in (q, k, v):
            self.assertEqual(
                tuple(tensor.shape), (BATCH, NUM_HEADS, IMG_LEN, HEAD_DIM)
            )


class QKNormTest(TestCase):
    def test_normalizes_query_and_key_independently(self):
        rng = np.random.default_rng(0)
        q = ops.cast(
            rng.normal(size=(BATCH, NUM_HEADS, 4, HEAD_DIM)), "float32"
        )
        k = ops.cast(
            rng.normal(size=(BATCH, NUM_HEADS, 4, HEAD_DIM)), "float32"
        )

        layer = QKNorm(HEAD_DIM)
        out_q, out_k = layer(q, k)

        self.assertEqual(tuple(out_q.shape), tuple(q.shape))
        self.assertEqual(tuple(out_k.shape), tuple(k.shape))
        # With unit scale this is pure RMS normalization, so the RMS of the
        # output along the feature axis should be ~1.
        rms = np.sqrt(np.mean(np.square(ops.convert_to_numpy(out_q)), axis=-1))
        self.assertAllClose(rms, np.ones_like(rms), atol=1e-3)


class DoubleStreamBlockTest(TestCase):
    def _build_block(self):
        block = DoubleStreamBlock(
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            mlp_ratio=2.0,
            use_bias=True,
        )
        image, text, modulation, pos = _block_inputs()
        block(
            image=image,
            text=text,
            modulation_encoding=modulation,
            positional_encoding=pos,
        )
        return block

    def test_output_shapes(self):
        block = self._build_block()
        image, text, modulation, pos = _block_inputs(seed=1)

        out_image, out_text = block(
            image=image,
            text=text,
            modulation_encoding=modulation,
            positional_encoding=pos,
        )

        self.assertEqual(tuple(out_image.shape), (BATCH, IMG_LEN, HIDDEN_SIZE))
        self.assertEqual(tuple(out_text.shape), (BATCH, TXT_LEN, HIDDEN_SIZE))

    def test_qk_norm_sublayers_exist(self):
        """Guards the checkpoint mapping.

        `double_blocks.{i}.img_attn.norm.*` / `txt_attn.norm.*` exist in the
        FLUX checkpoint. If these sublayers are removed the converter has
        nowhere to put those tensors and silently drops them.
        """
        block = self._build_block()
        self.assertIsInstance(block.image_attn_norm, QKNorm)
        self.assertIsInstance(block.text_attn_norm, QKNorm)

    def test_qk_norm_affects_output(self):
        """The norm must actually be wired into `call`, not just present."""
        block = self._build_block()
        image, text, modulation, pos = _block_inputs(seed=2)

        before, _ = block(
            image=image,
            text=text,
            modulation_encoding=modulation,
            positional_encoding=pos,
        )

        block.image_attn_norm.query_norm.scale.assign(
            ops.ones((HEAD_DIM,)) * 3.0
        )

        after, _ = block(
            image=image,
            text=text,
            modulation_encoding=modulation,
            positional_encoding=pos,
        )

        self.assertNotAllClose(before, after)

    def _reference_forward(self, block, image, text, modulation, pos, order):
        """Recompute the block with an explicit q/k/v concatenation order.

        `order` is "text_first" (correct, matches the reference
        implementation) or "image_first" (the regression).
        """
        img_mod1, img_mod2 = block.image_mod(modulation)
        txt_mod1, txt_mod2 = block.text_mod(modulation)

        img_n = (
            block.image_norm1(image) * (1 + img_mod1["scale"])
            + img_mod1["shift"]
        )
        txt_n = (
            block.text_norm1(text) * (1 + txt_mod1["scale"]) + txt_mod1["shift"]
        )

        img_q, img_k, img_v = rearrange_symbolic_tensors(
            block.image_qkv(img_n), 3, NUM_HEADS
        )
        txt_q, txt_k, txt_v = rearrange_symbolic_tensors(
            block.text_qkv(txt_n), 3, NUM_HEADS
        )
        img_q, img_k = block.image_attn_norm(img_q, img_k)
        txt_q, txt_k = block.text_attn_norm(txt_q, txt_k)

        if order == "text_first":
            parts = ((txt_q, img_q), (txt_k, img_k), (txt_v, img_v))
        else:
            parts = ((img_q, txt_q), (img_k, txt_k), (img_v, txt_v))
        q, k, v = (ops.concatenate(list(p), axis=2) for p in parts)

        attn = block.attention(q, k, v, pos)
        if order == "text_first":
            txt_attn = attn[:, :TXT_LEN, :]
            img_attn = attn[:, TXT_LEN:, :]
        else:
            img_attn = attn[:, :IMG_LEN, :]
            txt_attn = attn[:, IMG_LEN:, :]

        image = image + img_mod1["gate"] * block.image_attn_proj(img_attn)
        text = text + txt_mod1["gate"] * block.text_attn_proj(txt_attn)

        image = image + img_mod2["gate"] * block.image_mlp(
            block.image_norm2(image) * (1 + img_mod2["scale"])
            + img_mod2["shift"]
        )
        text = text + txt_mod2["gate"] * block.text_mlp(
            block.text_norm2(text) * (1 + txt_mod2["scale"]) + txt_mod2["shift"]
        )
        return image, text

    def test_attention_sequence_is_text_first(self):
        """`FluxBackbone` builds RoPE ids as `concat([text_ids, image_ids])`.

        The q/k/v sequence assembled inside the block must use the same
        order, otherwise every token is rotated by another token's position.
        Shapes are identical either way, so nothing else catches this.
        """
        block = self._build_block()
        image, text, modulation, pos = _block_inputs(seed=3)

        actual_image, actual_text = block(
            image=image,
            text=text,
            modulation_encoding=modulation,
            positional_encoding=pos,
        )

        expected_image, expected_text = self._reference_forward(
            block, image, text, modulation, pos, order="text_first"
        )
        self.assertAllClose(actual_image, expected_image, atol=1e-5)
        self.assertAllClose(actual_text, expected_text, atol=1e-5)

        # The two orderings must actually be distinguishable, otherwise the
        # assertions above would hold no matter what the block does.
        wrong_image, _ = self._reference_forward(
            block, image, text, modulation, pos, order="image_first"
        )
        self.assertNotAllClose(actual_image, wrong_image)

    def test_config_round_trip(self):
        block = DoubleStreamBlock(
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            mlp_ratio=2.0,
            use_bias=True,
        )
        config = block.get_config()
        restored = DoubleStreamBlock.from_config(config)

        self.assertEqual(restored.hidden_size, HIDDEN_SIZE)
        self.assertEqual(restored.num_heads, NUM_HEADS)
        self.assertEqual(restored.mlp_ratio, 2.0)
        self.assertTrue(restored.use_bias)

    def test_layer_norms_are_not_affine(self):
        """FLUX uses `elementwise_affine=False` for the stream norms."""
        block = self._build_block()
        for norm in (
            block.image_norm1,
            block.image_norm2,
            block.text_norm1,
            block.text_norm2,
        ):
            self.assertFalse(norm.scale)
            self.assertFalse(norm.center)


class SingleStreamBlockTest(TestCase):
    def test_output_shape_and_qk_norm(self):
        seq_len = TXT_LEN + IMG_LEN
        rng = np.random.default_rng(0)
        x = ops.cast(rng.normal(size=(BATCH, seq_len, HIDDEN_SIZE)), "float32")
        modulation = ops.cast(rng.normal(size=(BATCH, HIDDEN_SIZE)), "float32")
        pos = _positional_encoding()

        block = SingleStreamBlock(
            hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, mlp_ratio=2.0
        )
        out = block(x, modulation_encoding=modulation, positional_encoding=pos)

        self.assertEqual(tuple(out.shape), (BATCH, seq_len, HIDDEN_SIZE))
        # `single_blocks.{i}.norm.*` exists in the checkpoint.
        self.assertIsInstance(block.norm, QKNorm)

    def test_pre_norm_is_not_affine(self):
        block = SingleStreamBlock(
            hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, mlp_ratio=2.0
        )
        self.assertFalse(block.pre_norm.scale)
        self.assertFalse(block.pre_norm.center)


class SupportLayersTest(TestCase):
    def test_modulation_double_returns_two_sets(self):
        layer = Modulation(HIDDEN_SIZE, double=True)
        x = ops.ones((BATCH, HIDDEN_SIZE))
        first, second = layer(x)

        self.assertIsNotNone(second)
        for part in (first, second):
            for key in ("shift", "scale", "gate"):
                self.assertEqual(
                    tuple(part[key].shape), (BATCH, 1, HIDDEN_SIZE)
                )

    def test_modulation_single_returns_none(self):
        layer = Modulation(HIDDEN_SIZE, double=False)
        _, second = layer(ops.ones((BATCH, HIDDEN_SIZE)))
        self.assertIsNone(second)

    def test_mlp_embedder_shape(self):
        layer = MLPEmbedder(hidden_dim=HIDDEN_SIZE)
        out = layer(ops.ones((BATCH, 16)))
        self.assertEqual(tuple(out.shape), (BATCH, HIDDEN_SIZE))

    def test_embed_nd_shape(self):
        ids = ops.ones((BATCH, IMG_LEN, 3))
        out = EmbedND(theta=10_000, axes_dim=AXES_DIM)(ids)
        self.assertEqual(
            tuple(out.shape), (BATCH, IMG_LEN, sum(AXES_DIM) // 2, 2)
        )

    def test_embed_nd_rejects_axis_mismatch(self):
        """`axes_dim` must have one entry per positional axis."""
        embedder = EmbedND(theta=10_000, axes_dim=AXES_DIM)
        with self.assertRaisesRegex(ValueError, "positional axes"):
            embedder.build((BATCH, IMG_LEN, 4))


class DTypeTest(TestCase):
    def test_timestep_embedding_casts_low_precision_input(self):
        """`t` arrives as bfloat16 under a mixed policy; output must be f32."""
        t = ops.cast(ops.convert_to_tensor([0.5, 0.75]), "bfloat16")
        out = TimestepEmbedding()(t, dim=8)
        self.assertEqual(keras.backend.standardize_dtype(out.dtype), "float32")


class ActivationTest(TestCase):
    """FLUX's double-stream MLPs use the tanh approximation of GELU.

    The reference is `nn.GELU(approximate="tanh")` (`"gelu-approximate"` in
    diffusers). Keras's `Activation("gelu")` is the exact, erf-based
    formulation -- a genuinely different function. The discrepancy is small
    per-layer, which is exactly why it survived review: it does not change
    shapes and looks plausible in isolation.
    """

    def test_approximate_gelu_matches_tanh_formulation(self):
        x = np.linspace(-4.0, 4.0, 128).astype("float32")
        self.assertAllClose(
            ApproximateGELU()(x),
            keras.activations.gelu(x, approximate=True),
        )

    def test_approximate_and_exact_gelu_actually_differ(self):
        # Guards the test above from passing vacuously: if the two
        # formulations were interchangeable there would be nothing to fix.
        x = np.linspace(-4.0, 4.0, 128).astype("float32")
        gap = np.abs(
            ops.convert_to_numpy(keras.activations.gelu(x, approximate=True))
            - ops.convert_to_numpy(keras.activations.gelu(x, approximate=False))
        ).max()
        self.assertGreater(gap, 1e-4)

    def test_double_block_mlps_use_approximate_gelu(self):
        block = DoubleStreamBlock(
            hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, mlp_ratio=2.0
        )
        self.assertIsInstance(block.image_mlp.layers[1], ApproximateGELU)
        self.assertIsInstance(block.text_mlp.layers[1], ApproximateGELU)
