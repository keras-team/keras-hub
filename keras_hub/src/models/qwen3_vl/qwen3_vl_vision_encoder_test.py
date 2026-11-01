from unittest.mock import patch

import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_vl import qwen3_vl_vision_encoder
from keras_hub.src.models.qwen3_vl.qwen3_vl_vision_encoder import (
    Qwen3_VLVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu

PATCH = 4
TEMPORAL = 2
CHANNELS = 3
HIDDEN = 32
OUT_HIDDEN = 24


def _atol():
    # Batched and single-sample passes dispatch different GPU kernels, which
    # differ by up to ~1e-4 in float32.
    return 5e-4 if running_on_gpu() else 1e-5


try:
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionModel,
    )
except ImportError:
    torch = None


def _init_kwargs(**overrides):
    kwargs = dict(
        depth=4,
        hidden_size=HIDDEN,
        num_heads=2,
        intermediate_size=48,
        out_hidden_size=OUT_HIDDEN,
        patch_size=PATCH,
        temporal_patch_size=TEMPORAL,
        in_channels=CHANNELS,
        spatial_merge_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=(1, 2),
    )
    kwargs.update(overrides)
    return kwargs


def _run(encoder, pixels, grid):
    features, *deepstack = encoder(
        {"pixel_values": pixels, "image_grid_thw": grid}
    )
    return features, deepstack


def _pixels(batch, num_patches, seed=0):
    rng = np.random.RandomState(seed)
    return rng.rand(
        batch, num_patches, TEMPORAL, PATCH, PATCH, CHANNELS
    ).astype("float32")


class Qwen3VLVisionEncoderTest(TestCase):
    def test_output_shapes(self):
        encoder = Qwen3_VLVisionEncoder(**_init_kwargs())
        grid = np.array([[[1, 4, 4], [2, 4, 6]]], dtype="int32")
        num_patches = 16 + 48
        features, deepstack = _run(encoder, _pixels(1, num_patches), grid)
        self.assertEqual(features.shape, (1, num_patches // 4, OUT_HIDDEN))
        self.assertEqual(len(deepstack), 2)
        for feature in deepstack:
            self.assertEqual(feature.shape, features.shape)
        self.assertTrue(np.all(np.isfinite(ops.convert_to_numpy(features))))

    def test_batched_inputs(self):
        encoder = Qwen3_VLVisionEncoder(**_init_kwargs())
        grid = np.array([[[1, 4, 4]], [[1, 2, 8]]], dtype="int32")
        features, deepstack = _run(encoder, _pixels(2, 16), grid)
        self.assertEqual(features.shape, (2, 4, OUT_HIDDEN))
        self.assertEqual(len(deepstack), 2)

    def test_batch_matches_single_samples(self):
        encoder = Qwen3_VLVisionEncoder(**_init_kwargs())
        grid = np.array([[[1, 4, 4]], [[1, 2, 8]]], dtype="int32")
        pixels = _pixels(2, 16)
        batched, _ = _run(encoder, pixels, grid)
        for i in range(2):
            single, _ = _run(encoder, pixels[i : i + 1], grid[i : i + 1])
            self.assertAllClose(batched[i], single[0], atol=_atol())

    def test_padding_items_do_not_change_real_tokens(self):
        encoder = Qwen3_VLVisionEncoder(**_init_kwargs())
        pixels = _pixels(1, 16)
        grid = np.array([[[1, 4, 4]]], dtype="int32")
        padded_grid = np.array([[[1, 4, 4], [0, 0, 0]]], dtype="int32")
        padded_pixels = np.concatenate([pixels, _pixels(1, 8, seed=1)], axis=1)
        expected, _ = _run(encoder, pixels, grid)
        actual, _ = _run(encoder, padded_pixels, padded_grid)
        self.assertAllClose(actual[:, :4], expected, atol=_atol())

    def test_frames_do_not_attend_across_each_other(self):
        encoder = Qwen3_VLVisionEncoder(**_init_kwargs())
        grid = np.array([[[2, 4, 4]]], dtype="int32")
        pixels = _pixels(1, 32)
        base, _ = _run(encoder, pixels, grid)
        changed = pixels.copy()
        changed[:, 16:] += 1.0
        other, _ = _run(encoder, changed, grid)
        self.assertAllClose(base[:, :4], other[:, :4], atol=_atol())
        self.assertNotAllClose(base[:, 4:], other[:, 4:])

    def test_attention_output_spec(self):
        attention = qwen3_vl_vision_encoder.Qwen3_VLVisionAttention(
            hidden_size=HIDDEN, num_heads=2
        )
        attention.build((None, None, HIDDEN))
        x = keras.KerasTensor((2, 7, HIDDEN))
        cos = sin = keras.KerasTensor((2, 7, HIDDEN // 2))
        frame_id = keras.KerasTensor((2, 7), dtype="int32")
        spec = attention.compute_output_spec(x, cos, sin, frame_id)
        self.assertEqual(spec.shape, (2, 7, HIDDEN))
        # The symbolic call agrees with the eager output.
        eager = attention(
            np.ones((2, 7, HIDDEN), "float32"),
            np.ones((2, 7, HIDDEN // 2), "float32"),
            np.zeros((2, 7, HIDDEN // 2), "float32"),
            np.zeros((2, 7), "int32"),
        )
        self.assertEqual(tuple(eager.shape), spec.shape)

    def test_query_chunks_match_single_pass(self):
        encoder = Qwen3_VLVisionEncoder(**_init_kwargs())
        grid = np.array([[[1, 4, 4], [2, 4, 6]]], dtype="int32")
        pixels = _pixels(1, 16 + 48)
        expected, expected_ds = _run(encoder, pixels, grid)
        with patch.object(qwen3_vl_vision_encoder, "_QUERY_CHUNK", 10):
            actual, actual_ds = _run(encoder, pixels, grid)
        self.assertAllClose(actual, expected, atol=_atol())
        self.assertAllClose(actual_ds[0], expected_ds[0], atol=_atol())

    def test_config_arguments_change_behavior(self):
        grid = np.array([[[1, 4, 4]]], dtype="int32")
        pixels = _pixels(1, 16)
        base = Qwen3_VLVisionEncoder(**_init_kwargs())
        base_out, base_ds = _run(base, pixels, grid)
        other = Qwen3_VLVisionEncoder(
            **_init_kwargs(deepstack_visual_indexes=(0, 1, 3))
        )
        self.assertEqual(len(_run(other, pixels, grid)[1]), 3)
        self.assertEqual(len(base_ds), 2)
        self.assertEqual(base_out.shape, (1, 4, OUT_HIDDEN))

    def test_config_round_trip(self):
        encoder = Qwen3_VLVisionEncoder(
            **_init_kwargs(layer_norm_epsilon=1e-5, in_channels=1)
        )
        self.run_serialization_test(encoder)
        config = encoder.get_config()
        restored = Qwen3_VLVisionEncoder.from_config(config)
        self.assertEqual(config["deepstack_visual_indexes"], [1, 2])
        self.assertEqual(restored.in_channels, 1)
        named = Qwen3_VLVisionEncoder.from_config(
            Qwen3_VLVisionEncoder(
                **_init_kwargs(), name="vision", trainable=False
            ).get_config()
        )
        self.assertEqual(named.name, "vision")
        self.assertFalse(named.trainable)
        self.assertEqual(restored.layer_norm_epsilon, 1e-5)

    def test_invalid_position_table(self):
        with self.assertRaises(ValueError):
            Qwen3_VLVisionEncoder(**_init_kwargs(num_position_embeddings=15))

    def test_invalid_deepstack_indexes(self):
        with self.assertRaises(ValueError):
            Qwen3_VLVisionEncoder(
                **_init_kwargs(deepstack_visual_indexes=(0, 99))
            )

    def test_functional_model(self):
        encoder = Qwen3_VLVisionEncoder(**_init_kwargs())
        self.assertIsInstance(encoder, keras.Model)
        self.assertEqual(
            sorted(i.name for i in encoder.inputs),
            ["image_grid_thw", "pixel_values"],
        )
        # One output for the features and one per DeepStack index.
        self.assertEqual(len(encoder.outputs), 3)
        self.assertGreater(encoder.count_params(), 0)
        lines = []
        encoder.summary(print_fn=lines.append)
        self.assertTrue(any("Total params" in line for line in lines))
        grid = np.array([[[1, 4, 4], [1, 2, 4]]], dtype="int32")
        pixels = _pixels(1, 24)
        outputs = encoder.predict(
            {"pixel_values": pixels, "image_grid_thw": grid}, verbose=0
        )
        expected, expected_ds = _run(encoder, pixels, grid)
        self.assertAllClose(outputs[0], expected, atol=_atol())
        self.assertAllClose(outputs[1], expected_ds[0], atol=_atol())

    @pytest.mark.skipif(
        torch is None, reason="Requires torch and transformers."
    )
    def test_matches_hugging_face(self):
        torch.manual_seed(0)
        hf_config = Qwen3VLVisionConfig(
            depth=4,
            hidden_size=HIDDEN,
            intermediate_size=48,
            num_heads=2,
            in_channels=CHANNELS,
            patch_size=PATCH,
            spatial_merge_size=2,
            temporal_patch_size=TEMPORAL,
            out_hidden_size=OUT_HIDDEN,
            num_position_embeddings=16,
            deepstack_visual_indexes=[1, 2],
        )
        hf_model = Qwen3VLVisionModel(hf_config).eval().float()
        # Give every parameter, including norms and biases, a nonzero value.
        with torch.no_grad():
            for param in hf_model.parameters():
                param.copy_(torch.randn_like(param) * 0.2)

        encoder = Qwen3_VLVisionEncoder(**_init_kwargs())
        pixels = _pixels(1, 16 + 48 + 8)
        grid = np.array([[[1, 4, 4], [2, 4, 6], [1, 2, 4]]], dtype="int32")
        _run(encoder, pixels, grid)
        _copy_weights(hf_model, encoder)

        features, deepstack = _run(encoder, pixels, grid)

        # HF flattens each patch as (channel, time, height, width).
        hf_pixels = np.transpose(pixels[0], (0, 4, 1, 2, 3)).reshape(
            pixels.shape[1], -1
        )
        with torch.no_grad():
            hf_out = hf_model(
                torch.from_numpy(hf_pixels),
                grid_thw=torch.from_numpy(grid[0].astype("int64")),
            )
        diffs = [
            np.abs(
                ops.convert_to_numpy(features[0]) - hf_out.pooler_output.numpy()
            ).max()
        ]
        self.assertEqual(len(hf_out.deepstack_features), len(deepstack))
        for ours, theirs in zip(deepstack, hf_out.deepstack_features):
            diffs.append(
                np.abs(ops.convert_to_numpy(ours[0]) - theirs.numpy()).max()
            )
        # HF runs on CPU in float32. On GPU the encoder drifts by up to ~7e-4
        # (TF32 matmuls and convolutions); CPU keeps 1e-4.
        tol = 2e-3 if running_on_gpu() else 1e-4
        self.assertLess(max(diffs), tol, msg=f"max abs diffs: {diffs}")


def _copy_weights(hf_model, encoder):
    """Copy a Hugging Face vision tower's weights into the Keras encoder."""
    state = {k: v.numpy() for k, v in hf_model.state_dict().items()}

    def dense(layer, prefix):
        layer.kernel.assign(state[prefix + ".weight"].T)
        layer.bias.assign(state[prefix + ".bias"])

    def norm(layer, prefix):
        layer.gamma.assign(state[prefix + ".weight"])
        layer.beta.assign(state[prefix + ".bias"])

    def merger(layer, prefix):
        norm(layer.norm, prefix + ".norm")
        dense(layer.linear_fc1, prefix + ".linear_fc1")
        dense(layer.linear_fc2, prefix + ".linear_fc2")

    encoder.patch_embed.proj.kernel.assign(
        state["patch_embed.proj.weight"].transpose(2, 3, 4, 1, 0)
    )
    encoder.patch_embed.proj.bias.assign(state["patch_embed.proj.bias"])
    encoder.pos_embed.embeddings.assign(state["pos_embed.weight"])
    for i, block in enumerate(encoder.blocks):
        prefix = f"blocks.{i}"
        norm(block.norm1, prefix + ".norm1")
        norm(block.norm2, prefix + ".norm2")
        dense(block.attn.qkv, prefix + ".attn.qkv")
        dense(block.attn.proj, prefix + ".attn.proj")
        dense(block.mlp.fc1, prefix + ".mlp.linear_fc1")
        dense(block.mlp.fc2, prefix + ".mlp.linear_fc2")
    merger(encoder.merger, "merger")
    for i, layer in enumerate(encoder.deepstack_merger_list):
        merger(layer, f"deepstack_merger_list.{i}")
