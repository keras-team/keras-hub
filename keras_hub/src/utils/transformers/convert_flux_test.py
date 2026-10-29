import os

import numpy as np
import pytest
from safetensors.numpy import save_file

from keras_hub.src.models.flux.flux_backbone import FluxBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_flux
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

# A deliberately tiny FLUX. `sum(AXES_DIM)` must equal hidden // heads.
INPUT_CHANNELS = 8
HIDDEN = 16
NUM_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS
MLP_RATIO = 2.0
MLP_HIDDEN = int(HIDDEN * MLP_RATIO)
AXES_DIM = [2, 3, 3]
DEPTH = 1
DEPTH_SINGLE = 1
TEXT_DIM = 12
Y_DIM = 10
TIMESTEP_DIM = 256


def _tensor(*shape):
    """Deterministic, distinguishable values (never all-ones)."""
    size = int(np.prod(shape))
    return (np.arange(size, dtype="float32") * 0.01 + 0.5).reshape(shape)


def _linear(out_features, in_features, prefix, tensors):
    """Emit a PyTorch `nn.Linear`: weight is [out, in]."""
    tensors[f"{prefix}.weight"] = _tensor(out_features, in_features)
    tensors[f"{prefix}.bias"] = _tensor(out_features)


def _build_fake_checkpoint():
    """Synthesize a FLUX checkpoint in the Black Forest Labs key layout."""
    tensors = {}

    _linear(HIDDEN, INPUT_CHANNELS, "img_in", tensors)
    _linear(HIDDEN, TEXT_DIM, "txt_in", tensors)
    _linear(HIDDEN, TIMESTEP_DIM, "time_in.in_layer", tensors)
    _linear(HIDDEN, HIDDEN, "time_in.out_layer", tensors)
    _linear(HIDDEN, Y_DIM, "vector_in.in_layer", tensors)
    _linear(HIDDEN, HIDDEN, "vector_in.out_layer", tensors)

    for i in range(DEPTH):
        p = f"double_blocks.{i}"
        for stream in ("img", "txt"):
            _linear(6 * HIDDEN, HIDDEN, f"{p}.{stream}_mod.lin", tensors)
            _linear(3 * HIDDEN, HIDDEN, f"{p}.{stream}_attn.qkv", tensors)
            _linear(HIDDEN, HIDDEN, f"{p}.{stream}_attn.proj", tensors)
            tensors[f"{p}.{stream}_attn.norm.query_norm.scale"] = _tensor(
                HEAD_DIM
            )
            tensors[f"{p}.{stream}_attn.norm.key_norm.scale"] = _tensor(
                HEAD_DIM
            )
            _linear(MLP_HIDDEN, HIDDEN, f"{p}.{stream}_mlp.0", tensors)
            _linear(HIDDEN, MLP_HIDDEN, f"{p}.{stream}_mlp.2", tensors)

    for i in range(DEPTH_SINGLE):
        p = f"single_blocks.{i}"
        _linear(3 * HIDDEN, HIDDEN, f"{p}.modulation.lin", tensors)
        _linear(3 * HIDDEN + MLP_HIDDEN, HIDDEN, f"{p}.linear1", tensors)
        _linear(HIDDEN, HIDDEN + MLP_HIDDEN, f"{p}.linear2", tensors)
        tensors[f"{p}.norm.query_norm.scale"] = _tensor(HEAD_DIM)
        tensors[f"{p}.norm.key_norm.scale"] = _tensor(HEAD_DIM)

    _linear(INPUT_CHANNELS, HIDDEN, "final_layer.linear", tensors)
    _linear(2 * HIDDEN, HIDDEN, "final_layer.adaLN_modulation.1", tensors)

    return tensors


def _build_backbone():
    return FluxBackbone(
        input_channels=INPUT_CHANNELS,
        hidden_size=HIDDEN,
        mlp_ratio=MLP_RATIO,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        depth_single_blocks=DEPTH_SINGLE,
        axes_dim=AXES_DIM,
        theta=10_000,
        use_bias=True,
        guidance_embed=False,
        image_shape=(None, INPUT_CHANNELS),
        text_shape=(None, TEXT_DIM),
        image_ids_shape=(None, 3),
        text_ids_shape=(None, 3),
        y_shape=(Y_DIM,),
    )


class ConvertFluxConfigTest(TestCase):
    def test_diffusers_config_key_names(self):
        """diffusers uses `num_layers` / `axes_dims_rope` / `guidance_embeds`.

        Reading the wrong spelling silently falls through to a default, which
        for `guidance_embeds` means FLUX.1-dev loads as a non-guidance model.
        """
        hf_config = {
            "in_channels": 64,
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "num_layers": 19,
            "num_single_layers": 38,
            "axes_dims_rope": [16, 56, 56],
            "joint_attention_dim": 4096,
            "pooled_projection_dim": 768,
            "guidance_embeds": True,
        }

        config = convert_flux.convert_backbone_config(hf_config)

        self.assertEqual(config["hidden_size"], 24 * 128)
        self.assertEqual(config["num_heads"], 24)
        self.assertEqual(config["depth"], 19)
        self.assertEqual(config["depth_single_blocks"], 38)
        self.assertEqual(config["axes_dim"], [16, 56, 56])
        self.assertEqual(config["input_channels"], 64)
        self.assertTrue(config["guidance_embed"])
        self.assertEqual(config["text_shape"], (None, 4096))
        self.assertEqual(config["y_shape"], (768,))

    def test_original_config_key_names(self):
        """The Black Forest Labs spelling must also work."""
        hf_config = {
            "in_channels": 64,
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "depth": 19,
            "depth_single_blocks": 38,
            "axes_dim": [16, 56, 56],
            "guidance_embed": True,
        }

        config = convert_flux.convert_backbone_config(hf_config)

        self.assertEqual(config["hidden_size"], 3072)
        self.assertEqual(config["depth"], 19)
        self.assertEqual(config["axes_dim"], [16, 56, 56])
        self.assertTrue(config["guidance_embed"])

    def test_sequence_axes_stay_dynamic(self):
        """A preset must not be pinned to one resolution / prompt length."""
        config = convert_flux.convert_backbone_config({})
        self.assertIsNone(config["image_shape"][0])
        self.assertIsNone(config["text_shape"][0])
        self.assertIsNone(config["image_ids_shape"][0])
        self.assertIsNone(config["text_ids_shape"][0])


class ConvertFluxWeightsTest(TestCase):
    def setUp(self):
        super().setUp()
        self.tensors = _build_fake_checkpoint()
        self.preset_dir = self.get_temp_dir()
        save_file(
            self.tensors, os.path.join(self.preset_dir, "model.safetensors")
        )

    def _convert(self):
        backbone = _build_backbone()
        with SafetensorLoader(self.preset_dir) as loader:
            convert_flux.convert_weights(backbone, loader, {})
        return backbone

    def test_conversion_runs(self):
        """Regression guard: `convert_weights` used to call a loader method
        that did not exist, so this path raised `AttributeError` immediately.
        """
        self.assertIsNotNone(self._convert())

    def test_input_embedders(self):
        backbone = self._convert()
        self.assertAllClose(
            backbone.image_input_embedder.kernel,
            self.tensors["img_in.weight"].T,
        )
        self.assertAllClose(
            backbone.image_input_embedder.bias, self.tensors["img_in.bias"]
        )
        self.assertAllClose(
            backbone.text_input_embedder.kernel,
            self.tensors["txt_in.weight"].T,
        )

    def test_mlp_embedders(self):
        backbone = self._convert()
        self.assertAllClose(
            backbone.time_input_embedder.input_layer.kernel,
            self.tensors["time_in.in_layer.weight"].T,
        )
        self.assertAllClose(
            backbone.vector_embedder.output_layer.kernel,
            self.tensors["vector_in.out_layer.weight"].T,
        )

    def test_double_block_attention(self):
        backbone = self._convert()
        block = backbone.double_blocks[0]
        self.assertAllClose(
            block.image_qkv.kernel,
            self.tensors["double_blocks.0.img_attn.qkv.weight"].T,
        )
        self.assertAllClose(
            block.text_attn_proj.kernel,
            self.tensors["double_blocks.0.txt_attn.proj.weight"].T,
        )

    def test_double_block_qk_norm_is_ported(self):
        """These scales exist in the checkpoint and were previously dropped.

        The layers initialize to ones, so a missing mapping is invisible
        unless we assert against the checkpoint value.
        """
        backbone = self._convert()
        block = backbone.double_blocks[0]
        expected = self.tensors[
            "double_blocks.0.img_attn.norm.query_norm.scale"
        ]

        self.assertAllClose(block.image_attn_norm.query_norm.scale, expected)
        self.assertAllClose(
            block.image_attn_norm.key_norm.scale,
            self.tensors["double_blocks.0.img_attn.norm.key_norm.scale"],
        )
        self.assertAllClose(
            block.text_attn_norm.query_norm.scale,
            self.tensors["double_blocks.0.txt_attn.norm.query_norm.scale"],
        )
        # Sanity: the fixture must differ from the default init, otherwise
        # the assertions above would pass even with no mapping at all.
        self.assertNotAllClose(expected, np.ones_like(expected))

    def test_single_block_qk_norm_is_ported(self):
        backbone = self._convert()
        block = backbone.single_blocks[0]
        self.assertAllClose(
            block.norm.query_norm.scale,
            self.tensors["single_blocks.0.norm.query_norm.scale"],
        )

    def test_modulation_and_mlp(self):
        backbone = self._convert()
        block = backbone.double_blocks[0]
        self.assertAllClose(
            block.image_mod.linear_projection.kernel,
            self.tensors["double_blocks.0.img_mod.lin.weight"].T,
        )
        self.assertAllClose(
            block.image_mlp.layers[0].kernel,
            self.tensors["double_blocks.0.img_mlp.0.weight"].T,
        )

    def test_final_layer(self):
        backbone = self._convert()
        self.assertAllClose(
            backbone.final_layer.linear.kernel,
            self.tensors["final_layer.linear.weight"].T,
        )
        # `adaLN_modulation` is Sequential(SiLU, Dense); index 1 is the Dense.
        self.assertAllClose(
            backbone.final_layer.adaLN_modulation.layers[1].kernel,
            self.tensors["final_layer.adaLN_modulation.1.weight"].T,
        )

    def test_missing_block_raises(self):
        """Silently skipping a block leaves it randomly initialized."""
        truncated = {
            k: v
            for k, v in self.tensors.items()
            if not k.startswith("double_blocks.0.img_attn.qkv")
        }
        preset_dir = self.get_temp_dir()
        save_file(truncated, os.path.join(preset_dir, "model.safetensors"))

        backbone = _build_backbone()
        with self.assertRaisesRegex(ValueError, "double_blocks.0"):
            with SafetensorLoader(preset_dir) as loader:
                convert_flux.convert_weights(backbone, loader, {})

    def test_guidance_mismatch_raises(self):
        """A dev-style config against a schnell checkpoint must not pass."""
        backbone = FluxBackbone(
            input_channels=INPUT_CHANNELS,
            hidden_size=HIDDEN,
            mlp_ratio=MLP_RATIO,
            num_heads=NUM_HEADS,
            depth=DEPTH,
            depth_single_blocks=DEPTH_SINGLE,
            axes_dim=AXES_DIM,
            theta=10_000,
            use_bias=True,
            guidance_embed=True,
            image_shape=(None, INPUT_CHANNELS),
            text_shape=(None, TEXT_DIM),
            image_ids_shape=(None, 3),
            text_ids_shape=(None, 3),
            y_shape=(Y_DIM,),
        )
        with self.assertRaisesRegex(ValueError, "guidance_in"):
            with SafetensorLoader(self.preset_dir) as loader:
                convert_flux.convert_weights(backbone, loader, {})

    def test_wrong_backbone_type_raises(self):
        with self.assertRaisesRegex(ValueError, "FluxBackbone"):
            with SafetensorLoader(self.preset_dir) as loader:
                convert_flux.convert_weights(object(), loader, {})


class ConvertFluxPresetTest(TestCase):
    @pytest.mark.extra_large
    def test_convert_schnell_preset(self):
        backbone = FluxBackbone.from_preset(
            "hf://black-forest-labs/FLUX.1-schnell"
        )
        self.assertIsInstance(backbone, FluxBackbone)
