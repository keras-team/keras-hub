import os

import numpy as np
import pytest
from safetensors.numpy import save_file

from keras_hub.src.models.flux2.flux2_backbone import Flux2Backbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_flux2
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

# A deliberately tiny FLUX.2. `sum(AXES_DIM)` must equal HIDDEN // NUM_HEADS.
INPUT_CHANNELS = 8
HIDDEN = 16
NUM_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS
MLP_RATIO = 3.0
MLP_HIDDEN = int(HIDDEN * MLP_RATIO)
AXES_DIM = [2, 2, 2, 2]
DEPTH = 1
DEPTH_SINGLE = 2
TEXT_DIM = 12
TIMESTEP_DIM = 256


def _tensor(*shape):
    """Deterministic, distinguishable values (never all-ones)."""
    size = int(np.prod(shape))
    return (np.arange(size, dtype="float32") * 0.01 + 0.5).reshape(shape)


def _linear(out_features, in_features, prefix, tensors):
    """Emit a bias-free PyTorch `nn.Linear`: weight is [out, in]."""
    tensors[f"{prefix}.weight"] = _tensor(out_features, in_features)


def _build_fake_checkpoint(guidance=False):
    """Synthesize a FLUX.2 checkpoint in the Black Forest Labs layout."""
    tensors = {}
    _linear(HIDDEN, INPUT_CHANNELS, "img_in", tensors)
    _linear(HIDDEN, TEXT_DIM, "txt_in", tensors)
    _linear(HIDDEN, TIMESTEP_DIM, "time_in.in_layer", tensors)
    _linear(HIDDEN, HIDDEN, "time_in.out_layer", tensors)
    if guidance:
        _linear(HIDDEN, TIMESTEP_DIM, "guidance_in.in_layer", tensors)
        _linear(HIDDEN, HIDDEN, "guidance_in.out_layer", tensors)
    _linear(6 * HIDDEN, HIDDEN, "double_stream_modulation_img.lin", tensors)
    _linear(6 * HIDDEN, HIDDEN, "double_stream_modulation_txt.lin", tensors)
    _linear(3 * HIDDEN, HIDDEN, "single_stream_modulation.lin", tensors)

    for i in range(DEPTH):
        p = f"double_blocks.{i}"
        for stream in ("img", "txt"):
            _linear(3 * HIDDEN, HIDDEN, f"{p}.{stream}_attn.qkv", tensors)
            _linear(HIDDEN, HIDDEN, f"{p}.{stream}_attn.proj", tensors)
            tensors[f"{p}.{stream}_attn.norm.query_norm.scale"] = _tensor(
                HEAD_DIM
            )
            tensors[f"{p}.{stream}_attn.norm.key_norm.scale"] = (
                _tensor(HEAD_DIM) + 1.0
            )
            _linear(2 * MLP_HIDDEN, HIDDEN, f"{p}.{stream}_mlp.0", tensors)
            _linear(HIDDEN, MLP_HIDDEN, f"{p}.{stream}_mlp.2", tensors)

    for i in range(DEPTH_SINGLE):
        p = f"single_blocks.{i}"
        _linear(3 * HIDDEN + 2 * MLP_HIDDEN, HIDDEN, f"{p}.linear1", tensors)
        _linear(HIDDEN, HIDDEN + MLP_HIDDEN, f"{p}.linear2", tensors)
        tensors[f"{p}.norm.query_norm.scale"] = _tensor(HEAD_DIM)
        tensors[f"{p}.norm.key_norm.scale"] = _tensor(HEAD_DIM) + 1.0

    _linear(2 * HIDDEN, HIDDEN, "final_layer.adaLN_modulation.1", tensors)
    _linear(INPUT_CHANNELS, HIDDEN, "final_layer.linear", tensors)
    return tensors


def _build_backbone(guidance=False):
    return Flux2Backbone(
        input_channels=INPUT_CHANNELS,
        hidden_size=HIDDEN,
        mlp_ratio=MLP_RATIO,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        depth_single_blocks=DEPTH_SINGLE,
        axes_dim=AXES_DIM,
        theta=2000,
        guidance_embed=guidance,
        image_shape=(None, INPUT_CHANNELS),
        text_shape=(None, TEXT_DIM),
    )


class ConvertFlux2ConfigTest(TestCase):
    def test_diffusers_klein_4b_config(self):
        """`black-forest-labs/FLUX.2-klein-4B/transformer/config.json`."""
        hf_config = {
            "_class_name": "Flux2Transformer2DModel",
            "attention_head_dim": 128,
            "axes_dims_rope": [32, 32, 32, 32],
            "eps": 1e-06,
            "guidance_embeds": False,
            "in_channels": 128,
            "joint_attention_dim": 7680,
            "mlp_ratio": 3.0,
            "num_attention_heads": 24,
            "num_layers": 5,
            "num_single_layers": 20,
            "out_channels": None,
            "patch_size": 1,
            "rope_theta": 2000,
            "timestep_guidance_channels": 256,
        }
        config = convert_flux2.convert_backbone_config(hf_config)
        self.assertEqual(config["hidden_size"], 3072)
        self.assertEqual(config["num_heads"], 24)
        self.assertEqual(config["depth"], 5)
        self.assertEqual(config["depth_single_blocks"], 20)
        self.assertEqual(config["axes_dim"], [32, 32, 32, 32])
        self.assertEqual(config["theta"], 2000)
        self.assertEqual(config["mlp_ratio"], 3.0)
        self.assertEqual(config["input_channels"], 128)
        self.assertFalse(config["guidance_embed"])
        self.assertEqual(config["text_shape"], (None, 7680))
        self.assertEqual(config["image_ids_shape"], (None, 4))

    def test_original_config_key_names(self):
        """Black Forest Labs `Flux2Params` spelling (FLUX.2 [dev])."""
        hf_config = {
            "in_channels": 128,
            "context_in_dim": 15360,
            "hidden_size": 6144,
            "num_heads": 48,
            "depth": 8,
            "depth_single_blocks": 48,
            "axes_dim": [32, 32, 32, 32],
            "theta": 2000,
            "mlp_ratio": 3.0,
            "use_guidance_embed": True,
        }
        config = convert_flux2.convert_backbone_config(hf_config)
        self.assertEqual(config["hidden_size"], 6144)
        self.assertEqual(config["num_heads"], 48)
        self.assertEqual(config["depth"], 8)
        self.assertEqual(config["depth_single_blocks"], 48)
        self.assertEqual(config["text_shape"], (None, 15360))
        self.assertTrue(config["guidance_embed"])

    def test_defaults_are_klein_4b_with_dynamic_axes(self):
        config = convert_flux2.convert_backbone_config({})
        backbone_param_count_inputs = (
            config["hidden_size"],
            config["depth"],
            config["depth_single_blocks"],
        )
        self.assertEqual(backbone_param_count_inputs, (3072, 5, 20))
        self.assertIsNone(config["image_shape"][0])
        self.assertIsNone(config["text_shape"][0])
        self.assertIsNone(config["image_ids_shape"][0])
        self.assertIsNone(config["text_ids_shape"][0])

    def test_rejects_unsupported_patch_size(self):
        with self.assertRaisesRegex(ValueError, "patch_size"):
            convert_flux2.convert_backbone_config({"patch_size": 2})

    def test_rejects_mismatched_out_channels(self):
        with self.assertRaisesRegex(ValueError, "out_channels"):
            convert_flux2.convert_backbone_config(
                {"in_channels": 128, "out_channels": 64}
            )


class ConvertFlux2WeightsTest(TestCase):
    def setUp(self):
        super().setUp()
        self.tensors = _build_fake_checkpoint()
        self.preset_dir = self.get_temp_dir()
        save_file(
            self.tensors, os.path.join(self.preset_dir, "model.safetensors")
        )

    def _convert(self, guidance=False, preset_dir=None):
        backbone = _build_backbone(guidance=guidance)
        with SafetensorLoader(preset_dir or self.preset_dir) as loader:
            convert_flux2.convert_weights(backbone, loader, {})
        return backbone

    def test_parameter_count_matches_checkpoint(self):
        backbone = self._convert()
        self.assertEqual(
            backbone.count_params(),
            sum(t.size for t in self.tensors.values()),
        )

    def test_shared_modulation(self):
        backbone = self._convert()
        self.assertAllClose(
            backbone.double_stream_modulation_image.linear_projection.kernel,
            self.tensors["double_stream_modulation_img.lin.weight"].T,
        )
        self.assertAllClose(
            backbone.double_stream_modulation_text.linear_projection.kernel,
            self.tensors["double_stream_modulation_txt.lin.weight"].T,
        )
        self.assertAllClose(
            backbone.single_stream_modulation.linear_projection.kernel,
            self.tensors["single_stream_modulation.lin.weight"].T,
        )

    def test_embedders(self):
        backbone = self._convert()
        self.assertAllClose(
            backbone.image_input_embedder.kernel,
            self.tensors["img_in.weight"].T,
        )
        self.assertAllClose(
            backbone.text_input_embedder.kernel,
            self.tensors["txt_in.weight"].T,
        )
        self.assertAllClose(
            backbone.time_input_embedder.output_layer.kernel,
            self.tensors["time_in.out_layer.weight"].T,
        )

    def test_double_block(self):
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
        self.assertAllClose(
            block.image_mlp.input_layer.kernel,
            self.tensors["double_blocks.0.img_mlp.0.weight"].T,
        )
        self.assertAllClose(
            block.text_mlp.output_layer.kernel,
            self.tensors["double_blocks.0.txt_mlp.2.weight"].T,
        )
        # Query and key scales differ in the fixture, so a swap would fail.
        self.assertAllClose(
            block.image_attn_norm.key_norm.scale,
            self.tensors["double_blocks.0.img_attn.norm.key_norm.scale"],
        )
        self.assertAllClose(
            block.text_attn_norm.query_norm.scale,
            self.tensors["double_blocks.0.txt_attn.norm.query_norm.scale"],
        )

    def test_single_block(self):
        backbone = self._convert()
        block = backbone.single_blocks[1]
        self.assertAllClose(
            block.linear1.kernel,
            self.tensors["single_blocks.1.linear1.weight"].T,
        )
        self.assertAllClose(
            block.linear2.kernel,
            self.tensors["single_blocks.1.linear2.weight"].T,
        )
        self.assertAllClose(
            block.norm.key_norm.scale,
            self.tensors["single_blocks.1.norm.key_norm.scale"],
        )

    def test_final_layer(self):
        backbone = self._convert()
        self.assertAllClose(
            backbone.final_layer.adaLN_modulation.kernel,
            self.tensors["final_layer.adaLN_modulation.1.weight"].T,
        )
        self.assertAllClose(
            backbone.final_layer.linear.kernel,
            self.tensors["final_layer.linear.weight"].T,
        )

    def test_guidance_embedder(self):
        tensors = _build_fake_checkpoint(guidance=True)
        preset_dir = self.get_temp_dir()
        save_file(tensors, os.path.join(preset_dir, "model.safetensors"))
        backbone = self._convert(guidance=True, preset_dir=preset_dir)
        self.assertAllClose(
            backbone.guidance_input_embedder.input_layer.kernel,
            tensors["guidance_in.in_layer.weight"].T,
        )

    def test_guidance_mismatch_raises(self):
        """A [dev]-style config against a [klein] checkpoint must not pass."""
        with self.assertRaisesRegex(ValueError, "guidance_in"):
            self._convert(guidance=True)

    def test_missing_block_raises(self):
        """Silently skipping a block would leave it randomly initialized."""
        truncated = {
            k: v
            for k, v in self.tensors.items()
            if not k.startswith("single_blocks.1.linear2")
        }
        preset_dir = self.get_temp_dir()
        save_file(truncated, os.path.join(preset_dir, "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "single_blocks.1"):
            self._convert(preset_dir=preset_dir)

    def test_flux1_checkpoint_raises(self):
        """FLUX.1 has per-block modulation and no shared projections."""
        flux1_like = {
            k: v
            for k, v in self.tensors.items()
            if "stream_modulation" not in k
        }
        flux1_like["double_blocks.0.img_mod.lin.weight"] = _tensor(
            6 * HIDDEN, HIDDEN
        )
        preset_dir = self.get_temp_dir()
        save_file(flux1_like, os.path.join(preset_dir, "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "FluxBackbone"):
            self._convert(preset_dir=preset_dir)

    def test_wrong_backbone_type_raises(self):
        with self.assertRaisesRegex(ValueError, "Flux2Backbone"):
            with SafetensorLoader(self.preset_dir) as loader:
                convert_flux2.convert_weights(object(), loader, {})


class ConvertFlux2PresetTest(TestCase):
    @pytest.mark.extra_large
    def test_klein_4b_matches_checkpoint_size(self):
        """Built from the published config, the backbone must have exactly
        as many parameters as `flux-2-klein-4b.safetensors`."""
        backbone = Flux2Backbone(**convert_flux2.convert_backbone_config({}))
        self.assertEqual(backbone.count_params(), 3_875_544_576)
