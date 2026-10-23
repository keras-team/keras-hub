import os

import numpy as np
import pytest
from safetensors.numpy import save_file

from keras_hub.src.models.flux2.flux2_backbone import Flux2Backbone
from keras_hub.src.models.flux2.flux2_vae import Flux2VAE
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


# A tiny `AutoencoderKLFlux2`. GroupNorm uses 32 groups, so widths are
# multiples of 32; distinct stage widths catch a reversed decoder.
VAE_CHANNELS = [32, 64]
VAE_LAYERS_PER_BLOCK = 1
VAE_LATENT_CHANNELS = 4
VAE_CONFIG = {
    "block_out_channels": VAE_CHANNELS,
    "layers_per_block": VAE_LAYERS_PER_BLOCK,
    "latent_channels": VAE_LATENT_CHANNELS,
    "patch_size": [2, 2],
}


def _build_fake_vae_checkpoint():
    """Synthesize `AutoencoderKLFlux2` weights in the diffusers layout."""
    tensors = {}

    def conv(prefix, out_channels, in_channels, kernel_size):
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        tensors[f"{prefix}.weight"] = _tensor(*shape)
        tensors[f"{prefix}.bias"] = _tensor(out_channels) + 0.25

    def norm(prefix, channels):
        tensors[f"{prefix}.weight"] = _tensor(channels) + 1.0
        tensors[f"{prefix}.bias"] = _tensor(channels)

    def resnet(prefix, in_channels, out_channels):
        norm(f"{prefix}.norm1", in_channels)
        conv(f"{prefix}.conv1", out_channels, in_channels, 3)
        norm(f"{prefix}.norm2", out_channels)
        conv(f"{prefix}.conv2", out_channels, out_channels, 3)
        if in_channels != out_channels:
            conv(f"{prefix}.conv_shortcut", out_channels, in_channels, 1)

    def mid_block(prefix, channels):
        resnet(f"{prefix}.mid_block.resnets.0", channels, channels)
        attention = f"{prefix}.mid_block.attentions.0"
        norm(f"{attention}.group_norm", channels)
        for name in ("to_q", "to_k", "to_v", "to_out.0"):
            tensors[f"{attention}.{name}.weight"] = _tensor(channels, channels)
            tensors[f"{attention}.{name}.bias"] = _tensor(channels)
        resnet(f"{prefix}.mid_block.resnets.1", channels, channels)

    latent = VAE_LATENT_CHANNELS
    conv("encoder.conv_in", VAE_CHANNELS[0], 3, 3)
    in_channels = VAE_CHANNELS[0]
    for i, channels in enumerate(VAE_CHANNELS):
        for j in range(VAE_LAYERS_PER_BLOCK):
            resnet(
                f"encoder.down_blocks.{i}.resnets.{j}", in_channels, channels
            )
            in_channels = channels
        if i < len(VAE_CHANNELS) - 1:
            conv(
                f"encoder.down_blocks.{i}.downsamplers.0.conv",
                *[channels] * 2,
                3,
            )
    mid_block("encoder", VAE_CHANNELS[-1])
    norm("encoder.conv_norm_out", VAE_CHANNELS[-1])
    conv("encoder.conv_out", 2 * latent, VAE_CHANNELS[-1], 3)
    conv("quant_conv", 2 * latent, 2 * latent, 1)
    conv("post_quant_conv", latent, latent, 1)

    # diffusers' decoder stages run from the lowest resolution up.
    decoder_channels = VAE_CHANNELS[::-1]
    conv("decoder.conv_in", decoder_channels[0], latent, 3)
    mid_block("decoder", decoder_channels[0])
    in_channels = decoder_channels[0]
    for i, channels in enumerate(decoder_channels):
        for j in range(VAE_LAYERS_PER_BLOCK + 1):
            resnet(f"decoder.up_blocks.{i}.resnets.{j}", in_channels, channels)
            in_channels = channels
        if i < len(decoder_channels) - 1:
            conv(f"decoder.up_blocks.{i}.upsamplers.0.conv", *[channels] * 2, 3)
    norm("decoder.conv_norm_out", decoder_channels[-1])
    conv("decoder.conv_out", 3, decoder_channels[-1], 3)

    tensors["bn.running_mean"] = _tensor(4 * latent) - 0.5
    tensors["bn.running_var"] = _tensor(4 * latent) + 1.0
    tensors["bn.num_batches_tracked"] = np.array(0, dtype="int64")
    return tensors


class _TrackingLoader(SafetensorLoader):
    """Records which tensors were read and which variables were written."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.used = set()
        self.assigned = {}

    def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
        super().port_weight(keras_variable, hf_weight_key, hook_fn)
        self.used.add(hf_weight_key)
        entry = self.assigned.setdefault(
            id(keras_variable), [keras_variable.path, 0]
        )
        entry[1] += 1


def _to_keras_conv(weight):
    """PyTorch `[out, in, kh, kw]` -> Keras `[kh, kw, in, out]`."""
    return weight.transpose(2, 3, 1, 0)


class ConvertFlux2VAETest(TestCase):
    def setUp(self):
        super().setUp()
        self.tensors = _build_fake_vae_checkpoint()
        self.preset_dir = self.get_temp_dir()
        save_file(
            self.tensors, os.path.join(self.preset_dir, "model.safetensors")
        )

    def _convert(self):
        vae = Flux2VAE(**convert_flux2.convert_vae_config(VAE_CONFIG))
        with _TrackingLoader(self.preset_dir) as loader:
            convert_flux2.convert_vae_weights(vae, loader)
        return vae, loader

    def test_klein_vae_config(self):
        config = convert_flux2.convert_vae_config(
            {
                "act_fn": "silu",
                "batch_norm_eps": 1e-4,
                "block_out_channels": [128, 256, 512, 512],
                "latent_channels": 32,
                "layers_per_block": 2,
                "mid_block_add_attention": True,
                "norm_num_groups": 32,
                "patch_size": [2, 2],
                "use_post_quant_conv": True,
                "use_quant_conv": True,
            }
        )
        self.assertEqual(config["encoder_num_filters"], [128, 256, 512, 512])
        self.assertEqual(config["encoder_num_blocks"], [2, 2, 2, 2])
        self.assertEqual(config["decoder_num_filters"], [512, 512, 256, 128])
        self.assertEqual(config["decoder_num_blocks"], [3, 3, 3, 3])
        self.assertEqual(config["latent_channels"], 32)
        self.assertEqual(config["patch_size"], 2)
        self.assertEqual(config["batch_norm_epsilon"], 1e-4)

    def test_decoder_block_out_channels(self):
        config = convert_flux2.convert_vae_config(
            {**VAE_CONFIG, "decoder_block_out_channels": [64, 128]}
        )
        self.assertEqual(config["encoder_num_filters"], [32, 64])
        self.assertEqual(config["decoder_num_filters"], [128, 64])

    def test_rejects_unsupported_vae_config(self):
        with self.assertRaisesRegex(ValueError, "norm_num_groups"):
            convert_flux2.convert_vae_config(
                {**VAE_CONFIG, "norm_num_groups": 16}
            )
        with self.assertRaisesRegex(ValueError, "use_quant_conv"):
            convert_flux2.convert_vae_config(
                {**VAE_CONFIG, "use_quant_conv": False}
            )
        with self.assertRaisesRegex(ValueError, "square"):
            convert_flux2.convert_vae_config(
                {**VAE_CONFIG, "patch_size": [2, 1]}
            )

    def test_weights_are_complete_and_one_to_one(self):
        vae, loader = self._convert()
        never = [w.path for w in vae.weights if id(w) not in loader.assigned]
        twice = [path for path, n in loader.assigned.values() if n > 1]
        # `num_batches_tracked` is a training counter with no Keras use.
        unused = set(self.tensors) - loader.used - {"bn.num_batches_tracked"}
        self.assertEqual(never, [])
        self.assertEqual(twice, [])
        self.assertEqual(unused, set())
        self.assertEqual(
            vae.count_params(),
            sum(
                t.size
                for k, t in self.tensors.items()
                if k != "bn.num_batches_tracked"
            ),
        )

    def test_conv_and_attention_layouts(self):
        vae, _ = self._convert()
        self.assertAllClose(
            vae.encoder.input_projection.kernel,
            _to_keras_conv(self.tensors["encoder.conv_in.weight"]),
        )
        self.assertAllClose(
            vae.quant_conv.kernel,
            _to_keras_conv(self.tensors["quant_conv.weight"]),
        )
        # diffusers' attention uses `nn.Linear`; KerasHub uses 1x1 convs.
        self.assertAllClose(
            vae.decoder.mid_attention.query_conv2d.kernel,
            self.tensors["decoder.mid_block.attentions.0.to_q.weight"].T[
                None, None
            ],
        )

    def test_decoder_stage_order(self):
        # diffusers' `up_blocks.0` is the lowest resolution, which is also
        # the first `VAEDecoder` stack: no reversal (unlike the LDM layout).
        vae, _ = self._convert()
        blocks = vae.decoder.blocks
        self.assertAllClose(
            blocks[0].conv1.kernel,
            _to_keras_conv(
                self.tensors["decoder.up_blocks.0.resnets.0.conv1.weight"]
            ),
        )
        self.assertAllClose(
            blocks[2].residual_projection.kernel,
            _to_keras_conv(
                self.tensors[
                    "decoder.up_blocks.1.resnets.0.conv_shortcut.weight"
                ]
            ),
        )

    def test_latent_statistics(self):
        vae, _ = self._convert()
        self.assertAllClose(
            vae.latent_normalization.moving_mean,
            self.tensors["bn.running_mean"],
        )
        self.assertAllClose(
            vae.latent_normalization.moving_variance,
            self.tensors["bn.running_var"],
        )
