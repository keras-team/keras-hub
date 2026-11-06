import json
import os

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.flux2.flux2_backbone import Flux2Backbone
from keras_hub.src.models.flux2.flux2_text_to_image import Flux2TextToImage
from keras_hub.src.models.flux2.flux2_vae import Flux2VAE
from keras_hub.src.models.text_to_image import TextToImage
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_flux2
from keras_hub.src.utils.transformers.preset_loader import (
    DiffusersSafetensorLoader,
)

# `black-forest-labs/FLUX.2-klein-4B/transformer/config.json`.
KLEIN_4B_CONFIG = {
    "_class_name": "Flux2Transformer2DModel",
    "_diffusers_version": "0.37.0.dev0",
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

# `black-forest-labs/FLUX.2-klein-4B/vae/config.json`.
KLEIN_VAE_CONFIG = {
    "_class_name": "AutoencoderKLFlux2",
    "act_fn": "silu",
    "batch_norm_eps": 0.0001,
    "batch_norm_momentum": 0.1,
    "block_out_channels": [128, 256, 512, 512],
    "in_channels": 3,
    "latent_channels": 32,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "patch_size": [2, 2],
    "use_post_quant_conv": True,
    "use_quant_conv": True,
}

# A tiny `Flux2Transformer2DModel`. `sum(axes_dims_rope)` is the head size.
TINY_KWARGS = {
    "patch_size": 1,
    "in_channels": 16,
    "num_layers": 1,
    "num_single_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 12,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": [2, 2, 2, 2],
    "rope_theta": 2000,
}

# A tiny `AutoencoderKLFlux2`. GroupNorm uses 32 groups, so widths are
# multiples of 32; distinct stage widths catch a reversed decoder.
TINY_VAE_KWARGS = {
    "block_out_channels": (32, 64),
    "down_block_types": ("DownEncoderBlock2D",) * 2,
    "up_block_types": ("UpDecoderBlock2D",) * 2,
    "layers_per_block": 1,
    "latent_channels": 4,
    "sample_size": 32,
}


def write_model_index(preset_dir, **components):
    model_index = {"_class_name": "Flux2KleinPipeline", **components}
    with open(os.path.join(preset_dir, "model_index.json"), "w") as f:
        json.dump(model_index, f)


def save_tiny_transformer(preset_dir, guidance_embeds=False, **kwargs):
    """Save a random, tiny FLUX.2 transformer in a diffusers repo layout."""
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    transformer = diffusers.Flux2Transformer2DModel(
        **{**TINY_KWARGS, **kwargs}, guidance_embeds=guidance_embeds
    ).eval()
    with torch.no_grad():
        # Not the default init, so that all-ones norm scales cannot hide a
        # wrong mapping.
        for param in transformer.parameters():
            param.normal_(0.0, 0.2)
    transformer.save_pretrained(os.path.join(preset_dir, "transformer"))
    write_model_index(
        preset_dir, transformer=["diffusers", "Flux2Transformer2DModel"]
    )
    return transformer


def save_tiny_vae(preset_dir):
    """Save a random, tiny `AutoencoderKLFlux2` to `preset_dir/vae`."""
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    vae = diffusers.AutoencoderKLFlux2(**TINY_VAE_KWARGS).eval()
    with torch.no_grad():
        for param in vae.parameters():
            param.add_(0.05 * torch.randn_like(param))
        vae.bn.running_mean.normal_(0.0, 0.5)
        vae.bn.running_var.uniform_(0.5, 2.0)
    vae.save_pretrained(os.path.join(preset_dir, "vae"))
    return vae


def run_both(backbone, transformer, guidance=False, dtype="float32"):
    """Run the backbone and the diffusers reference on the same inputs."""
    import torch

    rng = np.random.default_rng(0)
    batch, rows, cols, text_length = 2, 3, 4, 5
    rows_ids, cols_ids = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing="ij"
    )
    zeros = np.zeros_like(rows_ids)
    image_ids = np.stack([zeros, rows_ids, cols_ids, zeros], axis=-1)
    image_ids = image_ids.reshape(-1, 4)
    index = np.arange(text_length)
    text_ids = np.stack([0 * index, 0 * index, 0 * index, index], axis=-1)
    inputs = {
        "image": rng.standard_normal((batch, rows * cols, 16)),
        "image_ids": np.broadcast_to(image_ids, (batch, rows * cols, 4)),
        "text": rng.standard_normal((batch, text_length, 12)),
        "text_ids": np.broadcast_to(text_ids, (batch, text_length, 4)),
        "timesteps": np.array([1.0, 0.25]),
    }
    if guidance:
        inputs["guidance"] = np.array([4.0, 2.5])
    inputs = {k: np.ascontiguousarray(v, "float32") for k, v in inputs.items()}
    actual = ops.convert_to_numpy(backbone(inputs))

    def tensor(name):
        return torch.from_numpy(inputs[name])

    with torch.no_grad():
        expected = transformer.to(getattr(torch, dtype))(
            hidden_states=tensor("image"),
            encoder_hidden_states=tensor("text"),
            timestep=tensor("timesteps"),
            img_ids=tensor("image_ids"),
            txt_ids=tensor("text_ids"),
            guidance=tensor("guidance") if guidance else None,
            return_dict=False,
        )[0].numpy()
    return actual, expected


class ConvertFlux2ConfigTest(TestCase):
    def test_klein_4b_config(self):
        config = convert_flux2.convert_backbone_config(KLEIN_4B_CONFIG)
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
        # Sequence axes stay dynamic.
        self.assertEqual(config["latent_shape"], (None, 128))
        self.assertIsNone(config["text_ids_shape"][0])

    def test_missing_key_raises(self):
        config = dict(KLEIN_4B_CONFIG)
        del config["num_single_layers"]
        with self.assertRaisesRegex(KeyError, "num_single_layers"):
            convert_flux2.convert_backbone_config(config)

    def test_rejects_unsupported_configs(self):
        for key, value in (
            ("patch_size", 2),
            ("out_channels", 64),
            ("eps", 1e-5),
            ("timestep_guidance_channels", 128),
        ):
            with self.assertRaisesRegex(ValueError, key):
                convert_flux2.convert_backbone_config(
                    {**KLEIN_4B_CONFIG, key: value}
                )

    def test_klein_vae_config(self):
        config = convert_flux2.convert_vae_config(KLEIN_VAE_CONFIG)
        self.assertEqual(config["encoder_num_filters"], [128, 256, 512, 512])
        self.assertEqual(config["encoder_num_blocks"], [2, 2, 2, 2])
        self.assertEqual(config["decoder_num_filters"], [512, 512, 256, 128])
        self.assertEqual(config["decoder_num_blocks"], [3, 3, 3, 3])
        self.assertEqual(config["latent_channels"], 32)
        self.assertEqual(config["patch_size"], 2)
        self.assertEqual(config["batch_norm_epsilon"], 1e-4)

    def test_vae_config_errors(self):
        config = dict(KLEIN_VAE_CONFIG)
        del config["latent_channels"]
        with self.assertRaisesRegex(KeyError, "latent_channels"):
            convert_flux2.convert_vae_config(config)
        for key, value in (
            ("norm_num_groups", 16),
            ("use_quant_conv", False),
            ("patch_size", [2, 1]),
        ):
            with self.assertRaisesRegex(ValueError, key.split("_")[0]):
                convert_flux2.convert_vae_config(
                    {**KLEIN_VAE_CONFIG, key: value}
                )


class ConvertFlux2WeightsTest(TestCase):
    def test_matches_diffusers(self):
        preset_dir = self.get_temp_dir()
        transformer = save_tiny_transformer(preset_dir)
        backbone = Flux2Backbone.from_preset(preset_dir)
        actual, expected = run_both(backbone, transformer)
        self.assertAllClose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_guidance_embed(self):
        preset_dir = self.get_temp_dir()
        transformer = save_tiny_transformer(preset_dir, guidance_embeds=True)
        backbone = Flux2Backbone.from_preset(preset_dir)
        self.assertIn("guidance", backbone.input)
        actual, expected = run_both(backbone, transformer, guidance=True)
        self.assertAllClose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_bfloat16_checkpoint(self):
        # The released checkpoints are bfloat16.
        import torch

        preset_dir = self.get_temp_dir()
        transformer = save_tiny_transformer(preset_dir).to(torch.bfloat16)
        transformer.save_pretrained(os.path.join(preset_dir, "transformer"))
        backbone = Flux2Backbone.from_preset(preset_dir)
        # The reference runs in float32 on the same bfloat16 values.
        actual, expected = run_both(backbone, transformer.float())
        self.assertAllClose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_class_detection(self):
        preset_dir = self.get_temp_dir()
        save_tiny_transformer(preset_dir)
        backbone = Backbone.from_preset(preset_dir, load_weights=False)
        self.assertIsInstance(backbone, Flux2Backbone)

    def test_missing_and_extra_tensors_raise(self):
        from safetensors.numpy import load_file
        from safetensors.numpy import save_file

        preset_dir = self.get_temp_dir()
        save_tiny_transformer(preset_dir)
        path = os.path.join(
            preset_dir, "transformer", "diffusion_pytorch_model.safetensors"
        )
        tensors = load_file(path)
        missing = dict(tensors)
        del missing["transformer_blocks.0.attn.norm_added_k.weight"]
        save_file(missing, path)
        with self.assertRaisesRegex(ValueError, "norm_added_k"):
            Flux2Backbone.from_preset(preset_dir)
        extra = dict(tensors)
        extra["single_transformer_blocks.2.attn.to_out.weight"] = tensors[
            "single_transformer_blocks.1.attn.to_out.weight"
        ]
        save_file(extra, path)
        with self.assertRaisesRegex(ValueError, "never read"):
            Flux2Backbone.from_preset(preset_dir)

    def test_flux1_checkpoint_raises(self):
        # FLUX.1 has per-block modulation and no shared projections.
        preset_dir = self.get_temp_dir()
        save_tiny_transformer(preset_dir)
        backbone = Flux2Backbone(
            **convert_flux2.convert_backbone_config(
                {**KLEIN_4B_CONFIG, **TINY_KWARGS, "guidance_embeds": False}
            )
        )
        path = os.path.join(
            preset_dir, "transformer", "diffusion_pytorch_model.safetensors"
        )
        from safetensors.numpy import load_file
        from safetensors.numpy import save_file

        tensors = load_file(path)
        tensors = {
            k: v for k, v in tensors.items() if "stream_modulation" not in k
        }
        save_file(tensors, path)
        with self.assertRaisesRegex(ValueError, "stream_modulation"):
            with DiffusersSafetensorLoader(preset_dir, "transformer") as loader:
                convert_flux2.convert_weights(backbone, loader)

    def test_wrong_backbone_type_raises(self):
        preset_dir = self.get_temp_dir()
        save_tiny_transformer(preset_dir)
        with self.assertRaisesRegex(ValueError, "Flux2Backbone"):
            with DiffusersSafetensorLoader(preset_dir, "transformer") as loader:
                convert_flux2.convert_weights(object(), loader)


class ConvertFlux2VAETest(TestCase):
    def setUp(self):
        super().setUp()
        self.preset_dir = self.get_temp_dir()
        self.reference = save_tiny_vae(self.preset_dir)
        config = self.reference.config
        self.vae = Flux2VAE(**convert_flux2.convert_vae_config(config))
        with DiffusersSafetensorLoader(self.preset_dir, "vae") as loader:
            convert_flux2.convert_vae_weights(self.vae, loader)

    def test_decode_matches_diffusers(self):
        import torch
        from diffusers import Flux2KleinPipeline

        rng = np.random.default_rng(0)
        latents = rng.standard_normal((2, 4, 4, 16)).astype("float32")
        actual = ops.convert_to_numpy(self.vae.decode(latents))
        # `Flux2KleinPipeline.__call__`, after the denoising loop.
        with torch.no_grad():
            x = torch.from_numpy(latents).permute(0, 3, 1, 2)
            bn = self.reference.bn
            std = torch.sqrt(
                bn.running_var.view(1, -1, 1, 1)
                + self.reference.config.batch_norm_eps
            )
            x = x * std + bn.running_mean.view(1, -1, 1, 1)
            x = Flux2KleinPipeline._unpatchify_latents(x)
            expected = self.reference.decode(x, return_dict=False)[0]
        expected = expected.permute(0, 2, 3, 1).numpy()
        self.assertAllClose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_encode_matches_diffusers(self):
        import torch
        from diffusers import Flux2KleinPipeline

        rng = np.random.default_rng(1)
        images = rng.uniform(-1, 1, (2, 16, 16, 3)).astype("float32")
        actual = ops.convert_to_numpy(self.vae.encode(images))
        # `Flux2KleinPipeline._encode_vae_image`.
        with torch.no_grad():
            x = torch.from_numpy(images).permute(0, 3, 1, 2)
            x = self.reference.encode(x).latent_dist.mode()
            x = Flux2KleinPipeline._patchify_latents(x)
            bn = self.reference.bn
            std = torch.sqrt(
                bn.running_var.view(1, -1, 1, 1)
                + self.reference.config.batch_norm_eps
            )
            expected = (x - bn.running_mean.view(1, -1, 1, 1)) / std
        expected = expected.permute(0, 2, 3, 1).numpy()
        self.assertAllClose(actual, expected, atol=1e-3, rtol=1e-3)


# Byte-level BPE vocabulary for the tiny tokenizer, and the chat template's
# special tokens.
MERGES = [["Ġ", "a"], ["a", "i"], ["p", "l"], ["n", "e"], ["ai", "r"]]
MERGES += [["pl", "a"], ["pla", "ne"], ["u", "s"], ["e", "r"], ["us", "er"]]
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
SPECIAL_TOKENS += ["<think>", "</think>"]


def save_tiny_text_encoder(preset_dir):
    """Save a tiny `Qwen3ForCausalLM` and its tokenizer, as klein does."""
    transformers = pytest.importorskip("transformers")
    tokens = sorted({t for pair in MERGES for t in pair + ["".join(pair)]})
    tokens += [chr(c) for c in range(ord("a"), ord("z") + 1)]
    tokens += ["Ċ", "ĊĊ", "Ġ"]
    vocab = {token: i for i, token in enumerate(dict.fromkeys(tokens))}
    added = [
        {"id": len(vocab) + i, "content": token, "special": True}
        for i, token in enumerate(SPECIAL_TOKENS)
    ]
    os.makedirs(os.path.join(preset_dir, "tokenizer"))
    with open(
        os.path.join(preset_dir, "tokenizer", "tokenizer.json"), "w"
    ) as f:
        json.dump(
            {
                "model": {"type": "BPE", "vocab": vocab, "merges": MERGES},
                "added_tokens": added,
            },
            f,
        )
    config = transformers.Qwen3Config(
        vocab_size=len(vocab) + len(added),
        hidden_size=16,
        intermediate_size=32,
        # `Flux2TextToImage` reads hidden states 9, 18 and 27.
        num_hidden_layers=27,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    model = transformers.Qwen3ForCausalLM(config)
    model.save_pretrained(os.path.join(preset_dir, "text_encoder"))


class ConvertFlux2PipelineTest(TestCase):
    def test_text_to_image_from_diffusers_repo(self):
        preset_dir = self.get_temp_dir()
        # Three Qwen3 hidden states of width 16 give 48 text features.
        save_tiny_transformer(preset_dir, joint_attention_dim=48)
        save_tiny_vae(preset_dir)
        save_tiny_text_encoder(preset_dir)
        write_model_index(
            preset_dir,
            is_distilled=True,
            scheduler=["diffusers", "FlowMatchEulerDiscreteScheduler"],
            text_encoder=["transformers", "Qwen3ForCausalLM"],
            tokenizer=["transformers", "Qwen2TokenizerFast"],
            transformer=["diffusers", "Flux2Transformer2DModel"],
            vae=["diffusers", "AutoencoderKLFlux2"],
        )
        for cls in (Flux2TextToImage, TextToImage):
            text_to_image = cls.from_preset(preset_dir, image_shape=(32, 32, 3))
            self.assertIsInstance(text_to_image, Flux2TextToImage)
            self.assertTrue(text_to_image.is_distilled)
            self.assertEqual(text_to_image.latent_shape, (None, 8, 8, 16))
        image = text_to_image.generate("airplane", num_steps=2, seed=0)
        self.assertEqual(image.shape, (32, 32, 3))

    def test_flux1_repo_has_no_task(self):
        preset_dir = self.get_temp_dir()
        from keras_hub.src.utils.transformers.convert_flux_test import (
            save_tiny_flux,
        )

        save_tiny_flux(preset_dir)
        with self.assertRaisesRegex(ValueError, "FLUX.1"):
            TextToImage.from_preset(preset_dir)


class ConvertFlux2PresetTest(TestCase):
    @pytest.mark.extra_large
    def test_klein_4b_matches_checkpoint_size(self):
        """Built from the published config, the backbone must have exactly
        as many parameters as `flux-2-klein-4b.safetensors`."""
        backbone = Flux2Backbone(
            **convert_flux2.convert_backbone_config(KLEIN_4B_CONFIG)
        )
        self.assertEqual(backbone.count_params(), 3_875_544_576)

    @pytest.mark.extra_large
    def test_klein_4b_from_hub(self):
        # Public, unlike FLUX.1 [schnell].
        backbone = Flux2Backbone.from_preset(
            "hf://black-forest-labs/FLUX.2-klein-4B"
        )
        self.assertEqual(backbone.count_params(), 3_875_544_576)
