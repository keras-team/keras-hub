import json
import os

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.flux.flux_backbone import FluxBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.diffusers import convert_flux
from keras_hub.src.utils.diffusers.preset_loader import (
    DiffusersSafetensorLoader,
)

# `black-forest-labs/FLUX.1-schnell/transformer/config.json`.
SCHNELL_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.30.0.dev0",
    "attention_head_dim": 128,
    "guidance_embeds": False,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}

# A tiny `FluxTransformer2DModel`. `sum(axes_dims_rope)` is the head size.
TINY_KWARGS = {
    "patch_size": 1,
    "in_channels": 8,
    "num_layers": 1,
    "num_single_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 12,
    "pooled_projection_dim": 10,
    "axes_dims_rope": [2, 2, 4],
}


def save_tiny_flux(preset_dir, guidance_embeds=False, **save_kwargs):
    """Save a random, tiny FLUX.1 transformer in a diffusers repo layout."""
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    transformer = diffusers.FluxTransformer2DModel(
        **TINY_KWARGS, guidance_embeds=guidance_embeds
    ).eval()
    with torch.no_grad():
        # Not the default init, so that zero-initialized or all-ones
        # weights cannot hide a wrong mapping.
        for param in transformer.parameters():
            param.normal_(0.0, 0.2)
    transformer.save_pretrained(
        os.path.join(preset_dir, "transformer"), **save_kwargs
    )
    model_index = {
        "_class_name": "FluxPipeline",
        "transformer": ["diffusers", "FluxTransformer2DModel"],
    }
    with open(os.path.join(preset_dir, "model_index.json"), "w") as f:
        json.dump(model_index, f)
    return transformer


def run_both(backbone, transformer, guidance=False):
    """Run the backbone and the diffusers reference on the same inputs."""
    import torch

    rng = np.random.default_rng(0)
    batch, rows, cols, text_length = 2, 3, 4, 5
    rows_ids, cols_ids = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing="ij"
    )
    image_ids = np.stack(
        [np.zeros_like(rows_ids), rows_ids, cols_ids], axis=-1
    ).reshape(-1, 3)
    text_ids = np.zeros((text_length, 3))
    inputs = {
        "image": rng.standard_normal((batch, rows * cols, 8)),
        "image_ids": np.broadcast_to(image_ids, (batch, rows * cols, 3)),
        "text": rng.standard_normal((batch, text_length, 12)),
        "text_ids": np.broadcast_to(text_ids, (batch, text_length, 3)),
        "y": rng.standard_normal((batch, 10)),
        "timesteps": np.array([1.0, 0.25]),
    }
    if guidance:
        inputs["guidance"] = np.array([3.5, 1.5])
    inputs = {k: np.ascontiguousarray(v, "float32") for k, v in inputs.items()}
    actual = ops.convert_to_numpy(backbone(inputs))

    def tensor(name):
        return torch.from_numpy(inputs[name])

    with torch.no_grad():
        expected = transformer(
            hidden_states=tensor("image"),
            encoder_hidden_states=tensor("text"),
            pooled_projections=tensor("y"),
            timestep=tensor("timesteps"),
            img_ids=torch.from_numpy(image_ids.astype("float32")),
            txt_ids=torch.from_numpy(text_ids.astype("float32")),
            guidance=tensor("guidance") if guidance else None,
            return_dict=False,
        )[0].numpy()
    return actual, expected


class ConvertFluxConfigTest(TestCase):
    def test_schnell_config(self):
        config = convert_flux.convert_backbone_config(SCHNELL_CONFIG)
        self.assertEqual(config["hidden_size"], 24 * 128)
        self.assertEqual(config["num_heads"], 24)
        self.assertEqual(config["depth"], 19)
        self.assertEqual(config["depth_single_blocks"], 38)
        # Not in this older config: diffusers' default applies.
        self.assertEqual(config["axes_dim"], [16, 56, 56])
        self.assertEqual(config["input_channels"], 64)
        self.assertFalse(config["guidance_embed"])
        self.assertEqual(config["text_shape"], (None, 4096))
        self.assertEqual(config["y_shape"], (768,))
        # Sequence axes stay dynamic.
        self.assertIsNone(config["image_shape"][0])
        self.assertIsNone(config["text_shape"][0])

    def test_missing_key_raises(self):
        config = dict(SCHNELL_CONFIG)
        del config["num_layers"]
        with self.assertRaisesRegex(KeyError, "num_layers"):
            convert_flux.convert_backbone_config(config)

    def test_rejects_patch_size(self):
        with self.assertRaisesRegex(ValueError, "patch_size"):
            convert_flux.convert_backbone_config(
                {**SCHNELL_CONFIG, "patch_size": 2}
            )


class ConvertFluxWeightsTest(TestCase):
    def test_matches_diffusers(self):
        preset_dir = self.get_temp_dir()
        transformer = save_tiny_flux(preset_dir)
        backbone = FluxBackbone.from_preset(preset_dir)
        actual, expected = run_both(backbone, transformer)
        self.assertAllClose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_matches_diffusers_sharded(self):
        preset_dir = self.get_temp_dir()
        transformer = save_tiny_flux(preset_dir, max_shard_size="20KB")
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    preset_dir,
                    "transformer",
                    "diffusion_pytorch_model.safetensors.index.json",
                )
            )
        )
        backbone = FluxBackbone.from_preset(preset_dir)
        actual, expected = run_both(backbone, transformer)
        self.assertAllClose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_guidance_embed(self):
        preset_dir = self.get_temp_dir()
        transformer = save_tiny_flux(preset_dir, guidance_embeds=True)
        backbone = FluxBackbone.from_preset(preset_dir)
        self.assertIn("guidance", backbone.input)
        actual, expected = run_both(backbone, transformer, guidance=True)
        self.assertAllClose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_class_detection(self):
        preset_dir = self.get_temp_dir()
        save_tiny_flux(preset_dir)
        backbone = Backbone.from_preset(preset_dir, load_weights=False)
        self.assertIsInstance(backbone, FluxBackbone)

    def test_missing_tensor_raises(self):
        # A name-suffix lookup would silently read
        # `single_transformer_blocks.0.attn.to_q.weight` instead.
        from safetensors.numpy import load_file
        from safetensors.numpy import save_file

        preset_dir = self.get_temp_dir()
        save_tiny_flux(preset_dir)
        path = os.path.join(
            preset_dir, "transformer", "diffusion_pytorch_model.safetensors"
        )
        tensors = load_file(path)
        del tensors["transformer_blocks.0.attn.to_q.weight"]
        save_file(tensors, path)
        with self.assertRaisesRegex(
            ValueError, "transformer_blocks.0.attn.to_q.weight"
        ):
            FluxBackbone.from_preset(preset_dir)

    def test_extra_tensor_raises(self):
        from safetensors.numpy import load_file
        from safetensors.numpy import save_file

        preset_dir = self.get_temp_dir()
        save_tiny_flux(preset_dir)
        path = os.path.join(
            preset_dir, "transformer", "diffusion_pytorch_model.safetensors"
        )
        tensors = load_file(path)
        tensors["transformer_blocks.1.attn.to_q.weight"] = tensors[
            "transformer_blocks.0.attn.to_q.weight"
        ]
        save_file(tensors, path)
        with self.assertRaisesRegex(ValueError, "never read"):
            FluxBackbone.from_preset(preset_dir)

    def test_wrong_backbone_type_raises(self):
        preset_dir = self.get_temp_dir()
        save_tiny_flux(preset_dir)
        with self.assertRaisesRegex(ValueError, "FluxBackbone"):
            with DiffusersSafetensorLoader(preset_dir, "transformer") as loader:
                convert_flux.convert_weights(object(), loader)


class ConvertFluxPresetTest(TestCase):
    @pytest.mark.large
    def test_tiny_flux_pipe(self):
        # A public, tiny FLUX.1 diffusers repo. FLUX.1 [schnell] itself is
        # gated on the Hub.
        backbone = FluxBackbone.from_preset(
            "hf://hf-internal-testing/tiny-flux-pipe"
        )
        self.assertIsInstance(backbone, FluxBackbone)
        self.assertEqual(backbone.depth, 1)
