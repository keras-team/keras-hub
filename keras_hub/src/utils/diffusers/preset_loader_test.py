import json
import os
from unittest import mock

import numpy as np
from safetensors.numpy import save_file

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.flux.flux_backbone import FluxBackbone
from keras_hub.src.models.flux2.flux2_backbone import Flux2Backbone
from keras_hub.src.models.flux2.flux2_text_to_image import Flux2TextToImage
from keras_hub.src.models.flux2.flux2_vae import Flux2VAE
from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from keras_hub.src.models.text_to_image import TextToImage
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils import preset_utils
from keras_hub.src.utils.diffusers import convert_flux2
from keras_hub.src.utils.diffusers.convert_flux2_test import (
    write_tiny_tokenizer,
)
from keras_hub.src.utils.diffusers.preset_loader import (
    DiffusersSafetensorLoader,
)
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

FLUX_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "attention_head_dim": 8,
    "axes_dims_rope": [2, 2, 4],
    "guidance_embeds": False,
    "in_channels": 8,
    "joint_attention_dim": 12,
    "num_attention_heads": 2,
    "num_layers": 1,
    "num_single_layers": 1,
    "patch_size": 1,
    "pooled_projection_dim": 10,
}
FLUX2_TRANSFORMER_CONFIG = {
    "_class_name": "Flux2Transformer2DModel",
    "attention_head_dim": 8,
    "axes_dims_rope": [2, 2, 2, 2],
    "eps": 1e-6,
    "guidance_embeds": False,
    "in_channels": 16,
    # Three Qwen3 hidden states of width 16.
    "joint_attention_dim": 48,
    "mlp_ratio": 3.0,
    "num_attention_heads": 2,
    "num_layers": 1,
    "num_single_layers": 1,
    "out_channels": None,
    "patch_size": 1,
    "rope_theta": 2000,
    "timestep_guidance_channels": 256,
}
QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "head_dim": 8,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    # `Flux2TextToImage` reads hidden states 9, 18 and 27.
    "num_hidden_layers": 27,
    "num_key_value_heads": 1,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "use_sliding_window": False,
}
FLUX2_VAE_CONFIG = {
    "_class_name": "AutoencoderKLFlux2",
    "act_fn": "silu",
    "batch_norm_eps": 1e-4,
    "block_out_channels": [32, 64],
    "in_channels": 3,
    "latent_channels": 4,
    "layers_per_block": 1,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "patch_size": [2, 2],
    "use_post_quant_conv": True,
    "use_quant_conv": True,
}
# `black-forest-labs/FLUX.2-klein-4B/model_index.json`.
FLUX2_MODEL_INDEX = {
    "_class_name": "Flux2KleinPipeline",
    "is_distilled": True,
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3ForCausalLM"],
    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    "transformer": ["diffusers", "Flux2Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}


def write_json(preset_dir, path, content):
    path = os.path.join(preset_dir, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(content, f)


def write_flux2_repo(preset_dir, model_index):
    """Write the configs of a tiny FLUX.2 pipeline repo, without weights."""
    write_json(preset_dir, "model_index.json", model_index)
    write_json(preset_dir, "transformer/config.json", FLUX2_TRANSFORMER_CONFIG)
    write_json(preset_dir, "vae/config.json", FLUX2_VAE_CONFIG)
    vocabulary_size = write_tiny_tokenizer(preset_dir)
    write_json(
        preset_dir,
        "text_encoder/config.json",
        {**QWEN3_CONFIG, "vocab_size": vocabulary_size},
    )


class DiffusersSafetensorLoaderTest(TestCase):
    def setUp(self):
        super().setUp()
        self.preset_dir = self.get_temp_dir()
        self.kernel = np.arange(6, dtype="float32").reshape(2, 3)
        os.makedirs(os.path.join(self.preset_dir, "transformer"))
        save_file(
            {"proj_out.weight": self.kernel},
            os.path.join(
                self.preset_dir,
                "transformer",
                "diffusion_pytorch_model.safetensors",
            ),
        )

    def test_single_file(self):
        with DiffusersSafetensorLoader(
            self.preset_dir, "transformer"
        ) as loader:
            self.assertAllClose(
                loader.get_tensor("proj_out.weight"), self.kernel
            )

    def test_sharded(self):
        # A transformers model, like the Qwen3 `text_encoder/` of FLUX.2,
        # with its `model.` prefix.
        folder = os.path.join(self.preset_dir, "text_encoder")
        os.makedirs(folder)
        save_file(
            {"model.embed_tokens.weight": self.kernel},
            os.path.join(folder, "model-00001-of-00002.safetensors"),
        )
        save_file(
            {"model.norm.weight": self.kernel[0]},
            os.path.join(folder, "model-00002-of-00002.safetensors"),
        )
        write_json(
            self.preset_dir,
            "text_encoder/model.safetensors.index.json",
            {
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": (
                        "model-00001-of-00002.safetensors"
                    ),
                    "model.norm.weight": "model-00002-of-00002.safetensors",
                },
            },
        )
        with DiffusersSafetensorLoader(
            self.preset_dir, "text_encoder"
        ) as loader:
            self.assertAllClose(
                loader.get_tensor("embed_tokens.weight"), self.kernel
            )
            self.assertAllClose(
                loader.get_tensor("norm.weight"), self.kernel[0]
            )

    def test_ignores_root_safetensors_index(self):
        # A root index must not take precedence over the folder's weights.
        write_json(
            self.preset_dir,
            "model.safetensors.index.json",
            {"weight_map": {"other.weight": "model.safetensors"}},
        )
        with DiffusersSafetensorLoader(
            self.preset_dir, "transformer"
        ) as loader:
            self.assertIsNone(loader.safetensor_config)
            self.assertAllClose(
                loader.get_tensor("proj_out.weight"), self.kernel
            )

    def test_only_probes_subfolder(self):
        # `check_file_exists` and `load_json` both go through `get_file`.
        with mock.patch.object(
            preset_utils, "get_file", wraps=preset_utils.get_file
        ) as get_file:
            with DiffusersSafetensorLoader(self.preset_dir, "transformer"):
                pass
        paths = [call.args[1] for call in get_file.call_args_list]
        self.assertNotEmpty(paths)
        for path in paths:
            self.assertStartsWith(path, "transformer/")

    def test_missing_weights_raise(self):
        with self.assertRaisesRegex(FileNotFoundError, "`vae/`"):
            DiffusersSafetensorLoader(self.preset_dir, "vae")

    def test_sets_all_safetensor_loader_attributes(self):
        # `SafetensorLoader.__init__` is skipped, so every attribute it sets
        # (and its methods read) must be set here too.
        root_preset_dir = os.path.join(self.get_temp_dir(), "transformers")
        os.makedirs(root_preset_dir)
        save_file(
            {"proj_out.weight": self.kernel},
            os.path.join(root_preset_dir, "model.safetensors"),
        )
        with SafetensorLoader(root_preset_dir) as reference:
            with DiffusersSafetensorLoader(
                self.preset_dir, "transformer"
            ) as loader:
                self.assertEmpty(set(vars(reference)) - set(vars(loader)))


class DiffusersPresetLoaderTest(TestCase):
    def test_unet_pipeline_raises(self):
        preset_dir = self.get_temp_dir()
        write_json(
            preset_dir,
            "model_index.json",
            {
                "_class_name": "StableDiffusionPipeline",
                "safety_checker": [None, None],
                "unet": ["diffusers", "UNet2DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            },
        )
        write_json(
            preset_dir,
            "unet/config.json",
            {"_class_name": "UNet2DConditionModel"},
        )
        with self.assertRaisesRegex(
            ValueError,
            "`StableDiffusionPipeline`.*`transformer` component.*"
            r"\['unet', 'vae'\]",
        ):
            Backbone.from_preset(preset_dir)

    def test_null_transformer_raises(self):
        preset_dir = self.get_temp_dir()
        write_json(
            preset_dir,
            "model_index.json",
            {"_class_name": "FluxPipeline", "transformer": [None, None]},
        )
        with self.assertRaisesRegex(ValueError, "`transformer` component"):
            Backbone.from_preset(preset_dir)

    def test_unsupported_transformer_raises(self):
        preset_dir = self.get_temp_dir()
        write_json(
            preset_dir,
            "model_index.json",
            {
                "_class_name": "StableDiffusion3Pipeline",
                "transformer": ["diffusers", "SD3Transformer2DModel"],
            },
        )
        write_json(
            preset_dir,
            "transformer/config.json",
            {"_class_name": "SD3Transformer2DModel"},
        )
        with self.assertRaisesRegex(ValueError, "SD3Transformer2DModel"):
            Backbone.from_preset(preset_dir)

    def test_missing_or_null_component_raises(self):
        without_text_encoder = dict(FLUX2_MODEL_INDEX)
        del without_text_encoder["text_encoder"]
        for i, model_index in enumerate(
            (
                without_text_encoder,
                {**FLUX2_MODEL_INDEX, "text_encoder": None},
                {**FLUX2_MODEL_INDEX, "text_encoder": [None, None]},
            )
        ):
            with self.subTest(model_index=model_index):
                # `get_temp_dir()` is the same for the whole test.
                preset_dir = os.path.join(self.get_temp_dir(), f"repo_{i}")
                write_flux2_repo(preset_dir, model_index)
                with self.assertRaisesRegex(
                    ValueError,
                    "`Qwen3ForCausalLM` as the `text_encoder`.*"
                    "no `text_encoder` component",
                ):
                    Flux2TextToImage.from_preset(preset_dir, load_weights=False)

    def test_wrong_component_class_raises(self):
        preset_dir = self.get_temp_dir()
        write_flux2_repo(
            preset_dir,
            {**FLUX2_MODEL_INDEX, "vae": ["diffusers", "AutoencoderKL"]},
        )
        loader = preset_utils.get_preset_loader(preset_dir)
        with self.assertRaisesRegex(
            ValueError, "`AutoencoderKLFlux2`.*Received: `AutoencoderKL`"
        ):
            loader.load_component_config("vae", "AutoencoderKLFlux2")

    def test_flux2_task_from_converter_hooks(self):
        preset_dir = self.get_temp_dir()
        write_flux2_repo(preset_dir, FLUX2_MODEL_INDEX)
        text_to_image = TextToImage.from_preset(
            preset_dir, load_weights=False, image_shape=(32, 32, 3)
        )
        self.assertIsInstance(text_to_image, Flux2TextToImage)
        self.assertIsInstance(text_to_image.backbone, Flux2Backbone)
        self.assertIsInstance(text_to_image.text_encoder, Qwen3Backbone)
        self.assertIsInstance(text_to_image.vae, Flux2VAE)
        self.assertIsInstance(
            text_to_image.preprocessor.tokenizer, Qwen3Tokenizer
        )
        # Read from `model_index.json`.
        self.assertTrue(text_to_image.is_distilled)

    def test_from_preset_kwargs_take_precedence(self):
        preset_dir = self.get_temp_dir()
        write_flux2_repo(preset_dir, FLUX2_MODEL_INDEX)
        vae = Flux2VAE(**convert_flux2.convert_vae_config(FLUX2_VAE_CONFIG))
        text_to_image = Flux2TextToImage.from_preset(
            preset_dir,
            load_weights=False,
            image_shape=(32, 32, 3),
            is_distilled=False,
            vae=vae,
        )
        self.assertFalse(text_to_image.is_distilled)
        self.assertIs(text_to_image.vae, vae)

    def test_converter_without_hooks_only_loads_the_backbone(self):
        # `convert_flux` defines neither `load_task_config` nor
        # `convert_tokenizer`.
        preset_dir = self.get_temp_dir()
        write_json(
            preset_dir,
            "model_index.json",
            {
                "_class_name": "FluxPipeline",
                "transformer": ["diffusers", "FluxTransformer2DModel"],
            },
        )
        write_json(
            preset_dir, "transformer/config.json", FLUX_TRANSFORMER_CONFIG
        )
        backbone = Backbone.from_preset(preset_dir, load_weights=False)
        self.assertIsInstance(backbone, FluxBackbone)
        with self.assertRaisesRegex(
            ValueError, "only load the backbone.*`FluxPipeline`"
        ):
            TextToImage.from_preset(preset_dir, load_weights=False)
        loader = preset_utils.get_preset_loader(preset_dir)
        with self.assertRaisesRegex(ValueError, "only load the backbone"):
            loader.load_tokenizer(Qwen3Tokenizer)
