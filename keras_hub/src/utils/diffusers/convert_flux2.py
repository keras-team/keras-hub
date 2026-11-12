"""Convert diffusers FLUX.2 checkpoints to `Flux2Backbone` and `Flux2VAE`.

The weights are read from the `transformer/` and `vae/` folders of a
diffusers pipeline repo, such as `hf://black-forest-labs/FLUX.2-klein-4B`,
through `DiffusersPresetLoader`. Only the key maps that differ from FLUX.1
live here: the shared modulation projections, the fused single-block
projection, the SwiGLU feed-forward layers, the missing biases and the VAE.

The `load_task_config` and `convert_tokenizer` hooks load the rest of a
`Flux2KleinPipeline` (the Qwen3 text encoder and tokenizer, and the VAE) for
`Flux2TextToImage`.
"""

import functools
import os

import numpy as np

from keras_hub.src.models.flux2.flux2_backbone import Flux2Backbone
from keras_hub.src.models.flux2.flux2_vae import Flux2VAE
from keras_hub.src.utils.diffusers.convert_flux import CheckedLoader
from keras_hub.src.utils.diffusers.convert_flux import port_dense
from keras_hub.src.utils.diffusers.convert_flux import port_fused_dense
from keras_hub.src.utils.diffusers.convert_flux import port_mlp_embedder
from keras_hub.src.utils.diffusers.convert_flux import port_norm_out
from keras_hub.src.utils.diffusers.convert_flux import port_qk_norm
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.transformers import convert_qwen3

backbone_cls = Flux2Backbone


def convert_backbone_config(hf_config):
    """Convert a diffusers `Flux2Transformer2DModel` config to kwargs.

    Structural keys are read directly, so a missing one fails here, with its
    name, instead of building a model of the wrong shape.
    """
    if hf_config["patch_size"] != 1:
        raise ValueError(
            "FLUX.2 packs latents into 2x2 patches before the transformer, "
            "so the transformer itself must have `patch_size=1`. Received: "
            f"patch_size={hf_config['patch_size']}"
        )
    input_channels = hf_config["in_channels"]
    # As in diffusers, `out_channels=None` means `in_channels`.
    out_channels = hf_config.get("out_channels") or input_channels
    if out_channels != input_channels:
        raise ValueError(
            "`Flux2Backbone` predicts a velocity with the same number of "
            "channels as its input. Received: "
            f"in_channels={input_channels}, out_channels={out_channels}"
        )
    for key, supported in (("eps", 1e-6), ("timestep_guidance_channels", 256)):
        if hf_config[key] != supported:
            raise ValueError(
                f"`Flux2Backbone` only supports `{key}={supported}`. "
                f"Received: {key}={hf_config[key]}"
            )
    num_heads = hf_config["num_attention_heads"]
    axes_dim = list(hf_config["axes_dims_rope"])
    return {
        "input_channels": input_channels,
        "hidden_size": num_heads * hf_config["attention_head_dim"],
        "mlp_ratio": hf_config["mlp_ratio"],
        "num_heads": num_heads,
        "depth": hf_config["num_layers"],
        "depth_single_blocks": hf_config["num_single_layers"],
        "axes_dim": axes_dim,
        "theta": hf_config["rope_theta"],
        "guidance_embed": hf_config["guidance_embeds"],
        # Sequence axes stay dynamic so a preset built from this config can
        # run at any resolution and prompt length.
        "latent_shape": (None, input_channels),
        "text_shape": (None, hf_config["joint_attention_dim"]),
        "image_ids_shape": (None, len(axes_dim)),
        "text_ids_shape": (None, len(axes_dim)),
    }


def _port_double_block(loader, block, prefix):
    attn = f"{prefix}.attn"
    port_fused_dense(
        loader,
        block.image_qkv,
        [f"{attn}.to_q", f"{attn}.to_k", f"{attn}.to_v"],
    )
    port_fused_dense(
        loader,
        block.text_qkv,
        [f"{attn}.add_q_proj", f"{attn}.add_k_proj", f"{attn}.add_v_proj"],
    )
    port_qk_norm(
        loader, block.image_attn_norm, f"{attn}.norm_q", f"{attn}.norm_k"
    )
    port_qk_norm(
        loader,
        block.text_attn_norm,
        f"{attn}.norm_added_q",
        f"{attn}.norm_added_k",
    )
    port_dense(loader, block.image_attn_proj, f"{attn}.to_out.0")
    port_dense(loader, block.text_attn_proj, f"{attn}.to_add_out")
    # The gate and up projections of the SwiGLU are fused in `linear_in`.
    port_dense(loader, block.image_mlp.input_layer, f"{prefix}.ff.linear_in")
    port_dense(loader, block.image_mlp.output_layer, f"{prefix}.ff.linear_out")
    port_dense(
        loader, block.text_mlp.input_layer, f"{prefix}.ff_context.linear_in"
    )
    port_dense(
        loader, block.text_mlp.output_layer, f"{prefix}.ff_context.linear_out"
    )


def _port_single_block(loader, block, prefix):
    attn = f"{prefix}.attn"
    # `to_qkv_mlp_proj` is `[q, k, v, mlp_gate, mlp_up]`, like `linear1`.
    port_dense(loader, block.linear1, f"{attn}.to_qkv_mlp_proj")
    port_qk_norm(loader, block.norm, f"{attn}.norm_q", f"{attn}.norm_k")
    port_dense(loader, block.linear2, f"{attn}.to_out")


def convert_weights(backbone, loader):
    """Port diffusers `Flux2Transformer2DModel` weights into `backbone`.

    Fails unless every backbone weight is assigned exactly once and every
    checkpoint tensor is read.

    Args:
        backbone: A `Flux2Backbone`.
        loader: A `SafetensorLoader` over the diffusers `transformer/`
            weights.
    """
    if not isinstance(backbone, Flux2Backbone):
        raise ValueError(
            "The provided backbone must be an instance of Flux2Backbone. "
            f"Received: {type(backbone)}"
        )
    loader = CheckedLoader(loader)
    port_dense(loader, backbone.image_input_embedder, "x_embedder")
    port_dense(loader, backbone.text_input_embedder, "context_embedder")
    port_mlp_embedder(
        loader,
        backbone.time_input_embedder,
        "time_guidance_embed.timestep_embedder",
    )
    if backbone.guidance_embed:
        port_mlp_embedder(
            loader,
            backbone.guidance_input_embedder,
            "time_guidance_embed.guidance_embedder",
        )
    # One modulation projection per stream, shared by all blocks.
    port_dense(
        loader,
        backbone.double_stream_modulation_image.linear_projection,
        "double_stream_modulation_img.linear",
    )
    port_dense(
        loader,
        backbone.double_stream_modulation_text.linear_projection,
        "double_stream_modulation_txt.linear",
    )
    port_dense(
        loader,
        backbone.single_stream_modulation.linear_projection,
        "single_stream_modulation.linear",
    )
    for i, block in enumerate(backbone.double_blocks):
        _port_double_block(loader, block, f"transformer_blocks.{i}")
    for i, block in enumerate(backbone.single_blocks):
        _port_single_block(loader, block, f"single_transformer_blocks.{i}")
    port_norm_out(
        loader, backbone.final_layer.adaLN_modulation, "norm_out.linear"
    )
    port_dense(loader, backbone.final_layer.linear, "proj_out")
    loader.check_complete(backbone)


# === VAE (`AutoencoderKLFlux2`) ===


def convert_vae_config(hf_vae_config):
    """Convert a diffusers `AutoencoderKLFlux2` config to `Flux2VAE` kwargs.

    Structural keys are read directly, so a missing one fails here, with its
    name.
    """
    supported = {
        "norm_num_groups": 32,
        "act_fn": "silu",
        "mid_block_add_attention": True,
        "use_quant_conv": True,
        "use_post_quant_conv": True,
    }
    for key, expected in supported.items():
        if hf_vae_config[key] != expected:
            raise ValueError(
                f"`Flux2VAE` only supports `{key}={expected!r}`. "
                f"Received: {key}={hf_vae_config[key]!r}"
            )
    patch_size = hf_vae_config["patch_size"]
    if isinstance(patch_size, (list, tuple)):
        if len(set(patch_size)) != 1:
            raise ValueError(
                f"Only square patches are supported. Received: {patch_size}"
            )
        patch_size = patch_size[0]
    filters = list(hf_vae_config["block_out_channels"])
    # As in diffusers, a missing or `None` value means `block_out_channels`.
    decoder_filters = list(
        hf_vae_config.get("decoder_block_out_channels") or filters
    )
    layers_per_block = hf_vae_config["layers_per_block"]
    return {
        "encoder_num_filters": filters,
        "encoder_num_blocks": [layers_per_block] * len(filters),
        # diffusers lists decoder stages from the highest resolution;
        # `VAEDecoder` starts from the lowest.
        "decoder_num_filters": decoder_filters[::-1],
        # diffusers' decoder stages have one more ResNet block.
        "decoder_num_blocks": [layers_per_block + 1] * len(decoder_filters),
        "latent_channels": hf_vae_config["latent_channels"],
        "patch_size": patch_size,
        "batch_norm_epsilon": hf_vae_config["batch_norm_eps"],
        "input_channels": hf_vae_config["in_channels"],
        "output_channels": hf_vae_config["out_channels"],
    }


def _port_conv2d(loader, conv, key):
    """PyTorch `[out, in, kh, kw]` -> Keras `[kh, kw, in, out]`."""
    kernel = np.transpose(loader.get_tensor(f"{key}.weight"), (2, 3, 1, 0))
    loader.assign(conv.kernel, kernel)
    loader.assign(conv.bias, loader.get_tensor(f"{key}.bias"))


def _port_linear_as_conv2d(loader, conv, key):
    """diffusers' VAE attention uses `nn.Linear`; KerasHub uses 1x1 convs."""
    loader.assign(conv.kernel, loader.get_tensor(f"{key}.weight").T[None, None])
    loader.assign(conv.bias, loader.get_tensor(f"{key}.bias"))


def _port_group_norm(loader, norm, key):
    loader.assign(norm.gamma, loader.get_tensor(f"{key}.weight"))
    loader.assign(norm.beta, loader.get_tensor(f"{key}.bias"))


def _port_vae_resnet(loader, block, prefix):
    _port_group_norm(loader, block.norm1, f"{prefix}.norm1")
    _port_conv2d(loader, block.conv1, f"{prefix}.conv1")
    _port_group_norm(loader, block.norm2, f"{prefix}.norm2")
    _port_conv2d(loader, block.conv2, f"{prefix}.conv2")
    if block.has_residual_projection:
        _port_conv2d(
            loader, block.residual_projection, f"{prefix}.conv_shortcut"
        )


def _port_vae_attention(loader, attention, prefix):
    _port_group_norm(loader, attention.group_norm, f"{prefix}.group_norm")
    _port_linear_as_conv2d(loader, attention.query_conv2d, f"{prefix}.to_q")
    _port_linear_as_conv2d(loader, attention.key_conv2d, f"{prefix}.to_k")
    _port_linear_as_conv2d(loader, attention.value_conv2d, f"{prefix}.to_v")
    _port_linear_as_conv2d(
        loader, attention.output_conv2d, f"{prefix}.to_out.0"
    )


def convert_vae_weights(vae, loader):
    """Port diffusers `AutoencoderKLFlux2` weights into a `Flux2VAE`.

    Unlike the LDM single-file layout used for Stable Diffusion 3, diffusers
    numbers decoder stages from the lowest resolution (`up_blocks.0`), which
    is also `VAEDecoder`'s order, and its mid-block attention uses linear
    projections. Fails unless every VAE weight is assigned exactly once and
    every checkpoint tensor is read.

    Args:
        vae: A `Flux2VAE`.
        loader: A `SafetensorLoader` over the diffusers `vae/` weights.
    """
    loader = CheckedLoader(loader)
    encoder = vae.encoder
    _port_conv2d(loader, encoder.input_projection, "encoder.conv_in")
    block_index = 0
    for i, _ in enumerate(encoder.stackwise_num_filters):
        for j in range(encoder.stackwise_num_blocks[i]):
            _port_vae_resnet(
                loader,
                encoder.blocks[block_index],
                f"encoder.down_blocks.{i}.resnets.{j}",
            )
            block_index += 1
        if i != len(encoder.stackwise_num_filters) - 1:
            # `downsamples` alternates `ZeroPadding2D` and `Conv2D`.
            _port_conv2d(
                loader,
                encoder.downsamples[2 * i + 1],
                f"encoder.down_blocks.{i}.downsamplers.0.conv",
            )
    _port_vae_resnet(loader, encoder.mid_block_0, "encoder.mid_block.resnets.0")
    _port_vae_attention(
        loader, encoder.mid_attention, "encoder.mid_block.attentions.0"
    )
    _port_vae_resnet(loader, encoder.mid_block_1, "encoder.mid_block.resnets.1")
    _port_group_norm(loader, encoder.output_norm, "encoder.conv_norm_out")
    _port_conv2d(loader, encoder.output_projection, "encoder.conv_out")

    _port_conv2d(loader, vae.quant_conv, "quant_conv")
    _port_conv2d(loader, vae.post_quant_conv, "post_quant_conv")

    decoder = vae.decoder
    _port_conv2d(loader, decoder.input_projection, "decoder.conv_in")
    _port_vae_resnet(loader, decoder.mid_block_0, "decoder.mid_block.resnets.0")
    _port_vae_attention(
        loader, decoder.mid_attention, "decoder.mid_block.attentions.0"
    )
    _port_vae_resnet(loader, decoder.mid_block_1, "decoder.mid_block.resnets.1")
    block_index = 0
    for i, _ in enumerate(decoder.stackwise_num_filters):
        for j in range(decoder.stackwise_num_blocks[i]):
            _port_vae_resnet(
                loader,
                decoder.blocks[block_index],
                f"decoder.up_blocks.{i}.resnets.{j}",
            )
            block_index += 1
        if i != len(decoder.stackwise_num_filters) - 1:
            # `upsamples` alternates `UpSampling2D` and `Conv2D`.
            _port_conv2d(
                loader,
                decoder.upsamples[2 * i + 1],
                f"decoder.up_blocks.{i}.upsamplers.0.conv",
            )
    _port_group_norm(loader, decoder.output_norm, "decoder.conv_norm_out")
    _port_conv2d(loader, decoder.output_projection, "decoder.conv_out")

    normalization = vae.latent_normalization
    loader.assign(
        normalization.moving_mean, loader.get_tensor("bn.running_mean")
    )
    loader.assign(
        normalization.moving_variance, loader.get_tensor("bn.running_var")
    )
    # `num_batches_tracked` is a training counter with no Keras equivalent.
    loader.check_complete(vae, ignore=["bn.num_batches_tracked"])


# === Pipeline (`Flux2KleinPipeline`) ===


def _load_text_encoder(loader, load_weights, dtype):
    """Build the Qwen3 text encoder from the `text_encoder/` folder."""
    hf_config = loader.load_component_config("text_encoder", "Qwen3ForCausalLM")
    text_encoder = convert_qwen3.backbone_cls(
        **convert_qwen3.convert_backbone_config(hf_config), dtype=dtype
    )
    if load_weights:
        loader.load_component_weights(
            "text_encoder",
            text_encoder,
            functools.partial(
                convert_qwen3.convert_weights, transformers_config=hf_config
            ),
        )
    return text_encoder


def _load_vae(loader, load_weights, dtype):
    """Build the VAE from the `vae/` folder."""
    hf_config = loader.load_component_config("vae", "AutoencoderKLFlux2")
    vae = Flux2VAE(**convert_vae_config(hf_config), dtype=dtype)
    if load_weights:
        loader.load_component_weights("vae", vae, convert_vae_weights)
    return vae


def load_task_config(loader, load_weights, **kwargs):
    """Load the `Flux2TextToImage` kwargs other than its backbone.

    Called by `DiffusersPresetLoader.load_task`. The text encoder and the VAE
    are built from their folders of the diffusers repo, unless they are
    passed to `from_preset()`.

    Args:
        loader: The `DiffusersPresetLoader` of the repo.
        load_weights: Whether to load the weights of the components.
        **kwargs: The kwargs passed to `from_preset()`.
    """
    dtype = kwargs.get("dtype")
    task_config = {
        # `Flux2KleinPipeline` defaults to `is_distilled=False` too.
        "is_distilled": loader.config.get("is_distilled", False),
    }
    if "text_encoder" not in kwargs:
        task_config["text_encoder"] = _load_text_encoder(
            loader, load_weights, dtype
        )
    if "vae" not in kwargs:
        task_config["vae"] = _load_vae(loader, load_weights, dtype)
    return task_config


def convert_tokenizer(cls, preset, **kwargs):
    """Load the Qwen3 tokenizer from the `tokenizer/` folder.

    Called by `DiffusersPresetLoader.load_tokenizer`.
    """
    # `get_file` fetches `tokenizer/tokenizer.json` (downloading it for
    # `hf://` presets); its local folder is a preset `convert_qwen3` reads.
    path = get_file(preset, "tokenizer/tokenizer.json")
    return convert_qwen3.convert_tokenizer(cls, os.path.dirname(path), **kwargs)
