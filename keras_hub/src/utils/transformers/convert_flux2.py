"""Convert FLUX.2 transformer checkpoints to `Flux2Backbone`.

Weights are read in the original Black Forest Labs single-file layout, as
shipped next to the diffusers folders on the Hub (e.g.
`black-forest-labs/FLUX.2-klein-4B/flux-2-klein-4b.safetensors`):
`img_in`, `txt_in`, `time_in`, the shared `double_stream_modulation_img`,
`double_stream_modulation_txt` and `single_stream_modulation` projections,
`double_blocks.*`, `single_blocks.*` and `final_layer`.
"""

import numpy as np

from keras_hub.src.models.flux2.flux2_backbone import Flux2Backbone

try:
    from safetensors import SafetensorError
except ImportError:  # safetensors is an optional dependency.
    SafetensorError = None

backbone_cls = Flux2Backbone

# Raised when a key is absent: `KeyError` from the sharded `weight_map`
# lookup, `SafetensorError` from a single-file `safe_open`.
_MISSING_TENSOR_ERRORS = tuple(
    error for error in (KeyError, SafetensorError) if error is not None
)


def _get_any(config, keys, default):
    """Return the value of the first key of `keys` present in `config`.

    FLUX.2 configs come in two spellings: diffusers'
    `transformer/config.json` (`num_layers`, `axes_dims_rope`,
    `joint_attention_dim`, `guidance_embeds`, ...) and the Black Forest Labs
    `Flux2Params` (`depth`, `axes_dim`, `context_in_dim`,
    `use_guidance_embed`, ...). Reading only one of them silently falls back
    to a default and builds a structurally wrong model.
    """
    for key in keys:
        if key in config:
            return config[key]
    return default


def convert_backbone_config(hf_config):
    """Convert a FLUX.2 transformer config to `Flux2Backbone` kwargs.

    Missing keys default to FLUX.2 [klein] 4B.
    """
    patch_size = _get_any(hf_config, ["patch_size"], 1)
    if patch_size != 1:
        raise ValueError(
            "FLUX.2 packs latents into 2x2 patches before the transformer, "
            "so the transformer itself must have `patch_size=1`. Received: "
            f"patch_size={patch_size}"
        )
    input_channels = _get_any(hf_config, ["in_channels"], 128)
    out_channels = hf_config.get("out_channels")
    if out_channels is not None and out_channels != input_channels:
        raise ValueError(
            "`Flux2Backbone` predicts a velocity with the same number of "
            "channels as its input. Received: "
            f"in_channels={input_channels}, out_channels={out_channels}"
        )

    num_heads = _get_any(hf_config, ["num_attention_heads", "num_heads"], 24)
    # diffusers expresses width as heads * head_dim rather than hidden_size.
    attention_head_dim = _get_any(hf_config, ["attention_head_dim"], None)
    if attention_head_dim is not None:
        hidden_size = num_heads * attention_head_dim
    else:
        hidden_size = _get_any(hf_config, ["hidden_size"], 3072)

    axes_dim = list(
        _get_any(hf_config, ["axes_dims_rope", "axes_dim"], [32, 32, 32, 32])
    )
    text_dim = _get_any(
        hf_config, ["joint_attention_dim", "context_in_dim"], 7680
    )
    return {
        "input_channels": input_channels,
        "hidden_size": hidden_size,
        "mlp_ratio": _get_any(hf_config, ["mlp_ratio"], 3.0),
        "num_heads": num_heads,
        "depth": _get_any(hf_config, ["num_layers", "depth"], 5),
        "depth_single_blocks": _get_any(
            hf_config, ["num_single_layers", "depth_single_blocks"], 20
        ),
        "axes_dim": axes_dim,
        "theta": _get_any(hf_config, ["rope_theta", "theta"], 2000),
        "guidance_embed": _get_any(
            hf_config,
            ["guidance_embeds", "use_guidance_embed", "guidance_embed"],
            False,
        ),
        # Sequence axes stay dynamic so a preset built from this config can
        # run at any resolution and prompt length.
        "image_shape": (None, input_channels),
        "text_shape": (None, text_dim),
        "image_ids_shape": (None, len(axes_dim)),
        "text_ids_shape": (None, len(axes_dim)),
    }


def _port_dense(loader, dense, prefix):
    """Port a bias-free PyTorch `nn.Linear` into a Keras `Dense`.

    PyTorch stores `[out_features, in_features]`; Keras wants
    `[in_features, out_features]`.
    """
    loader.port_weight(
        dense.kernel, f"{prefix}.weight", hook_fn=lambda x, _: x.T
    )


def _port_qk_norm(loader, qk_norm, prefix):
    loader.port_weight(qk_norm.query_norm.scale, f"{prefix}.query_norm.scale")
    loader.port_weight(qk_norm.key_norm.scale, f"{prefix}.key_norm.scale")


def _port_mlp_embedder(loader, embedder, prefix):
    _port_dense(loader, embedder.input_layer, f"{prefix}.in_layer")
    _port_dense(loader, embedder.output_layer, f"{prefix}.out_layer")


def _port_double_block(loader, block, prefix):
    for keras_stream, hf_stream in (("image", "img"), ("text", "txt")):
        _port_dense(
            loader,
            getattr(block, f"{keras_stream}_qkv"),
            f"{prefix}.{hf_stream}_attn.qkv",
        )
        _port_qk_norm(
            loader,
            getattr(block, f"{keras_stream}_attn_norm"),
            f"{prefix}.{hf_stream}_attn.norm",
        )
        _port_dense(
            loader,
            getattr(block, f"{keras_stream}_attn_proj"),
            f"{prefix}.{hf_stream}_attn.proj",
        )
        mlp = getattr(block, f"{keras_stream}_mlp")
        # `{img,txt}_mlp` is `Sequential(Linear, SiLUActivation, Linear)`.
        _port_dense(loader, mlp.input_layer, f"{prefix}.{hf_stream}_mlp.0")
        _port_dense(loader, mlp.output_layer, f"{prefix}.{hf_stream}_mlp.2")


def _port_single_block(loader, block, prefix):
    _port_dense(loader, block.linear1, f"{prefix}.linear1")
    _port_dense(loader, block.linear2, f"{prefix}.linear2")
    _port_qk_norm(loader, block.norm, f"{prefix}.norm")


def convert_weights(backbone, loader, hf_config):
    """Map FLUX.2 weights (Black Forest Labs layout) into a `Flux2Backbone`."""
    if not isinstance(backbone, Flux2Backbone):
        raise ValueError(
            "The provided backbone must be an instance of Flux2Backbone. "
            f"Received: {type(backbone)}"
        )

    # The shared modulation projections are what tell a FLUX.2 checkpoint
    # apart, so port them first and fail with a useful message if absent.
    try:
        _port_dense(
            loader,
            backbone.double_stream_modulation_image.linear_projection,
            "double_stream_modulation_img.lin",
        )
    except _MISSING_TENSOR_ERRORS as error:
        raise ValueError(
            "The checkpoint has no `double_stream_modulation_img.lin.weight`, "
            "so it is not a FLUX.2 checkpoint in the Black Forest Labs "
            "single-file layout. FLUX.1 checkpoints (per-block `img_mod` / "
            "`txt_mod` modulation) load with `FluxBackbone`. For a diffusers "
            "repo, convert its single-file checkpoint instead (e.g. "
            "`flux-2-klein-4b.safetensors` in "
            "`black-forest-labs/FLUX.2-klein-4B`)."
        ) from error
    _port_dense(
        loader,
        backbone.double_stream_modulation_text.linear_projection,
        "double_stream_modulation_txt.lin",
    )
    _port_dense(
        loader,
        backbone.single_stream_modulation.linear_projection,
        "single_stream_modulation.lin",
    )

    _port_dense(loader, backbone.image_input_embedder, "img_in")
    _port_dense(loader, backbone.text_input_embedder, "txt_in")
    _port_mlp_embedder(loader, backbone.time_input_embedder, "time_in")
    if backbone.guidance_embed:
        try:
            _port_mlp_embedder(
                loader, backbone.guidance_input_embedder, "guidance_in"
            )
        except _MISSING_TENSOR_ERRORS as error:
            raise ValueError(
                "The backbone was built with `guidance_embed=True`, but the "
                "checkpoint has no `guidance_in.*` weights. FLUX.2 [klein] "
                "is not guidance-distilled; only FLUX.2 [dev] is."
            ) from error

    # A missing block is a hard error: skipping it would leave randomly
    # initialized layers in a model that still runs.
    for i, block in enumerate(backbone.double_blocks):
        prefix = f"double_blocks.{i}"
        try:
            _port_double_block(loader, block, prefix)
        except _MISSING_TENSOR_ERRORS as error:
            raise ValueError(
                f"The checkpoint is missing weights for `{prefix}` ({error}). "
                f"Expected {len(backbone.double_blocks)} double-stream blocks."
            ) from error

    for i, block in enumerate(backbone.single_blocks):
        prefix = f"single_blocks.{i}"
        try:
            _port_single_block(loader, block, prefix)
        except _MISSING_TENSOR_ERRORS as error:
            raise ValueError(
                f"The checkpoint is missing weights for `{prefix}` ({error}). "
                f"Expected {len(backbone.single_blocks)} single-stream blocks."
            ) from error

    _port_dense(
        loader,
        backbone.final_layer.adaLN_modulation,
        "final_layer.adaLN_modulation.1",
    )
    _port_dense(loader, backbone.final_layer.linear, "final_layer.linear")


# === VAE (`AutoencoderKLFlux2`, diffusers layout) ===


def convert_vae_config(hf_vae_config):
    """Convert a diffusers `AutoencoderKLFlux2` config to `Flux2VAE` kwargs."""
    unsupported = {
        "norm_num_groups": (hf_vae_config.get("norm_num_groups", 32), 32),
        "act_fn": (hf_vae_config.get("act_fn", "silu"), "silu"),
        "mid_block_add_attention": (
            hf_vae_config.get("mid_block_add_attention", True),
            True,
        ),
        "use_quant_conv": (hf_vae_config.get("use_quant_conv", True), True),
        "use_post_quant_conv": (
            hf_vae_config.get("use_post_quant_conv", True),
            True,
        ),
    }
    for key, (value, expected) in unsupported.items():
        if value != expected:
            raise ValueError(
                f"`Flux2VAE` only supports `{key}={expected!r}`. "
                f"Received: {key}={value!r}"
            )
    patch_size = hf_vae_config.get("patch_size", [2, 2])
    if isinstance(patch_size, (list, tuple)):
        if len(set(patch_size)) != 1:
            raise ValueError(
                f"Only square patches are supported. Received: {patch_size}"
            )
        patch_size = patch_size[0]
    filters = list(
        hf_vae_config.get("block_out_channels", [128, 256, 512, 512])
    )
    decoder_filters = list(
        hf_vae_config.get("decoder_block_out_channels") or filters
    )
    layers_per_block = hf_vae_config.get("layers_per_block", 2)
    return {
        "encoder_num_filters": filters,
        "encoder_num_blocks": [layers_per_block] * len(filters),
        # diffusers lists decoder stages from the highest resolution;
        # `VAEDecoder` starts from the lowest.
        "decoder_num_filters": decoder_filters[::-1],
        # diffusers' decoder stages have one more ResNet block.
        "decoder_num_blocks": [layers_per_block + 1] * len(decoder_filters),
        "latent_channels": hf_vae_config.get("latent_channels", 32),
        "patch_size": patch_size,
        "batch_norm_epsilon": hf_vae_config.get("batch_norm_eps", 1e-4),
        "input_channels": hf_vae_config.get("in_channels", 3),
        "output_channels": hf_vae_config.get("out_channels", 3),
    }


def _port_conv2d(loader, conv, prefix):
    """PyTorch `[out, in, kh, kw]` -> Keras `[kh, kw, in, out]`."""
    loader.port_weight(
        conv.kernel,
        f"{prefix}.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    loader.port_weight(conv.bias, f"{prefix}.bias")


def _port_linear_as_conv2d(loader, conv, prefix):
    """diffusers' VAE attention uses `nn.Linear`; KerasHub uses 1x1 convs."""
    loader.port_weight(
        conv.kernel, f"{prefix}.weight", hook_fn=lambda x, _: x.T[None, None]
    )
    loader.port_weight(conv.bias, f"{prefix}.bias")


def _port_group_norm(loader, norm, prefix):
    loader.port_weight(norm.gamma, f"{prefix}.weight")
    loader.port_weight(norm.beta, f"{prefix}.bias")


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
    """Map diffusers `AutoencoderKLFlux2` weights into a `Flux2VAE`.

    Unlike the LDM single-file layout used for Stable Diffusion 3, diffusers
    numbers decoder stages from the lowest resolution (`up_blocks.0`), which
    is also `VAEDecoder`'s order, and its mid-block attention uses linear
    projections.
    """
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
    loader.port_weight(normalization.moving_mean, "bn.running_mean")
    loader.port_weight(normalization.moving_variance, "bn.running_var")
