from keras_hub.src.models.flux.flux_backbone import FluxBackbone

try:
    from safetensors import SafetensorError
except ImportError:  # safetensors is an optional dependency.
    SafetensorError = None

backbone_cls = FluxBackbone

# Raised when a key is absent: `KeyError` from the sharded `weight_map`
# lookup, `SafetensorError` from a single-file `safe_open`.
_MISSING_TENSOR_ERRORS = tuple(
    error for error in (KeyError, SafetensorError) if error is not None
)


def _has_tensor(loader, hf_weight_key):
    """Return whether `hf_weight_key` exists in the checkpoint.

    `SafetensorLoader` intentionally has no existence check and is shared by
    every other converter, so the probe lives here instead.

    Note that a `TypeError` is deliberately *not* treated as "missing".
    `SafetensorLoader` opens files with `framework="np"`, and numpy has no
    bfloat16 dtype, so a present bf16 tensor raises `TypeError`. Swallowing
    that would report FLUX's real weights as absent and produce a badly
    misleading "checkpoint is missing" error.
    """
    # The offline conversion script passes a loader shim that answers this
    # directly (and can read bf16 via torch).
    probe = getattr(loader, "has_tensor", None)
    if probe is not None:
        return probe(hf_weight_key)

    saved_prefix = loader.prefix
    try:
        loader.get_tensor(hf_weight_key)
        return True
    except _MISSING_TENSOR_ERRORS:
        # `get_prefixed_key` caches a prefix and resets it to "" when a key
        # is not found, so an unsuccessful probe would otherwise corrupt
        # every lookup that follows it.
        loader.prefix = saved_prefix
        return False


def _get_any(config, keys, default):
    """Return the first present key in `config`, else `default`.

    FLUX configs appear in two flavours depending on where the checkpoint
    came from:

    - diffusers (`black-forest-labs/FLUX.1-*`, `transformer/config.json`),
      which uses names like `num_layers` and `axes_dims_rope`.
    - the original Black Forest Labs single-file layout, which uses
      `depth`, `axes_dim`, and so on.

    Looking up only one flavour silently falls through to the default and
    produces a structurally wrong model, so probe both.
    """
    for key in keys:
        if key in config:
            return config[key]
    return default


def convert_backbone_config(hf_config):
    """Convert a Hugging Face FLUX config to `FluxBackbone` kwargs."""
    num_heads = _get_any(hf_config, ["num_attention_heads"], 24)

    # diffusers expresses width as heads * head_dim rather than hidden_size.
    attention_head_dim = _get_any(hf_config, ["attention_head_dim"], None)
    if attention_head_dim is not None:
        hidden_size = num_heads * attention_head_dim
    else:
        hidden_size = _get_any(hf_config, ["hidden_size"], 3072)

    # `guidance_embeds` is the diffusers spelling. Getting this wrong means
    # FLUX.1-dev silently loads as a non-guidance-distilled model.
    guidance_embed = _get_any(
        hf_config, ["guidance_embeds", "guidance_embed"], False
    )

    input_channels = _get_any(hf_config, ["in_channels"], 64)

    return {
        "input_channels": input_channels,
        "hidden_size": hidden_size,
        "mlp_ratio": _get_any(hf_config, ["mlp_ratio"], 4.0),
        "num_heads": num_heads,
        "depth": _get_any(hf_config, ["num_layers", "depth"], 19),
        "depth_single_blocks": _get_any(
            hf_config, ["num_single_layers", "depth_single_blocks"], 38
        ),
        "axes_dim": _get_any(
            hf_config, ["axes_dims_rope", "axes_dim"], [16, 56, 56]
        ),
        "theta": _get_any(hf_config, ["rope_theta", "theta"], 10_000),
        "use_bias": _get_any(hf_config, ["qkv_bias", "use_bias"], True),
        "guidance_embed": guidance_embed,
        # Sequence axes stay dynamic so a preset built from this config can
        # run at any resolution / prompt length.
        "image_shape": (None, input_channels),
        "text_shape": (
            None,
            _get_any(hf_config, ["joint_attention_dim"], 4096),
        ),
        "image_ids_shape": (None, 3),
        "text_ids_shape": (None, 3),
        "y_shape": (_get_any(hf_config, ["pooled_projection_dim"], 768),),
    }


def _port_dense(loader, dense, prefix):
    """Port a PyTorch `nn.Linear` into a Keras `Dense`.

    PyTorch stores `[out_features, in_features]`; Keras wants
    `[in_features, out_features]`.
    """
    loader.port_weight(
        dense.kernel, f"{prefix}.weight", hook_fn=lambda x, _: x.T
    )
    if dense.bias is not None:
        loader.port_weight(dense.bias, f"{prefix}.bias")


def _port_qk_norm(loader, qk_norm, prefix):
    """Port the RMS scales FLUX applies to q and k before attention."""
    loader.port_weight(qk_norm.query_norm.scale, f"{prefix}.query_norm.scale")
    loader.port_weight(qk_norm.key_norm.scale, f"{prefix}.key_norm.scale")


def _port_modulation(loader, mod_layer, prefix):
    """Port a FLUX modulation / adaLN_modulation projection.

    Handles both `Modulation` (which exposes `linear_projection`) and the
    `Sequential(SiLU, Dense)` used by `LastLayer.adaLN_modulation`, and both
    the `<prefix>.lin.*` and `<prefix>.1.*` checkpoint spellings.
    """
    if _has_tensor(loader, f"{prefix}.lin.weight"):
        weight_prefix = f"{prefix}.lin"
    elif _has_tensor(loader, f"{prefix}.1.weight"):
        weight_prefix = f"{prefix}.1"
    else:
        raise ValueError(
            f"Could not find modulation weights for `{prefix}`. Tried "
            f"`{prefix}.lin.weight` and `{prefix}.1.weight`."
        )

    dense = getattr(mod_layer, "linear_projection", None)
    if dense is None:
        dense = next(
            (layer for layer in mod_layer.layers if hasattr(layer, "kernel")),
            None,
        )
    if dense is None:
        raise TypeError(
            f"Could not find a Dense layer for modulation `{prefix}` on "
            f"{type(mod_layer)}."
        )

    _port_dense(loader, dense, weight_prefix)


def _port_mlp_embedder(loader, embedder, prefix):
    _port_dense(loader, embedder.input_layer, f"{prefix}.in_layer")
    _port_dense(loader, embedder.output_layer, f"{prefix}.out_layer")


def _port_double_block(loader, block, prefix):
    _port_modulation(loader, block.image_mod, f"{prefix}.img_mod")
    _port_modulation(loader, block.text_mod, f"{prefix}.txt_mod")

    _port_dense(loader, block.image_qkv, f"{prefix}.img_attn.qkv")
    _port_qk_norm(loader, block.image_attn_norm, f"{prefix}.img_attn.norm")
    _port_dense(loader, block.image_attn_proj, f"{prefix}.img_attn.proj")

    _port_dense(loader, block.text_qkv, f"{prefix}.txt_attn.qkv")
    _port_qk_norm(loader, block.text_attn_norm, f"{prefix}.txt_attn.norm")
    _port_dense(loader, block.text_attn_proj, f"{prefix}.txt_attn.proj")

    _port_dense(loader, block.image_mlp.layers[0], f"{prefix}.img_mlp.0")
    _port_dense(loader, block.image_mlp.layers[2], f"{prefix}.img_mlp.2")
    _port_dense(loader, block.text_mlp.layers[0], f"{prefix}.txt_mlp.0")
    _port_dense(loader, block.text_mlp.layers[2], f"{prefix}.txt_mlp.2")


def _port_single_block(loader, block, prefix):
    _port_modulation(loader, block.modulation, f"{prefix}.modulation")
    _port_dense(loader, block.linear1, f"{prefix}.linear1")
    _port_dense(loader, block.linear2, f"{prefix}.linear2")
    _port_qk_norm(loader, block.norm, f"{prefix}.norm")


def convert_weights(backbone, loader, hf_config):
    """Map Hugging Face FLUX weights into a `FluxBackbone`.

    Weight names follow the original Black Forest Labs layout
    (`img_in`, `double_blocks.*`, `single_blocks.*`, `final_layer`).
    """
    if not isinstance(backbone, FluxBackbone):
        raise ValueError(
            "The provided backbone must be an instance of FluxBackbone. "
            f"Received: {type(backbone)}"
        )

    _port_dense(loader, backbone.image_input_embedder, "img_in")
    _port_dense(loader, backbone.text_input_embedder, "txt_in")
    _port_mlp_embedder(loader, backbone.time_input_embedder, "time_in")
    _port_mlp_embedder(loader, backbone.vector_embedder, "vector_in")

    if backbone.guidance_embed:
        if not _has_tensor(loader, "guidance_in.in_layer.weight"):
            raise ValueError(
                "Backbone was built with `guidance_embed=True` but the "
                "checkpoint has no `guidance_in.*` weights. This usually "
                "means a FLUX.1-schnell checkpoint was paired with a "
                "FLUX.1-dev config."
            )
        _port_mlp_embedder(
            loader, backbone.guidance_input_embedder, "guidance_in"
        )

    # Missing blocks are a hard error. Skipping them silently produces a model
    # with randomly initialized transformer layers that still runs.
    #
    # This is checked by attempting the port and annotating any missing-key
    # failure, rather than by probing first: a probe would have to read a
    # full QKV tensor for every one of the ~57 blocks purely to test
    # presence, which is gigabytes of redundant I/O on a real checkpoint.
    for i, block in enumerate(backbone.double_blocks):
        prefix = f"double_blocks.{i}"
        try:
            _port_double_block(loader, block, prefix)
        except _MISSING_TENSOR_ERRORS as error:
            raise ValueError(
                f"Checkpoint is missing weights for `{prefix}` "
                f"({error}). Expected {len(backbone.double_blocks)} "
                "double-stream blocks."
            ) from error

    for i, block in enumerate(backbone.single_blocks):
        prefix = f"single_blocks.{i}"
        try:
            _port_single_block(loader, block, prefix)
        except _MISSING_TENSOR_ERRORS as error:
            raise ValueError(
                f"Checkpoint is missing weights for `{prefix}` "
                f"({error}). Expected {len(backbone.single_blocks)} "
                "single-stream blocks."
            ) from error

    _port_dense(loader, backbone.final_layer.linear, "final_layer.linear")
    _port_modulation(
        loader,
        backbone.final_layer.adaLN_modulation,
        "final_layer.adaLN_modulation",
    )
