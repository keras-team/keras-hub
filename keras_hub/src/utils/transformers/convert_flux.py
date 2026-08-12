import keras

from keras_hub.src.models.flux.flux_model import FluxBackbone

backbone_cls = FluxBackbone


def convert_backbone_config(hf_config):
    """Convert Hugging Face FLUX config to KerasHub FluxBackbone config."""

    hidden_size = hf_config.get("hidden_size", 3072)
    num_heads = hf_config.get("num_attention_heads", 24)
    depth = hf_config.get("num_hidden_layers", 19)
    depth_single_blocks = hf_config.get("num_single_layers", 38)

    axes_dim = hf_config.get(
        "axes_dim",
        [16, 56, 56],
    )

    return {
        # FLUX transformer parameters
        "input_channels": hf_config.get(
            "in_channels",
            64,
        ),
        "hidden_size": hidden_size,
        "mlp_ratio": hf_config.get(
            "mlp_ratio",
            4.0,
        ),
        "num_heads": num_heads,
        "depth": depth,
        "depth_single_blocks": depth_single_blocks,
        "axes_dim": axes_dim,
        "theta": hf_config.get(
            "theta",
            10000,
        ),
        "use_bias": hf_config.get(
            "use_bias",
            True,
        ),
        "guidance_embed": hf_config.get(
            "guidance_embed",
            False,
        ),
        # ---------------------------------------------------------------
        #
        # FLUX Schnell:
        #
        # image = latent sequence, feature dimension = 64
        # text  = T5/context sequence, feature dimension = 4096
        #
        # Do NOT use 3072 here.
        # ---------------------------------------------------------------
        "image_shape": (
            None,
            64,
        ),
        "text_shape": (
            None,
            4096,
        ),
        "image_ids_shape": (
            None,
            3,
        ),
        "text_ids_shape": (
            None,
            3,
        ),
        "y_shape": (768,),
    }


def convert_weights(backbone, loader, hf_config):
    """Map Hugging Face FLUX weights into KerasHub FluxBackbone."""

    print("Converting input embeddings...")

    if loader.has_tensor("img_in.weight"):
        backbone.image_input_embedder.set_weights(
            [
                loader.get_tensor("img_in.weight").T,
                loader.get_tensor("img_in.bias"),
            ]
        )

    if loader.has_tensor("txt_in.weight"):
        backbone.text_input_embedder.set_weights(
            [
                loader.get_tensor("txt_in.weight").T,
                loader.get_tensor("txt_in.bias"),
            ]
        )

    _convert_mlp_embedder(
        backbone.time_input_embedder,
        loader,
        prefix="time_in",
    )

    _convert_mlp_embedder(
        backbone.vector_embedder,
        loader,
        prefix="vector_in",
    )

    if getattr(backbone, "guidance_embed", False) and hasattr(
        backbone,
        "guidance_input_embedder",
    ):
        _convert_mlp_embedder(
            backbone.guidance_input_embedder,
            loader,
            prefix="guidance_in",
        )

    print("Converting double-stream blocks...")

    for i, block in enumerate(backbone.double_blocks):
        prefix = f"double_blocks.{i}"

        if not loader.has_tensor(f"{prefix}.img_attn.qkv.weight"):
            print(f"Skipping double block {i}")
            continue

        print(f"  double block {i}")

        _convert_double_block(
            block,
            loader,
            prefix,
        )

    print("Converting single-stream blocks...")

    for i, block in enumerate(backbone.single_blocks):
        prefix = f"single_blocks.{i}"

        if not loader.has_tensor(f"{prefix}.linear1.weight"):
            print(f"Skipping single block {i}")
            continue

        print(f"  single block {i}")

        _convert_single_block(
            block,
            loader,
            prefix,
        )

    print("Converting final layer...")

    if loader.has_tensor("final_layer.linear.weight"):
        backbone.final_layer.linear.set_weights(
            [
                loader.get_tensor("final_layer.linear.weight").T,
                loader.get_tensor("final_layer.linear.bias"),
            ]
        )

    if loader.has_tensor("final_layer.adaLN_modulation.1.weight"):
        _convert_modulation(
            backbone.final_layer.adaLN_modulation,
            loader,
            prefix="final_layer.adaLN_modulation",
        )

    print("Weight conversion finished.")


def _convert_modulation(mod_layer, loader, prefix):
    """
    Convert FLUX modulation / adaLN_modulation weights.

    Supports both:

    1. KerasHub Modulation-like layers:
           mod_layer.linear_projection

    2. Keras Sequential layers:
           Sequential(
               ...,
               Dense(...)
           )

    Hugging Face FLUX uses either:
        <prefix>.lin.weight
        <prefix>.lin.bias

    or:
        <prefix>.1.weight
        <prefix>.1.bias
    """

    weight_name = None
    bias_name = None

    # Custom Modulation / FLUX blocks
    if loader.has_tensor(f"{prefix}.lin.weight"):
        weight_name = f"{prefix}.lin.weight"
        bias_name = f"{prefix}.lin.bias"

    # Sequential adaLN_modulation
    elif loader.has_tensor(f"{prefix}.1.weight"):
        weight_name = f"{prefix}.1.weight"
        bias_name = f"{prefix}.1.bias"

    else:
        print(f"WARNING: No modulation weights found for {prefix}")
        return

    print(f"  Loading modulation: {weight_name}")

    weight = loader.get_tensor(weight_name)

    bias = loader.get_tensor(bias_name)

    # PyTorch Linear:
    #
    #     [out_features, in_features]
    #
    # Keras Dense:
    #
    #     [in_features, out_features]
    #
    weight = weight.T

    if hasattr(
        mod_layer,
        "linear_projection",
    ):
        mod_layer.linear_projection.set_weights(
            [
                weight,
                bias,
            ]
        )

        return

    if isinstance(
        mod_layer,
        keras.Sequential,
    ):
        # FLUX final_layer.adaLN_modulation is normally:
        #
        # Sequential(
        #     SiLU(),
        #     Dense(...)
        # )
        #
        # Therefore find the Dense layer rather than assuming
        # a particular layer index.

        for layer in mod_layer.layers:
            if isinstance(
                layer,
                keras.layers.Dense,
            ):
                layer.set_weights(
                    [
                        weight,
                        bias,
                    ]
                )

                return

        raise TypeError(
            f"Could not find a Dense layer inside "
            f"{prefix} Sequential modulation layer. "
            f"Layers: {mod_layer.layers}"
        )

    raise TypeError(
        f"Unsupported modulation layer type for {prefix}: {type(mod_layer)}"
    )


def _convert_mlp_embedder(
    embedder_layer,
    loader,
    prefix,
):
    """Convert FLUX MLPEmbedder."""

    weight_name = f"{prefix}.in_layer.weight"

    bias_name = f"{prefix}.in_layer.bias"

    if loader.has_tensor(weight_name):
        embedder_layer.input_layer.set_weights(
            [
                loader.get_tensor(weight_name).T,
                loader.get_tensor(bias_name),
            ]
        )

    weight_name = f"{prefix}.out_layer.weight"

    bias_name = f"{prefix}.out_layer.bias"

    if loader.has_tensor(weight_name):
        embedder_layer.output_layer.set_weights(
            [
                loader.get_tensor(weight_name).T,
                loader.get_tensor(bias_name),
            ]
        )


def _convert_double_block(
    block,
    loader,
    prefix,
):
    """Convert a FLUX double-stream block."""

    _convert_modulation(
        block.image_mod,
        loader,
        f"{prefix}.img_mod",
    )

    _convert_modulation(
        block.text_mod,
        loader,
        f"{prefix}.txt_mod",
    )

    block.image_qkv.set_weights(
        [
            loader.get_tensor(f"{prefix}.img_attn.qkv.weight").T,
            loader.get_tensor(f"{prefix}.img_attn.qkv.bias"),
        ]
    )

    block.image_attn_proj.set_weights(
        [
            loader.get_tensor(f"{prefix}.img_attn.proj.weight").T,
            loader.get_tensor(f"{prefix}.img_attn.proj.bias"),
        ]
    )

    block.text_qkv.set_weights(
        [
            loader.get_tensor(f"{prefix}.txt_attn.qkv.weight").T,
            loader.get_tensor(f"{prefix}.txt_attn.qkv.bias"),
        ]
    )

    block.text_attn_proj.set_weights(
        [
            loader.get_tensor(f"{prefix}.txt_attn.proj.weight").T,
            loader.get_tensor(f"{prefix}.txt_attn.proj.bias"),
        ]
    )

    block.image_mlp.layers[0].set_weights(
        [
            loader.get_tensor(f"{prefix}.img_mlp.0.weight").T,
            loader.get_tensor(f"{prefix}.img_mlp.0.bias"),
        ]
    )

    block.image_mlp.layers[2].set_weights(
        [
            loader.get_tensor(f"{prefix}.img_mlp.2.weight").T,
            loader.get_tensor(f"{prefix}.img_mlp.2.bias"),
        ]
    )

    block.text_mlp.layers[0].set_weights(
        [
            loader.get_tensor(f"{prefix}.txt_mlp.0.weight").T,
            loader.get_tensor(f"{prefix}.txt_mlp.0.bias"),
        ]
    )

    block.text_mlp.layers[2].set_weights(
        [
            loader.get_tensor(f"{prefix}.txt_mlp.2.weight").T,
            loader.get_tensor(f"{prefix}.txt_mlp.2.bias"),
        ]
    )


def _convert_single_block(
    block,
    loader,
    prefix,
):
    """Convert a FLUX single-stream block."""
    _convert_modulation(
        block.modulation,
        loader,
        f"{prefix}.modulation",
    )

    block.linear1.set_weights(
        [
            loader.get_tensor(f"{prefix}.linear1.weight").T,
            loader.get_tensor(f"{prefix}.linear1.bias"),
        ]
    )

    block.linear2.set_weights(
        [
            loader.get_tensor(f"{prefix}.linear2.weight").T,
            loader.get_tensor(f"{prefix}.linear2.bias"),
        ]
    )
