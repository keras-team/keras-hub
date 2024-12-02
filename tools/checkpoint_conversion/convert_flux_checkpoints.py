import os

import keras
from safetensors import safe_open

from keras_hub.src.models.flux.flux_model import Flux

DOWNLOAD_URL = "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors"
keras.config.set_dtype_policy("mixed_float16")


def convert_mlpembedder_weights(weights_dict, keras_model, prefix):
    in_layer_weight = weights_dict[f"{prefix}.in_layer.weight"].T
    in_layer_bias = weights_dict[f"{prefix}.in_layer.bias"]

    out_layer_weight = weights_dict[f"{prefix}.out_layer.weight"].T
    out_layer_bias = weights_dict[f"{prefix}.out_layer.bias"]

    keras_model.input_layer.set_weights([in_layer_weight, in_layer_bias])
    keras_model.output_layer.set_weights([out_layer_weight, out_layer_bias])


def convert_selfattention_weights(weights_dict, keras_model, prefix):
    qkv_weight = weights_dict[f"{prefix}.qkv.weight"].T
    qkv_bias = weights_dict.get(f"{prefix}.qkv.bias")

    proj_weight = weights_dict[f"{prefix}.proj.weight"].T
    proj_bias = weights_dict[f"{prefix}.proj.bias"]

    keras_model.qkv.set_weights(
        [qkv_weight] + ([qkv_bias] if qkv_bias is not None else [])
    )
    keras_model.proj.set_weights([proj_weight, proj_bias])


def convert_modulation_weights(weights_dict, keras_model, prefix):
    lin_weight = weights_dict[f"{prefix}.lin.weight"].T
    lin_bias = weights_dict[f"{prefix}.lin.bias"]

    keras_model.linear_projection.set_weights([lin_weight, lin_bias])


def convert_doublestreamblock_weights(weights_dict, keras_model, block_idx):
    # Convert img_mod weights
    convert_modulation_weights(
        weights_dict,
        keras_model.image_mod,
        f"double_blocks.{block_idx}.img_mod",
    )

    # Convert txt_mod weights
    convert_modulation_weights(
        weights_dict, keras_model.text_mod, f"double_blocks.{block_idx}.txt_mod"
    )

    # Convert img_attn weights
    convert_selfattention_weights(
        weights_dict,
        keras_model.image_attn,
        f"double_blocks.{block_idx}.img_attn",
    )

    # Convert txt_attn weights
    convert_selfattention_weights(
        weights_dict,
        keras_model.text_attention,
        f"double_blocks.{block_idx}.txt_attn",
    )

    # Convert img_mlp weights (2 layers)
    keras_model.image_mlp.layers[0].set_weights(
        [
            weights_dict[f"double_blocks.{block_idx}.img_mlp.0.weight"].T,
            weights_dict[f"double_blocks.{block_idx}.img_mlp.0.bias"],
        ]
    )
    keras_model.image_mlp.layers[2].set_weights(
        [
            weights_dict[f"double_blocks.{block_idx}.img_mlp.2.weight"].T,
            weights_dict[f"double_blocks.{block_idx}.img_mlp.2.bias"],
        ]
    )

    # Convert txt_mlp weights (2 layers)
    keras_model.text_mlp.layers[0].set_weights(
        [
            weights_dict[f"double_blocks.{block_idx}.txt_mlp.0.weight"].T,
            weights_dict[f"double_blocks.{block_idx}.txt_mlp.0.bias"],
        ]
    )
    keras_model.text_mlp.layers[2].set_weights(
        [
            weights_dict[f"double_blocks.{block_idx}.txt_mlp.2.weight"].T,
            weights_dict[f"double_blocks.{block_idx}.txt_mlp.2.bias"],
        ]
    )


def convert_singlestreamblock_weights(weights_dict, keras_model, block_idx):
    convert_modulation_weights(
        weights_dict,
        keras_model.modulation,
        f"single_blocks.{block_idx}.modulation",
    )

    # Convert linear1 weights
    keras_model.linear1.set_weights(
        [
            weights_dict[f"single_blocks.{block_idx}.linear1.weight"].T,
            weights_dict[f"single_blocks.{block_idx}.linear1.bias"],
        ]
    )

    # Convert linear2 weights
    keras_model.linear2.set_weights(
        [
            weights_dict[f"single_blocks.{block_idx}.linear2.weight"].T,
            weights_dict[f"single_blocks.{block_idx}.linear2.bias"],
        ]
    )


def convert_lastlayer_weights(weights_dict, keras_model):
    # Convert linear weights
    keras_model.linear.set_weights(
        [
            weights_dict["final_layer.linear.weight"].T,
            weights_dict["final_layer.linear.bias"],
        ]
    )

    # Convert adaLN_modulation weights
    keras_model.adaLN_modulation.layers[1].set_weights(
        [
            weights_dict["final_layer.adaLN_modulation.1.weight"].T,
            weights_dict["final_layer.adaLN_modulation.1.bias"],
        ]
    )


def convert_flux_weights(weights_dict, keras_model):
    # Convert img_in weights
    keras_model.image_input_embedder.set_weights(
        [weights_dict["img_in.weight"].T, weights_dict["img_in.bias"]]
    )

    # Convert time_in weights (MLPEmbedder)
    convert_mlpembedder_weights(
        weights_dict, keras_model.time_input_embedder, "time_in"
    )

    # Convert vector_in weights (MLPEmbedder)
    convert_mlpembedder_weights(
        weights_dict, keras_model.vector_embedder, "vector_in"
    )

    # Convert guidance_in weights (if present)
    if hasattr(keras_model, "guidance_embed"):
        convert_mlpembedder_weights(
            weights_dict, keras_model.guidance_input_embedder, "guidance_in"
        )

    # Convert txt_in weights
    keras_model.text_input_embedder.set_weights(
        [weights_dict["txt_in.weight"].T, weights_dict["txt_in.bias"]]
    )

    # Convert double_blocks weights
    for block_idx in range(len(keras_model.double_blocks)):
        convert_doublestreamblock_weights(
            weights_dict, keras_model.double_blocks[block_idx], block_idx
        )

    # Convert single_blocks weights
    for block_idx in range(len(keras_model.single_blocks)):
        convert_singlestreamblock_weights(
            weights_dict, keras_model.single_blocks[block_idx], block_idx
        )

    # Convert final_layer weights
    convert_lastlayer_weights(weights_dict, keras_model.final_layer)


def main(_):
    # get the original weights
    print("Downloading weights")

    os.system(f"wget {DOWNLOAD_URL}")

    flux_weights = {}
    with safe_open(
        "flux1-schnell.safetensors", framework="pt", device="cpu"
    ) as f:
        for key in f.keys():
            flux_weights[key] = f.get_tensor(key)

    keras_model = Flux(
        in_channels=64,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        use_bias=True,
        guidance_embed=False,
    )

    # Define input shapes
    img_shape = (1, 96, 64)
    txt_shape = (1, 96, 64)
    img_ids_shape = (1, 96, 3)
    txt_ids_shape = (1, 96, 3)
    timestep_shape = (32,)
    y_shape = (1, 64)
    guidance_shape = (32,)

    # Build the model
    keras_model.build(
        (
            img_shape,
            img_ids_shape,
            txt_shape,
            txt_ids_shape,
            timestep_shape,
            y_shape,
            guidance_shape,
        )
    )

    convert_flux_weights(flux_weights, keras_model)
    keras_model.save_to_preset("flux1-schnell")

    os.remove("flux1-schnell.safetensors")


if __name__ == "__main__":
    main()
