import gc
import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import keras
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from keras_hub.src.models.flux.flux_maths import TimestepEmbedding
from keras_hub.src.models.flux.flux_model import FluxBackbone

REPO_ID = "black-forest-labs/FLUX.1-schnell"
CHECKPOINT_FILENAME = "flux1-schnell.safetensors"
OUTPUT_PRESET = "flux1-schnell"
CHECKPOINT_PATH = os.path.abspath(CHECKPOINT_FILENAME)

INPUT_CHANNELS = 64
HIDDEN_SIZE = 3072
MLP_RATIO = 4.0
NUM_HEADS = 24
DEPTH = 19
DEPTH_SINGLE_BLOCKS = 38
AXES_DIM = [16, 56, 56]
THETA = 10_000
USE_BIAS = True
GUIDANCE_EMBED = False

TEXT_EMBEDDING_DIM = 4096

Y_DIM = 768

BUILD_IMAGE_TOKENS = 4
BUILD_TEXT_TOKENS = 4

keras.config.set_dtype_policy("bfloat16")

_original_timestep_embedding_call = TimestepEmbedding.call


def _timestep_embedding_call_float32(self, t, dim=256):
    t = keras.ops.cast(t, "float32")
    return _original_timestep_embedding_call(self, t, dim)


TimestepEmbedding.call = _timestep_embedding_call_float32


def validate_safetensors_file(path):
    if not os.path.exists(path):
        return False

    file_size = os.path.getsize(path)
    if file_size < 1_000_000_000:
        return False

    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())

        if not keys:
            return False

        return True

    except Exception:
        return False


def download_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint already exists:\n  {CHECKPOINT_PATH}")

        if validate_safetensors_file(CHECKPOINT_PATH):
            return CHECKPOINT_PATH

        os.remove(CHECKPOINT_PATH)

    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=CHECKPOINT_FILENAME,
            local_dir=os.path.dirname(CHECKPOINT_PATH),
            local_dir_use_symlinks=False,
            token=True,
        )

    except Exception as exc:
        raise RuntimeError(
            "\n"
            "Failed to download FLUX.1 Schnell checkpoint.\n\n"
            "If Hugging Face returns 401/403, authenticate first:\n\n"
            "    hf auth login\n\n"
            "or:\n\n"
            "    huggingface-cli login\n"
        ) from exc

    downloaded_path = os.path.abspath(downloaded_path)

    if downloaded_path != CHECKPOINT_PATH:
        print("Copying checkpoint into working directory...")
        shutil.copy2(downloaded_path, CHECKPOINT_PATH)

    if not validate_safetensors_file(CHECKPOINT_PATH):
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

        raise RuntimeError(
            "Downloaded FLUX checkpoint is invalid or incomplete."
        )

    return CHECKPOINT_PATH


def load_checkpoint():
    weights = {}

    with safe_open(
        CHECKPOINT_PATH,
        framework="pt",
        device="cpu",
    ) as f:
        keys = list(f.keys())

        print(f"Checkpoint contains {len(keys)} tensors.")

        for index, key in enumerate(keys, start=1):
            tensor = f.get_tensor(key)

            if hasattr(tensor, "dtype"):
                if str(tensor.dtype) == "torch.bfloat16":
                    tensor = tensor.float()

                tensor = tensor.detach().cpu().numpy()

            weights[key] = tensor

            if index % 100 == 0 or index == len(keys):
                print(
                    f"  Loaded {index}/{len(keys)} tensors",
                    end="\r",
                )

    print()

    return weights


def convert_mlpembedder_weights(
    weights_dict,
    keras_model,
    prefix,
):
    in_layer_weight = f"{prefix}.in_layer.weight"
    in_layer_bias = f"{prefix}.in_layer.bias"

    out_layer_weight = f"{prefix}.out_layer.weight"
    out_layer_bias = f"{prefix}.out_layer.bias"

    for key in [
        in_layer_weight,
        in_layer_bias,
        out_layer_weight,
        out_layer_bias,
    ]:
        if key not in weights_dict:
            raise KeyError(f"Missing checkpoint tensor: {key}")

    keras_model.input_layer.set_weights(
        [
            weights_dict[in_layer_weight].T,
            weights_dict[in_layer_bias],
        ]
    )

    keras_model.output_layer.set_weights(
        [
            weights_dict[out_layer_weight].T,
            weights_dict[out_layer_bias],
        ]
    )


def convert_selfattention_weights(
    weights_dict,
    qkv_layer,
    proj_layer,
    prefix,
):
    qkv_weight = f"{prefix}.qkv.weight"
    qkv_bias = f"{prefix}.qkv.bias"

    proj_weight = f"{prefix}.proj.weight"
    proj_bias = f"{prefix}.proj.bias"

    if qkv_weight not in weights_dict:
        raise KeyError(f"Missing checkpoint tensor: {qkv_weight}")

    if proj_weight not in weights_dict:
        raise KeyError(f"Missing checkpoint tensor: {proj_weight}")

    if proj_bias not in weights_dict:
        raise KeyError(f"Missing checkpoint tensor: {proj_bias}")

    qkv_weights = [weights_dict[qkv_weight].T]

    if qkv_bias in weights_dict:
        qkv_weights.append(weights_dict[qkv_bias])

    qkv_layer.set_weights(qkv_weights)

    proj_layer.set_weights(
        [
            weights_dict[proj_weight].T,
            weights_dict[proj_bias],
        ]
    )


def convert_modulation_weights(
    weights_dict,
    keras_model,
    prefix,
):
    lin_weight = f"{prefix}.lin.weight"
    lin_bias = f"{prefix}.lin.bias"

    sequential_weight = f"{prefix}.1.weight"
    sequential_bias = f"{prefix}.1.bias"

    if lin_weight in weights_dict:
        weight_name = lin_weight
        bias_name = lin_bias

    elif sequential_weight in weights_dict:
        weight_name = sequential_weight
        bias_name = sequential_bias

    else:
        raise KeyError(f"Missing modulation weights for {prefix}")

    if bias_name not in weights_dict:
        raise KeyError(f"Missing checkpoint tensor: {bias_name}")

    weight = weights_dict[weight_name].T
    bias = weights_dict[bias_name]

    if hasattr(
        keras_model,
        "linear_projection",
    ):
        keras_model.linear_projection.set_weights(
            [
                weight,
                bias,
            ]
        )
        return

    if isinstance(
        keras_model,
        keras.Sequential,
    ):
        for layer in keras_model.layers:
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
            "Could not find a Dense layer inside "
            f"{prefix} Sequential modulation layer. "
            f"Layers: {keras_model.layers}"
        )

    raise TypeError(
        f"Unsupported modulation layer type for {prefix}: {type(keras_model)}"
    )


def convert_doublestreamblock_weights(
    weights_dict,
    keras_model,
    block_idx,
):
    prefix = f"double_blocks.{block_idx}"

    # Convert img_mod weights
    convert_modulation_weights(
        weights_dict,
        keras_model.image_mod,
        f"{prefix}.img_mod",
    )

    # Convert txt_mod weights
    convert_modulation_weights(
        weights_dict,
        keras_model.text_mod,
        f"{prefix}.txt_mod",
    )

    # Convert img_attn weights
    convert_selfattention_weights(
        weights_dict,
        keras_model.image_qkv,
        keras_model.image_attn_proj,
        f"{prefix}.img_attn",
    )

    # Convert txt_attn weights
    convert_selfattention_weights(
        weights_dict,
        keras_model.text_qkv,
        keras_model.text_attn_proj,
        f"{prefix}.txt_attn",
    )

    # Convert img_mlp weights (2 layers)
    keras_model.image_mlp.layers[0].set_weights(
        [
            weights_dict[f"{prefix}.img_mlp.0.weight"].T,
            weights_dict[f"{prefix}.img_mlp.0.bias"],
        ]
    )

    keras_model.image_mlp.layers[2].set_weights(
        [
            weights_dict[f"{prefix}.img_mlp.2.weight"].T,
            weights_dict[f"{prefix}.img_mlp.2.bias"],
        ]
    )

    # Convert txt_mlp weights (2 layers)
    keras_model.text_mlp.layers[0].set_weights(
        [
            weights_dict[f"{prefix}.txt_mlp.0.weight"].T,
            weights_dict[f"{prefix}.txt_mlp.0.bias"],
        ]
    )

    keras_model.text_mlp.layers[2].set_weights(
        [
            weights_dict[f"{prefix}.txt_mlp.2.weight"].T,
            weights_dict[f"{prefix}.txt_mlp.2.bias"],
        ]
    )


def convert_singlestreamblock_weights(
    weights_dict,
    keras_model,
    block_idx,
):
    prefix = f"single_blocks.{block_idx}"

    convert_modulation_weights(
        weights_dict,
        keras_model.modulation,
        f"{prefix}.modulation",
    )

    # Convert linear1 weights
    keras_model.linear1.set_weights(
        [
            weights_dict[f"{prefix}.linear1.weight"].T,
            weights_dict[f"{prefix}.linear1.bias"],
        ]
    )

    # Convert linear2 weights
    keras_model.linear2.set_weights(
        [
            weights_dict[f"{prefix}.linear2.weight"].T,
            weights_dict[f"{prefix}.linear2.bias"],
        ]
    )


def convert_lastlayer_weights(
    weights_dict,
    keras_model,
):
    # Convert linear weights
    keras_model.linear.set_weights(
        [
            weights_dict["final_layer.linear.weight"].T,
            weights_dict["final_layer.linear.bias"],
        ]
    )

    # Convert adaLN_modulation weights
    convert_modulation_weights(
        weights_dict,
        keras_model.adaLN_modulation,
        "final_layer.adaLN_modulation",
    )


def convert_flux_weights(
    weights_dict,
    keras_model,
):
    # Convert img_in weights
    keras_model.image_input_embedder.set_weights(
        [
            weights_dict["img_in.weight"].T,
            weights_dict["img_in.bias"],
        ]
    )

    # Convert time_in weights (MLPEmbedder)
    convert_mlpembedder_weights(
        weights_dict,
        keras_model.time_input_embedder,
        "time_in",
    )

    # Convert vector_in weights (MLPEmbedder)
    convert_mlpembedder_weights(
        weights_dict,
        keras_model.vector_embedder,
        "vector_in",
    )

    if keras_model.guidance_embed:
        raise RuntimeError(
            "The conversion model unexpectedly has "
            "guidance_embed=True. "
            "FLUX.1 Schnell must use "
            "guidance_embed=False."
        )

    # Convert txt_in weights

    keras_model.text_input_embedder.set_weights(
        [
            weights_dict["txt_in.weight"].T,
            weights_dict["txt_in.bias"],
        ]
    )

    total_double = len(keras_model.double_blocks)

    # Convert double_blocks weights
    for block_idx, block in enumerate(keras_model.double_blocks):
        print(f"  DoubleStreamBlock {block_idx + 1}/{total_double}")

        convert_doublestreamblock_weights(
            weights_dict,
            block,
            block_idx,
        )

    total_single = len(keras_model.single_blocks)

    # Convert single_blocks weights
    for block_idx, block in enumerate(keras_model.single_blocks):
        print(f"  SingleStreamBlock {block_idx + 1}/{total_single}")

        convert_singlestreamblock_weights(
            weights_dict,
            block,
            block_idx,
        )

    # Convert final_layer weights
    convert_lastlayer_weights(
        weights_dict,
        keras_model.final_layer,
    )


def main():
    # get the original weights
    print("Downloading weights")

    download_checkpoint()

    flux_weights = load_checkpoint()

    gc.collect()

    keras.backend.clear_session()

    keras.config.set_dtype_policy("bfloat16")

    TimestepEmbedding.call = _timestep_embedding_call_float32

    keras_model = FluxBackbone(
        input_channels=INPUT_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        mlp_ratio=MLP_RATIO,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        depth_single_blocks=DEPTH_SINGLE_BLOCKS,
        axes_dim=AXES_DIM,
        theta=THETA,
        use_bias=USE_BIAS,
        guidance_embed=False,
        image_shape=(
            BUILD_IMAGE_TOKENS,
            INPUT_CHANNELS,
        ),
        text_shape=(
            BUILD_TEXT_TOKENS,
            TEXT_EMBEDDING_DIM,
        ),
        image_ids_shape=(
            BUILD_IMAGE_TOKENS,
            3,
        ),
        text_ids_shape=(
            BUILD_TEXT_TOKENS,
            3,
        ),
        y_shape=(Y_DIM,),
    )

    # Define input shapes
    img_shape = (
        BUILD_IMAGE_TOKENS,
        INPUT_CHANNELS,
    )

    txt_shape = (
        BUILD_TEXT_TOKENS,
        TEXT_EMBEDDING_DIM,
    )

    img_ids_shape = (
        BUILD_IMAGE_TOKENS,
        3,
    )

    txt_ids_shape = (
        BUILD_TEXT_TOKENS,
        3,
    )

    y_shape = (Y_DIM,)

    # Build the model
    keras_model.build(
        (
            img_shape,
            img_ids_shape,
            txt_shape,
            txt_ids_shape,
            y_shape,
        )
    )

    convert_flux_weights(
        flux_weights,
        keras_model,
    )

    keras_model.save_to_preset(OUTPUT_PRESET)

    os.remove(CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
