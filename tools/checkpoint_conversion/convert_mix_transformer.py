# Usage example
# python tools/checkpoint_conversion/convert_mix_transformer.py \
#     --preset "B0_ade_512"

from absl import app
from absl import flags
from transformers import SegformerForSemanticSegmentation

import keras_hub

FLAGS = flags.FLAGS


DOWNLOAD_URLS = {
    "B0_ade_512": "nvidia/segformer-b0-finetuned-ade-512-512",
    "B1_ade_512": "nvidia/segformer-b1-finetuned-ade-512-512",
    "B2_ade_512": "nvidia/segformer-b2-finetuned-ade-512-512",
    "B3_ade_512": "nvidia/segformer-b3-finetuned-ade-512-512",
    "B4_ade_512": "nvidia/segformer-b4-finetuned-ade-512-512",
    "B5_ade_640": "nvidia/segformer-b5-finetuned-ade-640-640",
    "B0_cityscapes_1024": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "B1_cityscapes_1024": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    "B2_cityscapes_1024": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "B3_cityscapes_1024": "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    "B4_cityscapes_1024": "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
    "B5_cityscapes_1024": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}


MODEL_CONFIGS = {
    "B0": {"hidden_dims": [32, 64, 160, 256], "layerwise_depths": [2, 2, 2, 2]},
    "B1": {
        "hidden_dims": [64, 128, 320, 512],
        "layerwise_depths": [2, 2, 2, 2],
    },
    "B2": {
        "hidden_dims": [64, 128, 320, 512],
        "layerwise_depths": [3, 4, 6, 3],
    },
    "B3": {
        "hidden_dims": [64, 128, 320, 512],
        "layerwise_depths": [3, 4, 18, 3],
    },
    "B4": {
        "hidden_dims": [64, 128, 320, 512],
        "layerwise_depths": [3, 8, 27, 3],
    },
    "B5": {
        "hidden_dims": [64, 128, 320, 512],
        "layerwise_depths": [3, 6, 40, 3],
    },
}

flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(DOWNLOAD_URLS.keys())}'
)


def get_indices_from_depths(depths):
    proj_indices = []
    norm_indices = []
    hierarchical_encoder_indices = []

    current_layer_idx = 1

    for layer_idx, depth in enumerate(depths):
        # Add projection index (before the hierarchical encoders)
        proj_indices.append(current_layer_idx)

        # Hierarchical encoder block indices
        for block_idx in range(depth):
            hierarchical_encoder_indices.append(
                (current_layer_idx + 1, layer_idx, block_idx)
            )
            current_layer_idx += 1

        # Add normalization index (after the hierarchical encoders)
        norm_indices.append(current_layer_idx + 1)

        # Skip to the next layer after output_level
        current_layer_idx += 3

    return proj_indices, norm_indices, hierarchical_encoder_indices


def set_conv_weights(conv_layer, state_dict):
    conv_weights = state_dict["weight"].numpy().transpose(2, 3, 1, 0)
    conv_bias = state_dict["bias"].numpy()
    conv_layer.set_weights([conv_weights, conv_bias])


def set_dwconv_weights(conv_layer, state_dict):
    conv_weights = state_dict["dwconv.weight"].numpy().transpose(2, 3, 0, 1)
    conv_bias = state_dict["dwconv.bias"].numpy()
    conv_layer.set_weights([conv_weights, conv_bias])


def set_layer_norm_weights(layer_norm, state_dict):
    gamma = state_dict["weight"].numpy()
    beta = state_dict["bias"].numpy()
    layer_norm.set_weights([gamma, beta])


def set_dense_weights(dense_layer, state_dict):
    weight = state_dict["weight"].numpy().T
    bias = state_dict["bias"].numpy()
    dense_layer.set_weights([weight, bias])


def set_hierarchical_encoder_weights(keras_layer, pytorch_layer, key):
    set_layer_norm_weights(
        keras_layer.norm1, pytorch_layer.layer_norm_1.state_dict()
    )

    set_dense_weights(
        keras_layer.attn.q, pytorch_layer.attention.self.query.state_dict()
    )
    set_dense_weights(
        keras_layer.attn.k, pytorch_layer.attention.self.key.state_dict()
    )
    set_dense_weights(
        keras_layer.attn.v, pytorch_layer.attention.self.value.state_dict()
    )
    set_dense_weights(
        keras_layer.attn.proj, pytorch_layer.attention.output.dense.state_dict()
    )

    if keras_layer.attn.sr_ratio > 1:
        set_conv_weights(
            keras_layer.attn.sr, pytorch_layer.attention.self.sr.state_dict()
        )
        set_layer_norm_weights(
            keras_layer.attn.norm,
            pytorch_layer.attention.self.layer_norm.state_dict(),
        )

    set_layer_norm_weights(
        keras_layer.norm2, pytorch_layer.layer_norm_2.state_dict()
    )

    set_dense_weights(
        keras_layer.mlp.fc1, pytorch_layer.mlp.dense1.state_dict()
    )
    set_dwconv_weights(
        keras_layer.mlp.dwconv, pytorch_layer.mlp.dwconv.state_dict()
    )
    set_dense_weights(
        keras_layer.mlp.fc2, pytorch_layer.mlp.dense2.state_dict()
    )


def main(_):
    print("\n-> Loading HuggingFace model")
    model = SegformerForSemanticSegmentation.from_pretrained(
        DOWNLOAD_URLS[FLAGS.preset]
    )
    original_mit = original_mit = model.segformer.encoder

    model_type = FLAGS.preset.split("_")[0]
    print("\n-> Instantiating KerasHub Model")
    keras_mit = keras_hub.models.MiTBackbone(
        layerwise_depths=MODEL_CONFIGS[model_type]["layerwise_depths"],
        image_shape=(224, 224, 3),
        hidden_dims=MODEL_CONFIGS[model_type]["hidden_dims"],
        num_layers=4,
        layerwise_num_heads=[1, 2, 5, 8],
        layerwise_sr_ratios=[8, 4, 2, 1],
        max_drop_path_rate=0.1,
        layerwise_patch_sizes=[7, 3, 3, 3],
        layerwise_strides=[4, 2, 2, 2],
    )

    # Indices for the different patch embeddings and layer norms
    proj_indices, layer_norm_indices, hierarchical_encoder_indices = (
        get_indices_from_depths(MODEL_CONFIGS[model_type]["layerwise_depths"])
    )

    print("\n-> Converting weights...")
    # Loop through the indices to set convolutional and normalization weights
    for i, idx in enumerate(proj_indices):
        set_conv_weights(
            keras_mit.layers[idx].proj,
            original_mit.patch_embeddings[i].proj.state_dict(),
        )
        set_layer_norm_weights(
            keras_mit.layers[idx].norm,
            original_mit.patch_embeddings[i].layer_norm.state_dict(),
        )

    # Set layer normalization weights
    for i, idx in enumerate(layer_norm_indices):
        set_layer_norm_weights(
            keras_mit.layers[idx], original_mit.layer_norm[i].state_dict()
        )

    # Set hierarchical encoder weights
    for layer_idx, block_idx, key in hierarchical_encoder_indices:
        set_hierarchical_encoder_weights(
            keras_mit.layers[layer_idx],
            original_mit.block[block_idx][int(key)],
            key=key,
        )

    directory = f"MiT_{FLAGS.preset}"
    print(f"\n-> Saving converted KerasHub model in {directory}")
    keras_mit.save_to_preset(directory)


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
