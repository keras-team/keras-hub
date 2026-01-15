"""Convert ModernBERT checkpoints.

python tools/checkpoint_conversion/convert_modernbert_checkpoints.py \
    --preset modernbert_base
python tools/checkpoint_conversion/convert_modernbert_checkpoints.py \
    --preset modernbert_large
"""

import json
import os

import numpy as np
import requests
import transformers
from absl import app
from absl import flags

from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone,
)

PRESET_MAP = {
    "modernbert_base": "answerdotai/ModernBERT-base",
    "modernbert_large": "answerdotai/ModernBERT-large",
}

EXTRACT_DIR = "./{}"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
)


def download_files(hf_model_name):
    extract_dir = EXTRACT_DIR.format(FLAGS.preset)
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Config.
    config_path = os.path.join(extract_dir, "config.json")
    response = requests.get(
        f"https://huggingface.co/{hf_model_name}/raw/main/config.json"
    )
    open(config_path, "wb").write(response.content)


def convert_model(hf_model):
    extract_dir = EXTRACT_DIR.format(FLAGS.preset)
    config_path = os.path.join(extract_dir, "config.json")

    # Build config.
    cfg = {}
    with open(config_path, "r") as pt_cfg_handler:
        pt_cfg = json.load(pt_cfg_handler)
    cfg["vocabulary_size"] = pt_cfg["vocab_size"]
    cfg["num_layers"] = pt_cfg["num_hidden_layers"]
    cfg["num_heads"] = pt_cfg["num_attention_heads"]
    cfg["hidden_dim"] = pt_cfg["hidden_size"]
    cfg["intermediate_dim"] = pt_cfg["intermediate_size"]
    cfg["dropout"] = pt_cfg["embedding_dropout"]
    cfg["max_sequence_length"] = pt_cfg["max_position_embeddings"]

    return ModernBertBackbone(**cfg)


def convert_weights(keras_model, hf_model):
    # Get `state_dict` from `hf_model`.
    state_dict = hf_model.state_dict()

    keras_model.get_layer("token_embedding").set_weights(
        [np.asarray(state_dict["embeddings.tok_embeddings.weight"])]
    )

    keras_model.get_layer("embeddings_layer_norm").set_weights(
        [np.asarray(state_dict["embeddings.norm.weight"])]
    )

    for i in range(keras_model.num_layers):
        keras_model.transformer_layers[i].attn.Wqkv.kernel.assign(
            state_dict[f"layers.{i}.attn.Wqkv.weight"].T
        )
        keras_model.transformer_layers[i].attn.Wo.kernel.assign(
            state_dict[f"layers.{i}.attn.Wo.weight"]
        )
        keras_model.transformer_layers[i].mlp_norm.gamma.assign(
            state_dict[f"layers.{i}.mlp_norm.weight"]
        )
        keras_model.transformer_layers[i].mlp.Wi.kernel.assign(
            state_dict[f"layers.{i}.mlp.Wi.weight"].T
        )
        keras_model.transformer_layers[i].mlp.Wo.kernel.assign(
            state_dict[f"layers.{i}.mlp.Wo.weight"].T
        )

    keras_model.get_layer("final_layernorm").set_weights(
        [np.asarray(state_dict["final_norm.weight"])]
    )


def main(_):
    hf_model_name = PRESET_MAP[FLAGS.preset]
    download_files(hf_model_name)

    hf_model = transformers.AutoModel.from_pretrained(hf_model_name)
    hf_model.eval()

    print(f"🏃 Coverting {FLAGS.preset}")
    keras_model = convert_model(hf_model)
    print("✅ KerasHub model loaded.")

    convert_weights(keras_model, hf_model)
    print("✅ Weights converted.")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
