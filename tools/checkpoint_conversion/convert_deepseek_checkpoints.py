import json
import logging
import os
import shutil
import time
from glob import glob

"""
NOTE: This script depends on the master branch of keras, due to it having
various operations, like keras.ops.polar(), keras.ops.view_as_complex(), etc.
which were added to support DeepSeek-R1.
"""

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoTokenizer

from keras_hub.src.models.deepseek_r1.deepseek_backbone import (
    DeepSeekV3Backbone,
)
from keras_hub.src.models.deepseek_r1.deepseek_causallm_preprocessor import (
    DeepSeekR1CausalLMPreprocessor,
)
from keras_hub.src.models.deepseek_r1.deepseek_tokenizer import (
    DeepSeekR1Tokenizer,
)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Set up base URL for HuggingFace model
hf_preset = "deepseek-ai/DeepSeek-V3-Base"
kh_preset = "deepseekr1_671b_en"

# Define the mapping for parameter renaming
mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}

# Number of model parts to download
end = 1  # Adjust this to 163 for full model


# Function to download and rename model weights
def download_and_rename_weight_files():
    for i in range(end):
        print(f"Downloading model part {i + 1}/{end}")
        weight_path = hf_hub_download(
            hf_preset, f"model-0000{i + 1}-of-000163.safetensors"
        )
        weight_folder = "/".join(weight_path.split("/")[:-1])

        print(f"Downloaded weights to {weight_folder}")
        print(f"Weight files: {glob(os.path.join(weight_folder, '*'))}")

        # Open and process the downloaded file immediately
        print("Converting weights from HuggingFace format...")
        with safe_open(weight_path, framework="pt", device="cpu") as f:
            state_dict = {}
            for name in f.keys():
                if "model.layers.61" in name:
                    continue  # Skip layers you don't want to process

                param: torch.Tensor = f.get_tensor(name)

                if name.startswith("model."):
                    name = name[len("model.") :]  # Remove "model." prefix
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")

                key = name.split(".")[-2]

                # Ensure the key is present in the mapping
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)

                # Save the parameter to the state_dict
                state_dict[name] = param

            # save with the same name in current directory
            filename = os.path.basename(weight_path)
            save_file(state_dict, filename)

            # Delete the original file after saving the renamed file
            os.remove(weight_path)
            print(f"Deleted original file: {weight_path}")

        # Optionally copy token files if needed
        for file_path in glob(os.path.join(weight_folder, "*token*")):
            new_file_path = os.path.join(
                weight_folder, os.path.basename(file_path)
            )
            shutil.copyfile(file_path, new_file_path)


def convert_block(keras_block, torch_weights, index):
    print("Weights and shapes")
    for i, w in enumerate(keras_block.weights):
        print(i, w.path, w.shape)
    print()
    for i, w in enumerate(torch_weights):
        if f"layers.{index - 1}" in w:
            print(i - 1, w, torch_weights[w].shape)

    keras_block.weights[0].assign(
        torch_weights[f"layers.{index - 1}.attn.wq.weight"]
    )
    keras_block.weights[1].assign(
        torch_weights[f"layers.{index - 1}.attn.wkv_a.weight"]
    )
    keras_block.weights[2].assign(
        torch_weights[f"layers.{index - 1}.attn.kv_norm.weight"]
    )
    keras_block.weights[3].assign(
        torch_weights[f"layers.{index - 1}.attn.wkv_b.weight"]
    )
    keras_block.weights[4].assign(
        torch_weights[f"layers.{index - 1}.attn.wo.weight"]
    )
    keras_block.weights[5].assign(
        torch_weights[f"layers.{index - 1}.ffn.w1.weight"]
    )
    keras_block.weights[6].assign(
        torch_weights[f"layers.{index - 1}.ffn.w2.weight"]
    )
    keras_block.weights[7].assign(
        torch_weights[f"layers.{index - 1}.ffn.w3.weight"]
    )
    keras_block.weights[8].assign(
        torch_weights[f"layers.{index - 1}.attn_norm.weight"]
    )
    keras_block.weights[9].assign(
        torch_weights[f"layers.{index - 1}.ffn_norm.weight"]
    )


def convert_tokenizer(hf_preset):
    logging.info("Initializing tokenizer...")
    # === Get the tokenizer from the Huggingface model ===
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    tokenizer_path = hf_hub_download(hf_preset, "tokenizer.json", token=True)
    with open(tokenizer_path, "r") as tokenizer_file:
        tokenizer_content = json.load(tokenizer_file)
    vocabulary = hf_tokenizer.vocab
    merges = tokenizer_content["model"]["merges"]

    keras_hub_tokenizer = DeepSeekR1Tokenizer(vocabulary, merges)
    return keras_hub_tokenizer, hf_tokenizer


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    keras_hub_preprocessor = DeepSeekR1CausalLMPreprocessor(keras_hub_tokenizer)

    hf_tokenizer_outputs = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    keras_tokenizer_outputs = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=6
    )

    assert (
        hf_tokenizer_outputs["input_ids"].cpu().numpy()
        == keras_tokenizer_outputs[0]["token_ids"].cpu().numpy()
    ).all()


def convert_weights():
    logging.info("Loading torch weights...")
    torch_weights = {}

    for i in range(end):
        with safe_open(
            f"model-0000{i + 1}-of-000163.safetensors",
            framework="pt",
            device="cpu",
        ) as f:
            for k in f.keys():
                torch_weights[k] = f.get_tensor(k)

    logging.info("Initializing model...")
    model = DeepSeekV3Backbone.from_preset(
        "keras-hub/keras_hub/src/models/deepseek_r1/deepseek",
        load_weights=False,
    )

    logging.info("Running dummy input...")
    x = keras.random.randint((1, 128), 0, model.vocab_size)
    model(x)

    logging.info(model.summary())

    # print keras weights
    logging.info("Keras weight shapes:")
    for layer in model.layers:
        if not layer.name == "tokens":
            logging.info(f"{layer.name}, {layer.get_weights()[0].shape}")

    # General structure is starting embedding + N blocks + head.
    # Starting by converting the first and last layers
    logging.info("Converting embedding")
    model.layers[1].set_weights(weights=[torch_weights["embed.weight"]])
    logging.info(model.layers[0].weights)

    logging.info("Converting head")
    model.layers[-1].set_weights(weights=[torch_weights["head.weight"]])
    logging.info(model.layers[-1].weights)

    logging.info("Converting head norm")
    model.layers[-2].set_weights(weights=[torch_weights["norm.weight"]])
    logging.info(model.layers[-2].weights)

    n_blocks = len(model.layers) - 3  # (3 = len(embed, head, norm))
    for i in range(n_blocks):
        convert_block(model.layers[i + 2], torch_weights, i + 1)

    # Run some tokens as a sanity check
    total_tokens_generated = 0
    total_generation_time = 0.0
    steps = 10
    logging.info(f"Generating {steps} tokens sequentially")
    x = keras.random.randint((1, 128), 0, model.vocab_size, seed=42)

    outputs = []
    for i in tqdm(range(steps)):
        start_time = time.time()
        outs = model(x)
        res_token = outs.argmax(1).unsqueeze(0)
        outputs.append(res_token)
        end_time = time.time() - start_time
        total_generation_time += end_time
        total_tokens_generated += 1

    tokens_per_second = total_tokens_generated / total_generation_time
    logging.info(f"Total tokens generated: {total_tokens_generated}")
    logging.info(f"Total generation time: {total_generation_time:.2f} seconds")
    logging.info(f"Tokens per second: {tokens_per_second:.2f}")
    logging.info(f"Tokens: {outputs}")

    return model


def save_model(keras_hub_model, keras_hub_tokenizer):
    keras_hub_model.save_to_preset(kh_preset)
    keras_hub_tokenizer.save_to_preset(kh_preset)


def main():
    keras_hub_tokenizer, hf_tokenizer = convert_tokenizer(hf_preset)
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
    download_and_rename_weight_files()
    keras_hub_model = convert_weights()
    save_model(keras_hub_model, keras_hub_tokenizer)


if __name__ == "__main__":
    main()
