import json
import os

import jax.numpy as jnp
import keras
import numpy as np
import requests
import tensorflow as tf
import torch
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer

from keras_hub.src.models.stablelm.stablelm_backbone import StableLMBackbone

# Set the desired Keras backend (e.g., "torch", "tensorflow", "jax")
os.environ["KERAS_BACKEND"] = "torch"

# Detect and verify the current Keras backend
backend = keras.backend.backend()
print(f"Current Keras backend: {backend}")

# Configuration
PRESET_NAME = "stablelm-3b-4e1t"
BASE_MODEL = "stabilityai/stablelm-3b-4e1t"
EXTRACT_DIR = "./{}"

extract_dir = EXTRACT_DIR.format(PRESET_NAME)
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Function to download files with progress bar
def download_file(url, filepath):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(filepath, 'wb') as f, tqdm(
        desc=os.path.basename(filepath),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

# Download vocab and merges
vocab_path = os.path.join(extract_dir, "vocab.json")
merges_path = os.path.join(extract_dir, "merges.txt")
tokenizer_url = f"https://huggingface.co/{BASE_MODEL}/raw/main/tokenizer.json"
download_file(tokenizer_url, os.path.join(extract_dir, "tokenizer.json"))
with open(os.path.join(extract_dir, "tokenizer.json"), "r") as f:
    tokenizer_data = json.load(f)
vocab = {
    token: idx for idx, token in enumerate(tokenizer_data["model"]["vocab"])
}
with open(vocab_path, "w") as f:
    json.dump(vocab, f)
merges = tokenizer_data["model"]["merges"]
with open(merges_path, "w") as f:
    for merge in merges:
        f.write(merge + "\n")

# Download config
config_path = os.path.join(extract_dir, "config.json")
config_url = f"https://huggingface.co/{BASE_MODEL}/raw/main/config.json"
download_file(config_url, config_path)
cfg = {}
with open(config_path, "r") as pt_cfg_handler:
    pt_cfg = json.load(pt_cfg_handler)

cfg["vocabulary_size"] = pt_cfg["vocab_size"]
cfg["num_layers"] = pt_cfg["num_hidden_layers"]
cfg["num_query_heads"] = pt_cfg["num_attention_heads"]
cfg["num_key_value_heads"] = pt_cfg["num_key_value_heads"]
cfg["hidden_dim"] = pt_cfg["hidden_size"]
cfg["intermediate_dim"] = pt_cfg["intermediate_size"]
cfg["max_sequence_length"] = pt_cfg["max_position_embeddings"]
cfg["layer_norm_epsilon"] = pt_cfg["layer_norm_eps"]
cfg["rope_max_wavelength"] = pt_cfg["rope_theta"]
cfg["partial_rotary_factor"] = pt_cfg["partial_rotary_factor"]

# Load Hugging Face model
hf_model = AutoModel.from_pretrained(BASE_MODEL)
hf_model.eval()
hf_wts = hf_model.state_dict()

# Initialize Keras model
keras_model = StableLMBackbone(**cfg)

# Function to convert tensors to NumPy based on tensor type
def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    elif isinstance(tensor, jnp.ndarray):
        return np.array(tensor)
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")

# Transfer weights
keras_model.get_layer("token_embedding").embeddings.assign(
    to_numpy(hf_model.embed_tokens.weight)
)

for layer_index in range(cfg["num_layers"]):
    hidden_size = cfg["hidden_dim"]
    num_attention_heads = cfg["num_query_heads"]
    num_key_value_heads = cfg["num_key_value_heads"]
    head_dim = hidden_size // num_attention_heads

    # Query projection
    q_weight_key = f"layers.{layer_index}.self_attn.q_proj.weight"
    q_weight_hf = to_numpy(hf_wts[q_weight_key])
    q_weight = q_weight_hf.T.reshape(hidden_size, num_attention_heads, head_dim)
    weights = [q_weight]
    q_bias_key = f"layers.{layer_index}.self_attn.q_proj.bias"
    if q_bias_key in hf_wts:
        q_bias = to_numpy(hf_wts[q_bias_key])
        weights.append(q_bias)
    keras_model.get_layer(
        f"transformer_layer_{layer_index}"
    )._self_attention_layer._query_dense.set_weights(weights)

    # Key projection
    k_weight_key = f"layers.{layer_index}.self_attn.k_proj.weight"
    k_weight_hf = to_numpy(hf_wts[k_weight_key])
    k_weight = k_weight_hf.T.reshape(hidden_size, num_key_value_heads, head_dim)
    weights = [k_weight]
    k_bias_key = f"layers.{layer_index}.self_attn.k_proj.bias"
    if k_bias_key in hf_wts:
        k_bias = to_numpy(hf_wts[k_bias_key])
        weights.append(k_bias)
    keras_model.get_layer(
        f"transformer_layer_{layer_index}"
    )._self_attention_layer._key_dense.set_weights(weights)

    # Value projection
    v_weight_key = f"layers.{layer_index}.self_attn.v_proj.weight"
    v_weight_hf = to_numpy(hf_wts[v_weight_key])
    v_weight = v_weight_hf.T.reshape(hidden_size, num_key_value_heads, head_dim)
    weights = [v_weight]
    v_bias_key = f"layers.{layer_index}.self_attn.v_proj.bias"
    if v_bias_key in hf_wts:
        v_bias = to_numpy(hf_wts[v_bias_key])
        weights.append(v_bias)
    keras_model.get_layer(
        f"transformer_layer_{layer_index}"
    )._self_attention_layer._value_dense.set_weights(weights)

    # Output projection
    o_weight_key = f"layers.{layer_index}.self_attn.o_proj.weight"
    o_weight_hf = to_numpy(hf_wts[o_weight_key])
    o_weight = o_weight_hf.T.reshape(num_attention_heads, head_dim, hidden_size)
    weights = [o_weight]
    o_bias_key = f"layers.{layer_index}.self_attn.o_proj.bias"
    if o_bias_key in hf_wts:
        o_bias = to_numpy(hf_wts[o_bias_key])
        weights.append(o_bias)
    keras_model.get_layer(
        f"transformer_layer_{layer_index}"
    )._self_attention_layer._output_dense.set_weights(weights)

    # LayerNorms
    ln_weight_key = f"layers.{layer_index}.input_layernorm.weight"
    ln_bias_key = f"layers.{layer_index}.input_layernorm.bias"
    keras_model.get_layer(
        f"transformer_layer_{layer_index}"
    )._self_attention_layernorm.set_weights(
        [
            to_numpy(hf_wts[ln_weight_key]),
            to_numpy(hf_wts[ln_bias_key])
        ]
    )

    ln_weight_key = f"layers.{layer_index}.post_attention_layernorm.weight"
    ln_bias_key = f"layers.{layer_index}.post_attention_layernorm.bias"
    keras_model.get_layer(
        f"transformer_layer_{layer_index}"
    )._feedforward_layernorm.set_weights(
        [
            to_numpy(hf_wts[ln_weight_key]),
            to_numpy(hf_wts[ln_bias_key])
        ]
    )

    # Feedforward
    ff_gate_weight_key = f"layers.{layer_index}.mlp.gate_proj.weight"
    ff_gate_weight_hf = to_numpy(hf_wts[ff_gate_weight_key])
    weights = [ff_gate_weight_hf.T]
    ff_gate_bias_key = f"layers.{layer_index}.mlp.gate_proj.bias"
    if ff_gate_bias_key in hf_wts:
        ff_gate_bias = to_numpy(hf_wts[ff_gate_bias_key])
        weights.append(ff_gate_bias)
    keras_model.get_layer(
        f"transformer_layer_{layer_index}"
    )._feedforward_gate_dense.set_weights(weights)

    ff_inter_weight_key = f"layers.{layer_index}.mlp.up_proj.weight"
    ff_inter_weight_hf = to_numpy(hf_wts[ff_inter_weight_key])
    weights = [ff_inter_weight_hf.T]
    ff_inter_bias_key = f"layers.{layer_index}.mlp.up_proj.bias"
    if ff_inter_bias_key in hf_wts:
        ff_inter_bias = to_numpy(hf_wts[ff_inter_bias_key])
        weights.append(ff_inter_bias)
    keras_model.get_layer(
        f"transformer_layer_{layer_index}"
    )._feedforward_intermediate_dense.set_weights(weights)

    ff_out_weight_key = f"layers.{layer_index}.mlp.down_proj.weight"
    ff_out_weight_hf = to_numpy(hf_wts[ff_out_weight_key])
    weights = [ff_out_weight_hf.T]
    ff_out_bias_key = f"layers.{layer_index}.mlp.down_proj.bias"
    if ff_out_bias_key in hf_wts:
        ff_out_bias = to_numpy(hf_wts[ff_out_bias_key])
        weights.append(ff_out_bias)
    keras_model.get_layer(
        f"transformer_layer_{layer_index}"
    )._feedforward_output_dense.set_weights(weights)

# Final LayerNorm
keras_model.get_layer("sequence_output_layernorm").set_weights(
    [
        to_numpy(hf_wts["norm.weight"]),
        to_numpy(hf_wts["norm.bias"])
    ]
)

# Tokenization and comparison
hf_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
sample_text = ["Royal Challengers Bangalore will be winning this IPL"]
hf_inputs = hf_tokenizer(sample_text, return_tensors="pt")
print("HF inputs:", hf_inputs)

if backend == "torch":
    token_ids = hf_inputs["input_ids"]
    padding_mask = hf_inputs["attention_mask"]
elif backend == "tensorflow":
    token_ids = tf.convert_to_tensor(hf_inputs["input_ids"].numpy())
    padding_mask = tf.convert_to_tensor(hf_inputs["attention_mask"].numpy())
elif backend == "jax":
    token_ids = jnp.array(hf_inputs["input_ids"].numpy())
    padding_mask = jnp.array(hf_inputs["attention_mask"].numpy())
else:
    raise ValueError(f"Unsupported backend: {backend}")

keras_inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
keras_outputs = keras_model(keras_inputs)
keras_outputs_np = to_numpy(keras_outputs)
print("Keras output:", keras_outputs_np)

hf_outputs = hf_model(**hf_inputs).last_hidden_state
hf_outputs_np = hf_outputs.detach().cpu().numpy()
print("HF output:", hf_outputs_np)

try:
    np.testing.assert_allclose(hf_outputs_np, keras_outputs_np, atol=1e-3)
    print("✅ Model outputs match!")
except AssertionError:
    print("❌ Model outputs differ!")