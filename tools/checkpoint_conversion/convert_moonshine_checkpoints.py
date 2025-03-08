"""
Convert Moonshine checkpoints to KerasHub format.

The weights are sourced from:
https://huggingface.co/UsefulSensors/moonshine/tree/main/base

The Hugging Face config is available at:
https://huggingface.co/UsefulSensors/moonshine-base/blob/main/config.json

Usage:
```shell
python -m tools.checkpoint_conversion.convert_moonshine_checkpoints
```
"""

import hashlib
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import h5py
import keras
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel

from keras_hub.src.models.moonshine.moonshine_audio_converter import (
    MoonshineAudioConverter,
)
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_decoder import MoonshineDecoder
from keras_hub.src.models.moonshine.moonshine_encoder import MoonshineEncoder


# Utility functions.
def to_numpy(tensor):
    if keras.backend.backend() == "torch":
        return tensor.detach().numpy()
    elif keras.backend.backend() == "tensorflow":
        return tensor.numpy()
    elif keras.backend.backend() == "jax":
        import jax

        return jax.device_get(tensor)
    else:
        raise ValueError("Unsupported backend")


def compute_md5_checksum(array):
    m = hashlib.md5()
    m.update(array.tobytes())
    return m.hexdigest()


def find_precision(keras_output, hf_output):
    diff = np.abs(keras_output - hf_output)
    max_diff = np.max(diff)
    min_diff = np.min(diff)
    if max_diff == 0:
        return "exact match", max_diff, min_diff
    for n in range(10, -1, -1):
        atol = 10**-n
        if np.all(diff <= atol):
            return f"1e-{n}", max_diff, min_diff
    return "less than 1e0", max_diff, min_diff


def shorthand_repr(tensor, max_rows=3, max_cols=3, indent=0, precision=6):
    tensor = np.array(tensor)
    indent_str = "  " * indent

    if tensor.ndim == 0:
        return f"{tensor:.{precision}f}"
    elif tensor.ndim == 1:
        row_list = list(tensor)
        if len(row_list) > max_cols:
            row_str = (
                ", ".join(f"{x:.{precision}f}" for x in row_list[:max_cols])
                + ", ..."
            )
        else:
            row_str = ", ".join(f"{x:.{precision}f}" for x in row_list)
        return f"{indent_str}[{row_str}]"

    else:
        lines = []
        for i, row in enumerate(tensor):
            if i >= max_rows:
                lines.append(f"{indent_str}  ...")
                break
            row_repr = shorthand_repr(
                row, max_rows, max_cols, indent + 1, precision
            )
            lines.append(row_repr)
        return f"{indent_str}[\n" + ",\n".join(lines) + f"\n{indent_str}]"


# Weight downloading.
def download_weights(weights_dir):
    repo_id = "UsefulSensors/moonshine"
    files = [
        "encoder.weights.h5",
        "preprocessor.weights.h5",
        "decoder.weights.h5",
    ]
    os.makedirs(weights_dir, exist_ok=True)
    for fname in files:
        file_path = os.path.join(weights_dir, fname)
        if not os.path.exists(file_path):
            print(f"Downloading {fname} to {file_path}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{weights_dir.split('_')[-1]}/{fname}",
                local_dir=weights_dir,
                local_dir_use_symlinks=False,
            )
        else:
            print(f"{file_path} already exists. Skipping download.")
    return weights_dir


# Weight loading and mapping.
def load_h5_weights(filepath):
    weights_dict = {}
    with h5py.File(filepath, "r") as f:

        def recursive_load(group, prefix=""):
            result = {}
            for key in group.keys():
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(group[key], h5py.Dataset):
                    result[path] = np.array(group[key])
                else:
                    result.update(recursive_load(group[key], path))
            return result

        weights_dict = recursive_load(f)
    return weights_dict


def map_rotary_embedding_weights(orig_weights, moonshine_encoder):
    moonshine_encoder.rotary_embedding.inv_freq.assign(
        orig_weights["layers/rotary_embedding/vars/0"]
    )
    print("Successfully mapped rotary embedding weights")


def map_encoder_block_weights(orig_weights, moonshine_encoder, block_idx):
    moonshine_block = moonshine_encoder.encoder_layers[block_idx]
    if block_idx == 0:
        base_prefix = "layers/functional/layers"
    else:
        base_prefix = f"layers/functional_{block_idx}/layers"
    attention_prefix = f"{base_prefix}/mha_with_rope"
    ff_prefix = f"{base_prefix}/functional/layers/sequential/layers"
    attention_mappings = {
        "query": f"{attention_prefix}/query_dense/vars/0",
        "key": f"{attention_prefix}/key_dense/vars/0",
        "value": f"{attention_prefix}/value_dense/vars/0",
        "output": f"{attention_prefix}/output_dense/vars/0",
    }
    for weight_type, path in attention_mappings.items():
        weight = orig_weights[path]
        if weight_type == "query":
            moonshine_block.self_attention_layer._query_dense.kernel.assign(
                weight
            )
        elif weight_type == "key":
            moonshine_block.self_attention_layer._key_dense.kernel.assign(
                weight
            )
        elif weight_type == "value":
            moonshine_block.self_attention_layer._value_dense.kernel.assign(
                weight
            )
        elif weight_type == "output":
            moonshine_block.self_attention_layer._output_dense.kernel.assign(
                weight
            )
    moonshine_block.self_attention_layer_norm.gamma.assign(
        orig_weights[f"{base_prefix}/layer_normalization/vars/0"]
    )
    moonshine_block.feedforward_layer_norm.gamma.assign(
        orig_weights[f"{base_prefix}/layer_normalization_1/vars/0"]
    )
    moonshine_block.feedforward.dense_1.kernel.assign(
        orig_weights[f"{ff_prefix}/dense/vars/0"]
    )
    moonshine_block.feedforward.dense_1.bias.assign(
        orig_weights[f"{ff_prefix}/dense/vars/1"]
    )
    moonshine_block.feedforward.dense_2.kernel.assign(
        orig_weights[f"{ff_prefix}/dense_1/vars/0"]
    )
    moonshine_block.feedforward.dense_2.bias.assign(
        orig_weights[f"{ff_prefix}/dense_1/vars/1"]
    )
    print(f"Successfully mapped encoder block {block_idx} weights")


def map_final_layer_norm_weights(orig_weights, moonshine_encoder):
    moonshine_encoder.final_layer_norm.gamma.assign(
        orig_weights["layers/layer_normalization/vars/0"]
    )
    print("Successfully mapped final layer norm weights")


def map_encoder_weights(orig_weights, moonshine_encoder):
    map_rotary_embedding_weights(orig_weights, moonshine_encoder)
    for block_idx in range(moonshine_encoder.num_layers):
        map_encoder_block_weights(orig_weights, moonshine_encoder, block_idx)
    final_layer_norm_gamma = orig_weights["layers/layer_normalization/vars/0"]
    moonshine_encoder.final_layer_norm.gamma.assign(final_layer_norm_gamma)
    print("Successfully mapped final layer norm weights")
    return final_layer_norm_gamma


def map_preprocessor_weights(weights_path, moonshine_preprocessor):
    with h5py.File(weights_path, "r") as f:
        base_path = "layers/sequential/layers/"
        weights = [
            np.array(f[f"{base_path}conv1d/vars/0"]),  # conv1 kernel
            np.array(f[f"{base_path}group_normalization/vars/0"]),  # gamma
            np.array(f[f"{base_path}group_normalization/vars/1"]),  # beta
            np.array(f[f"{base_path}conv1d_1/vars/0"]),  # conv2 kernel
            np.array(f[f"{base_path}conv1d_1/vars/1"]),  # conv2 bias
            np.array(f[f"{base_path}conv1d_2/vars/0"]),  # conv3 kernel
            np.array(f[f"{base_path}conv1d_2/vars/1"]),  # conv3 bias
        ]
    moonshine_preprocessor.preprocess.set_weights(weights)
    print("Successfully mapped preprocessor weights")


def map_decoder_block_weights(h5_file, moonshine_decoder, block_idx):
    moonshine_block = moonshine_decoder.decoder_layers[block_idx]
    if block_idx == 0:
        base_prefix = "layers/functional/layers"
    else:
        base_prefix = f"layers/functional_{block_idx}/layers"
    self_attention_prefix = f"{base_prefix}/mha_causal_with_rope"
    cross_attention_prefix = f"{base_prefix}/mha_precomputed_kv"
    ff_prefix = f"{base_prefix}/functional/layers"
    self_attention_mappings = {
        "query": f"{self_attention_prefix}/query_dense/vars/0",
        "key": f"{self_attention_prefix}/key_dense/vars/0",
        "value": f"{self_attention_prefix}/value_dense/vars/0",
        "output": f"{self_attention_prefix}/output_dense/vars/0",
    }
    for weight_type, path in self_attention_mappings.items():
        weight = h5_file[path][()]
        if weight_type == "query":
            moonshine_block.self_attention._query_dense.kernel.assign(weight)
        elif weight_type == "key":
            moonshine_block.self_attention._key_dense.kernel.assign(weight)
        elif weight_type == "value":
            moonshine_block.self_attention._value_dense.kernel.assign(weight)
        elif weight_type == "output":
            moonshine_block.self_attention._output_dense.kernel.assign(weight)
    cross_attention_mappings = {
        "query": f"{cross_attention_prefix}/query_dense/vars/0",
        "key": f"{cross_attention_prefix}/key_dense/vars/0",
        "value": f"{cross_attention_prefix}/value_dense/vars/0",
        "output": f"{cross_attention_prefix}/output_dense/vars/0",
    }
    for weight_type, path in cross_attention_mappings.items():
        weight = h5_file[path][()]
        if weight_type == "query":
            moonshine_block.cross_attention._query_dense.kernel.assign(weight)
        elif weight_type == "key":
            moonshine_block.cross_attention._key_dense.kernel.assign(weight)
        elif weight_type == "value":
            moonshine_block.cross_attention._value_dense.kernel.assign(weight)
        elif weight_type == "output":
            moonshine_block.cross_attention._output_dense.kernel.assign(weight)
    moonshine_block.norm1.gamma.assign(
        h5_file[f"{base_prefix}/layer_normalization/vars/0"][()]
    )
    moonshine_block.norm2.gamma.assign(
        h5_file[f"{base_prefix}/layer_normalization_1/vars/0"][()]
    )
    moonshine_block.norm3.gamma.assign(
        h5_file[f"{base_prefix}/layer_normalization_2/vars/0"][()]
    )
    moonshine_block.ff.dense_1.kernel.assign(
        h5_file[f"{ff_prefix}/dense/vars/0"][()]
    )
    moonshine_block.ff.dense_1.bias.assign(
        h5_file[f"{ff_prefix}/dense/vars/1"][()]
    )
    moonshine_block.ff.dense_2.kernel.assign(
        h5_file[f"{ff_prefix}/dense_1/vars/0"][()]
    )
    moonshine_block.ff.dense_2.bias.assign(
        h5_file[f"{ff_prefix}/dense_1/vars/1"][()]
    )
    print(f"Successfully mapped decoder block {block_idx} weights")


def map_decoder_weights(weights_path, moonshine_decoder):
    with h5py.File(weights_path, "r") as f:
        moonshine_decoder.embedding_layer.embeddings.assign(
            f["layers/reversible_embedding/vars/0"][()]
        )
        moonshine_decoder.rotary_embedding.inv_freq.assign(
            f["layers/rotary_embedding/vars/0"][()]
        )
        moonshine_decoder.post_norm.gamma.assign(
            f["layers/layer_normalization/vars/0"][()]
        )
        for block_idx in range(moonshine_decoder.num_layers):
            map_decoder_block_weights(f, moonshine_decoder, block_idx)
    print("Successfully mapped decoder weights")


# Model building.
def build_moonshine_encoder(config):
    encoder = MoonshineEncoder(
        attention_bias=config["attention_bias"],
        attention_dropout=config["attention_dropout"],
        dtype=config["dtype"],
        feedforward_expansion_factor=config["feedforward_expansion_factor"],
        hidden_dim=config["hidden_dim"],
        initializer_range=config["initializer_range"],
        intermediate_dim=config["intermediate_dim"],
        max_position_embeddings=config["max_position_embeddings"],
        num_heads=config["encoder_num_heads"],
        num_layers=config["encoder_num_layers"],
        partial_rotary_factor=config["partial_rotary_factor"],
        rope_scaling=config["rope_scaling"],
        rope_theta=config["rope_theta"],
        use_swiglu_activation=config["encoder_use_swiglu_activation"],
    )
    encoder.build(input_shape=[(1, 50, config["hidden_dim"]), (1,)])
    return encoder


def build_moonshine_preprocessor(config):
    preprocessor = MoonshineAudioConverter(
        filter_dim=config["filter_dim"],
        initializer_range=config["initializer_range"],
        sampling_rate=config["sampling_rate"],
        padding_value=config["padding_value"],
        do_normalize=config["do_normalize"],
        return_attention_mask=config["return_attention_mask"],
    )
    return preprocessor


def build_moonshine_decoder(config):
    decoder = MoonshineDecoder(
        attention_bias=config["attention_bias"],
        attention_dropout=config["attention_dropout"],
        dtype=config["dtype"],
        feedforward_expansion_factor=config["feedforward_expansion_factor"],
        hidden_dim=config["hidden_dim"],
        initializer_range=config["initializer_range"],
        intermediate_dim=config["intermediate_dim"],
        max_position_embeddings=config["max_position_embeddings"],
        num_heads=config["decoder_num_heads"],
        num_layers=config["decoder_num_layers"],
        partial_rotary_factor=config["partial_rotary_factor"],
        rope_scaling=config["rope_scaling"],
        rope_theta=config["rope_theta"],
        use_swiglu_activation=config["decoder_use_swiglu_activation"],
        vocabulary_size=config["vocabulary_size"],
    )
    dummy_token_ids = keras.random.randint(
        (1, 32), minval=0, maxval=config["vocabulary_size"], dtype="int32"
    )
    dummy_context = keras.random.uniform(
        (1, 40, config["hidden_dim"]), dtype="float32"
    )
    dummy_seq_len = keras.ops.convert_to_tensor([32], dtype="int32")
    decoder([dummy_token_ids, dummy_context, dummy_seq_len], training=False)
    return decoder


def build_moonshine_backbone(config):
    backbone = MoonshineBackbone(
        vocabulary_size=config["vocabulary_size"],
        encoder_num_layers=config["encoder_num_layers"],
        decoder_num_layers=config["decoder_num_layers"],
        hidden_dim=config["hidden_dim"],
        intermediate_dim=config["intermediate_dim"],
        encoder_num_heads=config["encoder_num_heads"],
        decoder_num_heads=config["decoder_num_heads"],
        feedforward_expansion_factor=config["feedforward_expansion_factor"],
        decoder_use_swiglu_activation=config["decoder_use_swiglu_activation"],
        encoder_use_swiglu_activation=config["encoder_use_swiglu_activation"],
        max_position_embeddings=config["max_position_embeddings"],
        partial_rotary_factor=config["partial_rotary_factor"],
        dropout=config["attention_dropout"],
        initializer_range=config["initializer_range"],
        rope_theta=config["rope_theta"],
        attention_bias=config["attention_bias"],
        attention_dropout=config["attention_dropout"],
        rope_scaling=config["rope_scaling"],
        dtype=config["dtype"],
    )
    return backbone


# Validation functions.
def validate_preprocessor(
    keras_preprocessor, hf_encoder, raw_audio, raw_audio_torch
):
    keras_output = keras_preprocessor(raw_audio, padding="longest")[
        "input_values"
    ]
    keras_output_np = to_numpy(keras_output)

    # Process with Hugging Face encoder.
    with torch.no_grad():
        hf_output = hf_encoder.conv1(raw_audio_torch.unsqueeze(1))
        hf_output = torch.tanh(hf_output)
        hf_output = hf_encoder.groupnorm(hf_output)
        hf_output = F.gelu(hf_encoder.conv2(hf_output))
        hf_output = F.gelu(hf_encoder.conv3(hf_output))
        hf_output = hf_output.permute(0, 2, 1).detach().numpy()

    # Validation logging.
    precision, max_diff, min_diff = find_precision(keras_output_np, hf_output)
    print(f"Preprocessor outputs match up to {precision} precision.")
    print(f"Maximum difference: {max_diff}")
    print(f"Minimum difference: {min_diff}")
    if max_diff > 1e-4:
        print("Warning: Preprocessor outputs do not match within 1e-4")
    print(f"Keras Preprocessor MD5: {compute_md5_checksum(keras_output_np)}")
    print(f"Hugging Face Preprocessor MD5: {compute_md5_checksum(hf_output)}")
    print(f"Keras Preprocessor Shape: {keras_output_np.shape}")
    print(f"Hugging Face Preprocessor Shape: {hf_output.shape}")
    print(f"Keras Preprocessor Sample: {shorthand_repr(keras_output_np)}")
    print(f"Hugging Face Preprocessor Sample: {shorthand_repr(hf_output)}")

    return keras_output


def validate_encoder(keras_encoder, hf_encoder, preprocessed, raw_audio_torch):
    seq_length = keras.ops.convert_to_tensor(
        [preprocessed.shape[1]], dtype="int32"
    )
    keras_output = keras_encoder([preprocessed, seq_length])
    keras_output_np = to_numpy(keras_output)

    with torch.no_grad():
        hf_output = (
            hf_encoder(raw_audio_torch).last_hidden_state.detach().numpy()
        )

    # Validation logging.
    precision, max_diff, min_diff = find_precision(keras_output_np, hf_output)
    print(f"Encoder outputs match up to {precision} precision.")
    print(f"Maximum difference: {max_diff}")
    print(f"Minimum difference: {min_diff}")
    if max_diff > 1e-4:
        print("Warning: Encoder outputs do not match within 1e-4")
    print(f"Keras Encoder MD5: {compute_md5_checksum(keras_output_np)}")
    print(f"Hugging Face Encoder MD5: {compute_md5_checksum(hf_output)}")
    print(f"Keras Encoder Shape: {keras_output_np.shape}")
    print(f"Hugging Face Encoder Shape: {hf_output.shape}")
    print(f"Keras Encoder Sample: {shorthand_repr(keras_output_np)}")
    print(f"Hugging Face Encoder Sample: {shorthand_repr(hf_output)}")

    return keras_output_np, hf_output


def validate_decoder(
    keras_decoder, hf_decoder, keras_encoder_output, hf_encoder_output
):
    batch_size, token_seq_length = 1, 32
    test_token_ids_torch = torch.randint(
        0, 32768, (batch_size, token_seq_length), dtype=torch.int32
    )
    position_ids = (
        torch.arange(0, token_seq_length, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    with torch.no_grad():
        hf_output = (
            hf_decoder(
                input_ids=test_token_ids_torch,
                encoder_hidden_states=torch.from_numpy(hf_encoder_output),
                position_ids=position_ids,
            )
            .last_hidden_state.detach()
            .numpy()
        )
    test_token_ids = keras.ops.convert_to_tensor(
        test_token_ids_torch.numpy(), dtype="int32"
    )
    test_context = keras.ops.convert_to_tensor(
        keras_encoder_output, dtype="float32"
    )
    test_seq_len = keras.ops.convert_to_tensor(
        [token_seq_length], dtype="int32"
    )
    keras_output = keras_decoder(
        [test_token_ids, test_context, test_seq_len], training=False
    )
    keras_output_np = (
        keras_output.numpy()
        if keras.backend.backend() == "tensorflow"
        else keras_output.detach().numpy()
        if keras.backend.backend() == "torch"
        else keras_output
    )
    precision, max_diff, min_diff = find_precision(keras_output_np, hf_output)
    print(f"Decoder outputs match up to {precision} precision.")
    print(f"Maximum difference: {max_diff}")
    print(f"Minimum difference: {min_diff}")
    if max_diff > 1e-4:
        print("Warning: Decoder outputs do not match within 1e-4")
    print(f"Keras Decoder MD5: {compute_md5_checksum(keras_output_np)}")
    print(f"Hugging Face Decoder MD5: {compute_md5_checksum(hf_output)}")
    print(f"Keras Decoder Shape: {keras_output_np.shape}")
    print(f"Hugging Face Decoder Shape: {hf_output.shape}")
    print(f"Keras Decoder Sample: {shorthand_repr(keras_output_np)}")
    print(f"Hugging Face Decoder Sample: {shorthand_repr(hf_output)}")


def compare_model_outputs(
    keras_backbone, hf_model, raw_audio_np, keras_preprocessed
):
    batch_size, audio_length = raw_audio_np.shape
    raw_audio_torch = torch.from_numpy(raw_audio_np)

    # Generate decoder input tokens (same for both models).
    token_seq_length = 32
    test_token_ids_torch = torch.randint(
        0, 32768, (batch_size, token_seq_length), dtype=torch.int32
    )
    test_token_ids = keras.ops.convert_to_tensor(
        test_token_ids_torch.numpy()
    )  # Convert to Keras tensor.
    position_ids = (
        torch.arange(0, token_seq_length, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    # Define attention and padding masks.
    encoder_attention_mask = keras.ops.ones(
        (batch_size, keras_preprocessed.shape[1]), dtype="int32"
    )
    decoder_padding_mask = keras.ops.ones(
        (batch_size, token_seq_length), dtype="int32"
    )

    # Prepare inputs for Keras model as a dictionary.
    keras_inputs = {
        "encoder_input_values": keras_preprocessed,  # Preprocessed audio
        # features.
        "encoder_attention_mask": encoder_attention_mask,  # Encoder mask.
        "decoder_token_ids": test_token_ids,  # Decoder token IDs.
        "decoder_padding_mask": decoder_padding_mask,  # Decoder mask.
    }

    # Compute Keras model outputs.
    keras_outputs = keras_backbone(keras_inputs, training=False)
    keras_logits = keras_outputs["decoder_sequence_output"]
    keras_logits_np = to_numpy(keras_logits)

    # Compute Hugging Face model outputs.
    with torch.no_grad():
        hf_outputs = hf_model(
            input_values=raw_audio_torch,
            decoder_input_ids=test_token_ids_torch,
            decoder_position_ids=position_ids,
            return_dict=True,
        )
        hf_decoder_hidden_states = hf_outputs.last_hidden_state
        hf_logits = (
            (hf_decoder_hidden_states @ hf_model.decoder.embed_tokens.weight.T)
            .detach()
            .numpy()
        )

    # Compare final outputs (logits).
    precision, max_diff, min_diff = find_precision(keras_logits_np, hf_logits)
    print("\n=== Final Model Output Comparison ===")
    print(f"Model outputs match up to {precision} precision.")
    print(f"Maximum difference: {max_diff}")
    print(f"Minimum difference: {min_diff}")

    print(f"Keras Model Output MD5: {compute_md5_checksum(keras_logits_np)}")
    print(f"HuggingFace Model Output MD5: {compute_md5_checksum(hf_logits)}")
    print(f"Keras Model Output Shape: {keras_logits_np.shape}")
    print(f"HuggingFace Model Output Shape: {hf_logits.shape}")
    print(f"Keras Model Output Sample: {shorthand_repr(keras_logits_np)}")
    print(f"HuggingFace Model Output Sample: {shorthand_repr(hf_logits)}")

    if max_diff <= 1e-4:
        print("✅ MATCH: Full model outputs match within 1e-4 tolerance!")
    else:
        print("⚠️ MISMATCH: Full model outputs differ by more than 1e-4!")

    return keras_logits_np, hf_logits


def validate_backbone(
    keras_backbone, hf_model, keras_preprocessed, raw_audio_torch
):
    batch_size, seq_length = (
        keras_preprocessed.shape[0],
        keras_preprocessed.shape[1],
    )
    token_seq_length = 32
    encoder_attention_mask = keras.ops.ones(
        (batch_size, seq_length), dtype="int32"
    )
    test_token_ids_np = np.random.randint(
        0,
        keras_backbone.vocabulary_size,
        (batch_size, token_seq_length),
        dtype=np.int32,
    )
    test_token_ids = keras.ops.convert_to_tensor(
        test_token_ids_np, dtype="int32"
    )
    test_token_ids_torch = torch.from_numpy(test_token_ids_np)

    decoder_padding_mask = keras.ops.ones(
        (batch_size, token_seq_length), dtype="int32"
    )
    position_ids = (
        torch.arange(0, token_seq_length, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    # Convert keras_preprocessed to a PyTorch tensor.
    encoder_input_values = keras.ops.convert_to_tensor(
        keras_preprocessed, dtype="float32"
    )

    # Prepare inputs as a dictionary of PyTorch tensors.
    inputs = {
        "encoder_input_values": encoder_input_values,
        "encoder_attention_mask": encoder_attention_mask,
        "decoder_token_ids": test_token_ids,
        "decoder_padding_mask": decoder_padding_mask,
    }

    # Call the backbone with consistent tensor inputs.
    keras_outputs = keras_backbone(inputs, training=False)

    keras_encoder_output = keras_outputs["encoder_sequence_output"]
    keras_decoder_output = keras_outputs["decoder_sequence_output"]
    keras_encoder_output_np = to_numpy(keras_encoder_output)
    keras_decoder_output_np = to_numpy(keras_decoder_output)
    with torch.no_grad():
        hf_encoder_output = hf_model.encoder(raw_audio_torch).last_hidden_state
        hf_decoder_output = hf_model.decoder(
            input_ids=test_token_ids_torch,
            encoder_hidden_states=hf_encoder_output,
            position_ids=position_ids,
        )
        hf_decoder_hidden_states = hf_decoder_output.last_hidden_state
        hf_decoder_logits = (
            hf_decoder_hidden_states @ hf_model.decoder.embed_tokens.weight.T
        )
        hf_encoder_output_np = hf_encoder_output.detach().numpy()
        hf_decoder_logits_np = hf_decoder_logits.detach().numpy()
    encoder_precision, encoder_max_diff, encoder_min_diff = find_precision(
        keras_encoder_output_np, hf_encoder_output_np
    )
    print("\n--- Backbone Encoder Output Validation ---")
    print(
        f"Backbone encoder outputs match up to {encoder_precision} precision."
    )
    print(f"Maximum difference: {encoder_max_diff}")
    print(f"Minimum difference: {encoder_min_diff}")
    if encoder_max_diff > 1e-4:
        print("Warning: Backbone encoder outputs do not match within 1e-4")
    print(
        f"Keras Backbone Encoder MD5: "
        f"{compute_md5_checksum(keras_encoder_output_np)}"
    )
    print(
        f"Hugging Face Encoder MD5: "
        f"{compute_md5_checksum(hf_encoder_output_np)}"
    )
    print(f"Keras Backbone Encoder Shape: {keras_encoder_output_np.shape}")
    print(f"Hugging Face Encoder Shape: {hf_encoder_output_np.shape}")
    print(
        f"Keras Backbone Encoder Sample: "
        f"{shorthand_repr(keras_encoder_output_np)}"
    )
    print(
        f"Hugging Face Encoder Sample: {shorthand_repr(hf_encoder_output_np)}"
    )
    decoder_precision, decoder_max_diff, decoder_min_diff = find_precision(
        keras_decoder_output_np, hf_decoder_logits_np
    )
    print("\n--- Backbone Decoder Output Validation ---")
    print(
        f"Backbone decoder outputs match up to {decoder_precision} precision."
    )
    print(f"Maximum difference: {decoder_max_diff}")
    print(f"Minimum difference: {decoder_min_diff}")
    if decoder_max_diff > 1e-4:
        print("Warning: Backbone decoder outputs do not match within 1e-4")
    print(
        f"Keras Backbone Decoder MD5: "
        f"{compute_md5_checksum(keras_decoder_output_np)}"
    )
    print(
        f"Hugging Face Decoder MD5: "
        f"{compute_md5_checksum(hf_decoder_logits_np)}"
    )
    print(f"Keras Backbone Decoder Shape: {keras_decoder_output_np.shape}")
    print(f"Hugging Face Decoder Shape: {hf_decoder_logits_np.shape}")
    print(
        f"Keras Backbone Decoder Sample: "
        f"{shorthand_repr(keras_decoder_output_np)}"
    )
    print(
        f"Hugging Face Decoder Sample: {shorthand_repr(hf_decoder_logits_np)}"
    )
    return keras_encoder_output_np, keras_decoder_output_np


# Main conversion and validation.
def convert_and_validate_moonshine(config):
    print(f"🏃 Converting {config['name']}")

    # Check backend compatibility.
    if keras.backend.backend() not in ["tensorflow", "jax", "torch"]:
        raise ValueError(
            "Unsupported backend. Please use TensorFlow, JAX, or PyTorch"
        )

    # Download weights.
    weights_dir = download_weights(config["weights_dir"])
    weights_dir = weights_dir + "/" + weights_dir.split("_")[-1]
    encoder_weights_path = os.path.join(weights_dir, "encoder.weights.h5")
    preprocessor_weights_path = os.path.join(
        weights_dir, "preprocessor.weights.h5"
    )
    decoder_weights_path = os.path.join(weights_dir, "decoder.weights.h5")

    # Build models.
    keras_encoder = build_moonshine_encoder(config)
    keras_preprocessor = build_moonshine_preprocessor(config)
    keras_decoder = build_moonshine_decoder(config)
    keras_backbone = build_moonshine_backbone(config)
    print("✅ Keras models built")

    # Load and map weights.
    encoder_weights = load_h5_weights(encoder_weights_path)
    map_encoder_weights(encoder_weights, keras_encoder)
    map_preprocessor_weights(preprocessor_weights_path, keras_preprocessor)
    map_decoder_weights(decoder_weights_path, keras_decoder)

    # Map weights to backbone components.
    keras_backbone.encoder.set_weights(keras_encoder.get_weights())
    keras_backbone.decoder.set_weights(keras_decoder.get_weights())
    print("✅ Weights converted")

    # Load Hugging Face model for validation.
    hf_model = AutoModel.from_pretrained(config["hf_model_id"])
    hf_encoder = hf_model.encoder
    hf_decoder = hf_model.decoder
    print("✅ Hugging Face model loaded")

    # Prepare test inputs.
    batch_size, audio_length = 1, 16000
    raw_audio_np = np.random.randn(batch_size, audio_length).astype("float32")
    raw_audio = keras.ops.convert_to_tensor(raw_audio_np)
    raw_audio_torch = torch.from_numpy(raw_audio_np)

    # Validate models.
    keras_preprocessed = validate_preprocessor(
        keras_preprocessor, hf_encoder, raw_audio, raw_audio_torch
    )
    keras_encoder_output, hf_encoder_output = validate_encoder(
        keras_encoder, hf_encoder, keras_preprocessed, raw_audio_torch
    )
    validate_decoder(
        keras_decoder, hf_decoder, keras_encoder_output, hf_encoder_output
    )
    print("✅ Individual components validated")

    # Validate the backbone.
    print("\n=== MoonshineBackbone Validation ===")
    validate_backbone(
        keras_backbone, hf_model, keras_preprocessed, raw_audio_torch
    )

    # Add this after the existing validation steps.
    print("\n=== Full Model Forward Pass Comparison ===")
    compare_model_outputs(
        keras_backbone, hf_model, raw_audio_np, keras_preprocessed
    )
    print(f"🏁 Conversion and validation completed for {config['name']}")


def main():
    config = {
        "name": "Moonshine Base",
        "hf_model_id": "UsefulSensors/moonshine-base",
        "weights_dir": "pt_moonshine_base",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "dtype": "float32",
        "do_normalize": False,
        "feedforward_expansion_factor": 4,
        "feature_size": 1,
        "filter_dim": 416,
        "hidden_dim": 416,
        "initializer_range": 0.02,
        "intermediate_dim": 1664,
        "encoder_num_layers": 8,
        "decoder_num_layers": 8,
        "encoder_num_heads": 8,
        "decoder_num_heads": 8,
        "vocabulary_size": 32768,
        "max_position_embeddings": 194,
        "padding_value": 0.0,
        "partial_rotary_factor": 0.62,
        "encoder_use_swiglu_activation": False,
        "decoder_use_swiglu_activation": True,
        "return_attention_mask": True,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "sampling_rate": 16000,
    }
    keras.utils.set_random_seed(42)
    convert_and_validate_moonshine(config)


if __name__ == "__main__":
    main()
