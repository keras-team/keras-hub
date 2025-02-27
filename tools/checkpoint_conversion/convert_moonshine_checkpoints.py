"""
Moonshine Weights Conversion & Verification Script

This script converts weights from the original H5 files (encoder.weights.h5,
preprocessor.weights.h5, and decoder.weights.h5) into the KerasHub Moonshine
format. The weights can be found at:
    https://huggingface.co/UsefulSensors/moonshine/tree/main/base

The HF config for Moonshine is available at:
    https://huggingface.co/UsefulSensors/moonshine-base/blob/main/config.json

After conversion, the script loads the original HF model using:
    from transformers import AutoModel
and compares outputs of the converted encoder and decoder to HF implementation
by computing MD5 checksums.
"""

import hashlib
import os

import h5py
import keras
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel

from keras_hub.src.models.moonshine.moonshine_decoder import MoonshineDecoder
from keras_hub.src.models.moonshine.moonshine_encoder import MoonshineEncoder
from keras_hub.src.models.moonshine.moonshine_preprocessor import (
    MoonshinePreprocessor,
)

# Set random seed.
keras.utils.set_random_seed(42)
# Check if backend is supported.
if keras.backend.backend() not in ["tensorflow", "jax", "torch"]:
    raise ValueError(
        "Unsupported backend. Please use TensorFlow, JAX, or PyTorch"
    )

# -----------------
# Utility Functions
# -----------------


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


def download_weights():
    repo_id = "UsefulSensors/moonshine"
    base_dir = "pt_moonshine"
    os.makedirs(base_dir, exist_ok=True)
    files = [
        "encoder.weights.h5",
        "preprocessor.weights.h5",
        "decoder.weights.h5",
    ]
    for fname in files:
        file_path = os.path.join(base_dir, fname)
        if not os.path.exists(file_path):
            print(f"Downloading {fname}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=f"base/{fname}",
                local_dir=base_dir,
                local_dir_use_symlinks=False,
            )
        else:
            print(f"{fname} already exists. Skipping download.")


def shorthand_repr(tensor, max_elements=3):
    import numpy as np

    tensor = np.array(tensor)
    if tensor.ndim == 1:
        row_list = list(tensor)
        if len(row_list) > max_elements:
            row_str = ", ".join(map(str, row_list[:max_elements])) + " ..."
        else:
            row_str = ", ".join(map(str, row_list))
        return f"[{row_str}]"

    lines = []
    for row in tensor:
        row_list = list(row)
        if len(row_list) > max_elements:
            row_str = ", ".join(map(str, row_list[:max_elements])) + " ..."
        else:
            row_str = ", ".join(map(str, row_list))
        lines.append(f"[{row_str}]")
    return "[\n " + "\n ".join(lines) + "\n]"


# ------------------------------------
# Encoder Weights Conversion Functions
# ------------------------------------


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
    try:
        orig_rot_emb = orig_weights["layers/rotary_embedding/vars/0"]
        print(f"Target rotary embedding shape: {orig_rot_emb.shape}")
        print(
            "MoonshineEncoder rotary embedding shape: "
            f"{moonshine_encoder.rotary_embedding.inv_freq.shape}"
        )
        moonshine_encoder.rotary_embedding.inv_freq.assign(orig_rot_emb)
        print("Successfully mapped rotary embedding weights")
    except KeyError as e:
        print(f"Failed to find rotary embedding weights: {e}")
        raise


def map_encoder_block_weights(orig_weights, moonshine_encoder, block_idx):
    try:
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
                print(
                    "ASSIGNED: Attention self_attention_layer._query_dense."
                    "kernel.assign"
                )
            elif weight_type == "key":
                moonshine_block.self_attention_layer._key_dense.kernel.assign(
                    weight
                )
                print(
                    "ASSIGNED: Attention self_attention_layer._key_dense."
                    "kernel.assign"
                )
            elif weight_type == "value":
                moonshine_block.self_attention_layer._value_dense.kernel.assign(
                    weight
                )
                print(
                    "ASSIGNED: Attention self_attention_layer._value_dense."
                    "kernel.assign"
                )
            elif weight_type == "output":
                moonshine_block.self_attention_layer._output_dense.kernel.assign(  # noqa: E501
                    weight
                )
                print(
                    "ASSIGNED: Attention self_attention_layer._output_dense."
                    "kernel.assign"
                )

        moonshine_block.self_attention_layer_norm.gamma.assign(
            orig_weights[f"{base_prefix}/layer_normalization/vars/0"]
        )
        print("ASSIGNED: Attention self_attention_layer_norm.gamma.assign")
        moonshine_block.feedforward_layer_norm.gamma.assign(
            orig_weights[f"{base_prefix}/layer_normalization_1/vars/0"]
        )
        print("ASSIGNED: FF feedforward_layer_norm.gamma.assign")
        moonshine_block.feedforward.dense_1.kernel.assign(
            orig_weights[f"{ff_prefix}/dense/vars/0"]
        )
        print("ASSIGNED: FF feedforward.dense_1.kernel.assign")
        moonshine_block.feedforward.dense_1.bias.assign(
            orig_weights[f"{ff_prefix}/dense/vars/1"]
        )
        print("ASSIGNED: FF feedforward.dense_1.bias.assign")
        moonshine_block.feedforward.dense_2.kernel.assign(
            orig_weights[f"{ff_prefix}/dense_1/vars/0"]
        )
        print("ASSIGNED: FF feedforward.dense_2.kernel.assign")
        moonshine_block.feedforward.dense_2.bias.assign(
            orig_weights[f"{ff_prefix}/dense_1/vars/1"]
        )
        print("ASSIGNED: FF feedforward.dense_2.bias.assign")

        print(f"Successfully mapped encoder block {block_idx} weights")
    except KeyError as e:
        print(f"Failed to map encoder block {block_idx} weights: {str(e)}")
        raise
    except ValueError as e:
        print(f"Shape mismatch in encoder block {block_idx}: {str(e)}")
        raise


def map_final_layer_norm_weights(orig_weights, moonshine_encoder):
    try:
        moonshine_encoder.final_layer_norm.gamma.assign(
            orig_weights["layers/layer_normalization/vars/0"]
        )
        print("Successfully mapped final layer norm weights")
    except KeyError:
        pass


def convert_encoder_weights(h5_path, moonshine_encoder):
    print(f"Loading encoder weights from {h5_path}")
    try:
        orig_weights = load_h5_weights(h5_path)
        map_rotary_embedding_weights(orig_weights, moonshine_encoder)
        for block_idx in range(moonshine_encoder.num_layers):
            map_encoder_block_weights(
                orig_weights, moonshine_encoder, block_idx
            )
        try:
            moonshine_encoder.final_layer_norm.gamma.assign(
                orig_weights["layers/layer_normalization/vars/0"]
            )
            print("Successfully mapped final layer norm weights")
        except KeyError:
            print("Final layer norm weights not found, skipping...")
        print("Encoder weight conversion completed successfully")
    except Exception as e:
        print(f"Encoder weight conversion failed: {str(e)}")
        raise


def verify_encoder_conversion(moonshine_encoder, test_input):
    try:
        sequence = keras.random.normal(test_input)
        seq_length = keras.ops.convert_to_tensor([test_input[1]], dtype="int32")
        output = moonshine_encoder([sequence, seq_length])
        print(f"Encoder verification successful. Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"Encoder verification failed: {e}")
        return False


# -----------------------------------------
# Preprocessor Weights Conversion Functions
# -----------------------------------------


def convert_preprocessor_weights(weights_path, dim=416):
    with h5py.File(weights_path, "r") as f:
        base_path = "layers/sequential/layers/"
        try:
            weights = [
                np.array(f[f"{base_path}conv1d/vars/0"]),
                # === GroupNormalization layer ===
                np.array(f[f"{base_path}group_normalization/vars/0"]),  # gamma
                np.array(f[f"{base_path}group_normalization/vars/1"]),  # beta
                # === Conv1D layer (first convolutional layer) ===
                np.array(f[f"{base_path}conv1d_1/vars/0"]),  # kernel
                np.array(f[f"{base_path}conv1d_1/vars/1"]),  # bias
                # === Conv1D_1 layer (second convolutional layer) ===
                np.array(f[f"{base_path}conv1d_2/vars/0"]),  # kernel
                np.array(f[f"{base_path}conv1d_2/vars/1"]),  # bias
            ]
            expected_shapes = [
                (127, 1, dim),  # conv1 kernel
                (dim,),  # group_norm gamma
                (dim,),  # group_norm beta
                (7, dim, 2 * dim),  # conv2 kernel
                (2 * dim,),  # conv2 bias
                (3, 2 * dim, dim),  # conv3 kernel
                (dim,),  # conv3 bias
            ]
            for w, expected_shape in zip(weights, expected_shapes):
                if w.shape != expected_shape:
                    raise ValueError(
                        f"Weight shape mismatch. Expected {expected_shape}, got"
                        f" {w.shape}"
                    )
            return weights
        except KeyError as e:
            raise KeyError(
                f"Failed to find expected weight in h5 file: {e}."
                "Weight structure may have changed."
            )
        except Exception as e:
            raise Exception(
                f"Error during preprocessor weight conversion: {e}."
            )


def apply_converted_preprocessor_weights(model, converted_weights):
    try:
        model.preprocess.set_weights(converted_weights)
        print("Weights successfully applied to MoonshinePreprocessor.")
    except ValueError as e:
        raise ValueError(f"Failed to apply weights: {e}")


# ------------------------------------
# Decoder Weights Conversion Functions
# ------------------------------------


def convert_decoder_weights(weights_path, moonshine_decoder):
    """Convert weights from H5 file to Moonshine decoder model"""
    try:
        with h5py.File(weights_path, "r") as f:
            moonshine_decoder.embedding_layer.embeddings.assign(
                f["layers/reversible_embedding/vars/0"][()]
            )
            print("ASSIGNED: embedding_layer.embeddings.assign")
            moonshine_decoder.rotary_embedding.inv_freq.assign(
                f["layers/rotary_embedding/vars/0"][()]
            )
            print("ASSIGNED: rotary_embedding_layer.inv_freq.assign")
            moonshine_decoder.post_norm.gamma.assign(
                f["layers/layer_normalization/vars/0"][()]
            )
            print("ASSIGNED: post_norm.gamma.assign")
            for block_idx in range(moonshine_decoder.num_layers):
                map_decoder_block_weights(f, moonshine_decoder, block_idx)
        print("Decoder weight conversion completed successfully")
        return True
    except Exception as e:
        print(f"Decoder weight conversion failed: {str(e)}")
        raise


def map_decoder_block_weights(h5_file, moonshine_decoder, block_idx):
    try:
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
                moonshine_block.self_attention._query_dense.kernel.assign(
                    weight
                )
                print("ASSIGNED: Self-Attention query_dense.kernel.assign")
            elif weight_type == "key":
                moonshine_block.self_attention._key_dense.kernel.assign(weight)
                print("ASSIGNED: Self-Attention key_dense.kernel.assign")
            elif weight_type == "value":
                moonshine_block.self_attention._value_dense.kernel.assign(
                    weight
                )
                print("ASSIGNED: Self-Attention value_dense.kernel.assign")
            elif weight_type == "output":
                moonshine_block.self_attention._output_dense.kernel.assign(
                    weight
                )
                print("ASSIGNED: Self-Attention output_dense.kernel.assign")
        cross_attention_mappings = {
            "query": f"{cross_attention_prefix}/query_dense/vars/0",
            "key": f"{cross_attention_prefix}/key_dense/vars/0",
            "value": f"{cross_attention_prefix}/value_dense/vars/0",
            "output": f"{cross_attention_prefix}/output_dense/vars/0",
        }
        for weight_type, path in cross_attention_mappings.items():
            weight = h5_file[path][()]
            if weight_type == "query":
                moonshine_block.cross_attention._query_dense.kernel.assign(
                    weight
                )
                print("ASSIGNED: Cross-Attention query_dense.kernel.assign")
            elif weight_type == "key":
                moonshine_block.cross_attention._key_dense.kernel.assign(weight)
                print("ASSIGNED: Cross-Attention key_dense.kernel.assign")
            elif weight_type == "value":
                moonshine_block.cross_attention._value_dense.kernel.assign(
                    weight
                )
                print("ASSIGNED: Cross-Attention value_dense.kernel.assign")
            elif weight_type == "output":
                moonshine_block.cross_attention._output_dense.kernel.assign(
                    weight
                )
                print("ASSIGNED: Cross-Attention output_dense.kernel.assign")
        moonshine_block.norm1.gamma.assign(
            h5_file[f"{base_prefix}/layer_normalization/vars/0"][()]
        )
        print("ASSIGNED: norm1.gamma.assign")
        moonshine_block.norm2.gamma.assign(
            h5_file[f"{base_prefix}/layer_normalization_1/vars/0"][()]
        )
        print("ASSIGNED: norm2.gamma.assign")
        moonshine_block.norm3.gamma.assign(
            h5_file[f"{base_prefix}/layer_normalization_2/vars/0"][()]
        )
        print("ASSIGNED: norm3.gamma.assign")
        moonshine_block.ff.dense_1.kernel.assign(
            h5_file[f"{ff_prefix}/dense/vars/0"][()]
        )
        print("ASSIGNED: FF dense_1.kernel.assign")
        moonshine_block.ff.dense_1.bias.assign(
            h5_file[f"{ff_prefix}/dense/vars/1"][()]
        )
        print("ASSIGNED: FF dense_1.bias.assign")
        moonshine_block.ff.dense_2.kernel.assign(
            h5_file[f"{ff_prefix}/dense_1/vars/0"][()]
        )
        print("ASSIGNED: FF dense_2.kernel.assign")
        moonshine_block.ff.dense_2.bias.assign(
            h5_file[f"{ff_prefix}/dense_1/vars/1"][()]
        )
        print("ASSIGNED: FF dense_2.bias.assign")
        print(f"Successfully mapped decoder block {block_idx} weights")
    except KeyError as e:
        print(f"Failed to map decoder block {block_idx} weights: {str(e)}")
        raise
    except ValueError as e:
        print(f"Shape mismatch in decoder block {block_idx}: {str(e)}")
        raise


def verify_decoder_conversion(moonshine_decoder, test_input_shape):
    try:
        batch_size, seq_length, hidden_dim = test_input_shape
        dummy_token_ids = keras.random.uniform(
            (batch_size, seq_length),
            minval=0,
            maxval=moonshine_decoder.vocabulary_size,
            dtype="float32",
        )
        dummy_context = keras.random.uniform(
            (batch_size, seq_length, hidden_dim), dtype="float32"
        )
        dummy_seq_len = keras.ops.convert_to_tensor([seq_length], dtype="int32")
        _ = moonshine_decoder(
            [dummy_token_ids, dummy_context, dummy_seq_len], training=False
        )
        print("Decoder weight conversion verified successfully!")
        return True
    except Exception as e:
        print(f"Decoder verification failed: {str(e)}")
        raise


# -----------------------------
# HF Model Comparison Functions
# -----------------------------


def compare_with_hf(encoder_keras, decoder_keras, preprocessor_keras):
    """
    Loads the original HF model via AutoModel.from_pretrained and compares the
    outputs (by MD5 checksum) of its encoder, decoder, and preprocessor with the
    converted implementations.
    """
    print("\n================ HF Model Comparison ===============\n")
    hf_model = AutoModel.from_pretrained("UsefulSensors/moonshine-base")
    hf_encoder = hf_model.encoder
    hf_decoder = hf_model.decoder

    batch_size, audio_length = 1, 16000
    raw_audio_np = np.random.randn(batch_size, audio_length).astype("float32")
    raw_audio_torch = torch.from_numpy(raw_audio_np)
    raw_audio = keras.ops.convert_to_tensor(raw_audio_np)
    raw_audio_input = raw_audio
    preprocessed_output = preprocessor_keras(raw_audio_input)

    # ----- Compare Preprocessor Outputs -----
    if keras.backend.backend() == "torch":
        preprocessed = preprocessed_output.detach().numpy()
    elif keras.backend.backend() == "tensorflow":
        preprocessed = preprocessed_output.numpy()
    elif keras.backend.backend() == "jax":
        import jax

        preprocessed = jax.device_get(preprocessed_output)

    with torch.no_grad():
        hf_preprocessed = hf_encoder.conv1(raw_audio_torch.unsqueeze(1))
        hf_preprocessed = torch.tanh(hf_preprocessed)
        hf_preprocessed = hf_encoder.groupnorm(hf_preprocessed)
        hf_preprocessed = F.gelu(hf_encoder.conv2(hf_preprocessed))
        hf_preprocessed = F.gelu(hf_encoder.conv3(hf_preprocessed))
        hf_preprocessed = hf_preprocessed.permute(0, 2, 1).detach().numpy()

    print("Preprocessor Outputs MD5 Checksum:")
    print("KerasHub Preprocessor Output:", preprocessed)
    print("HF Preprocessor Output:", hf_preprocessed)
    print(
        "KerasHub Preprocessor MD5 Checksum:",
        compute_md5_checksum(preprocessed),
    )
    print(
        "HF Preprocessor MD5 Checksum:      ",
        compute_md5_checksum(hf_preprocessed),
    )
    print("HF Preprocessor Output Shape:", hf_preprocessed.shape)
    print("KerasHub Preprocessor Output Shape:", preprocessed.shape)
    precision, max_diff, min_diff = find_precision(
        preprocessed, hf_preprocessed
    )
    print(f"Preprocessor outputs match up to {precision} precision.")
    print(f"Maximum difference: {max_diff}")
    print(f"Minimum difference: {min_diff}")
    print()

    # ----- Compare Encoder Outputs -----
    seq_length = keras.ops.convert_to_tensor(
        [preprocessed.shape[1]], dtype="int32"
    )

    encoder_input = [preprocessed, seq_length]
    kerashub_encoder_output_tensor = encoder_keras(encoder_input)
    if keras.backend.backend() == "torch":
        kerashub_encoder_output = (
            kerashub_encoder_output_tensor.detach().numpy()
        )
    elif keras.backend.backend() == "tensorflow":
        kerashub_encoder_output = kerashub_encoder_output_tensor.numpy()
    elif keras.backend.backend() == "jax":
        import jax

        kerashub_encoder_output = jax.device_get(kerashub_encoder_output_tensor)

    hf_encoder_output = (
        hf_encoder(raw_audio_torch).last_hidden_state.detach().numpy()
    )

    print("Encoder Outputs MD5 Checksum:")
    print("KerasHub Encoder Output:", kerashub_encoder_output)
    print("HF Encoder Output:", hf_encoder_output)
    print(
        "KerasHub Encoder MD5 Checksum:",
        compute_md5_checksum(kerashub_encoder_output),
    )
    print(
        "HF Encoder MD5 Checksum:      ",
        compute_md5_checksum(hf_encoder_output),
    )
    print("HF Encoder Output Shape:", hf_encoder_output.shape)
    print("KerasHub Encoder Output Shape:", kerashub_encoder_output.shape)
    precision, max_diff, min_diff = find_precision(
        kerashub_encoder_output, hf_encoder_output
    )
    print(f"Encoder outputs match up to {precision} precision.")
    print(f"Maximum difference: {max_diff}")
    print(f"Minimum difference: {min_diff}")
    print()

    # ----- Compare Decoder Outputs -----
    token_seq_length = 32
    test_token_ids_torch = torch.randint(
        0, 32768, (batch_size, token_seq_length), dtype=torch.int32
    )
    test_context_torch = torch.from_numpy(
        hf_encoder_output
    )  # Shape: [1, 40, 416]
    test_context = keras.ops.convert_to_tensor(
        kerashub_encoder_output, dtype="float32"
    )  # Shape: [1, 40, 416]

    position_ids = torch.arange(0, token_seq_length, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand(
        batch_size, -1
    )  # [batch_size, seq_length]

    hf_decoder_output = (
        hf_decoder(
            input_ids=test_token_ids_torch,
            encoder_hidden_states=test_context_torch,
            position_ids=position_ids,
        )
        .last_hidden_state.detach()
        .numpy()
    )

    test_token_ids = keras.ops.convert_to_tensor(
        test_token_ids_torch.numpy(), dtype="int32"
    )
    test_context = keras.ops.convert_to_tensor(
        test_context_torch.numpy(), dtype="float32"
    )
    test_seq_len = keras.ops.convert_to_tensor(
        [token_seq_length], dtype="int32"
    )

    if keras.backend.backend() == "torch":
        decoder_input = [
            torch.from_numpy(test_token_ids_torch.numpy()),
            torch.from_numpy(test_context_torch.numpy()),
            torch.tensor([token_seq_length], dtype=torch.int32),
        ]
        decoder_output = decoder_keras(decoder_input, training=False)
        if isinstance(decoder_output, list):
            kerashub_decoder_output = decoder_output[0].detach().numpy()
        else:
            kerashub_decoder_output = decoder_output.detach().numpy()
    elif keras.backend.backend() == "tensorflow":
        decoder_input = [test_token_ids, test_context, test_seq_len]
        decoder_output = decoder_keras(decoder_input, training=False)
        if isinstance(decoder_output, list):
            kerashub_decoder_output = decoder_output[0].numpy()
        else:
            kerashub_decoder_output = decoder_output.numpy()
    elif keras.backend.backend() == "jax":
        import jax

        decoder_input = [
            jax.device_put(test_token_ids),
            jax.device_put(test_context),
            jax.device_put(test_seq_len),
        ]
        decoder_output = decoder_keras(decoder_input, training=False)
        if isinstance(decoder_output, list):
            kerashub_decoder_output = jax.device_get(decoder_output[0])
        else:
            kerashub_decoder_output = jax.device_get(decoder_output)

    print("Decoder Outputs MD5 Checksum:")
    print("KerasHub Decoder Output:", kerashub_decoder_output)
    print("HF Decoder Output:", hf_decoder_output)
    print(
        "KerasHub Decoder MD5 Checksum:",
        compute_md5_checksum(kerashub_decoder_output),
    )
    print(
        "HF Decoder MD5 Checksum:      ",
        compute_md5_checksum(hf_decoder_output),
    )
    print("HF Decoder Output Shape:", hf_decoder_output.shape)
    print("KerasHub Decoder Output Shape:", kerashub_decoder_output.shape)
    precision, max_diff, min_diff = find_precision(
        kerashub_decoder_output, hf_decoder_output
    )
    print(f"Decoder outputs match up to {precision} precision.")
    print(f"Maximum difference: {max_diff}")
    print(f"Minimum difference: {min_diff}")
    print()


# -------------------------------------
# Main Conversion & Comparison Function
# -------------------------------------


def main():
    # ----- Encoder Conversion -----
    download_weights()
    print(
        "\n================ Encoder Weights Conversion Script ===============\n"
    )
    encoder_weights_path = "pt_moonshine/base/encoder.weights.h5"
    moonshine_encoder = MoonshineEncoder(
        num_layers=8,
        hidden_dim=416,
        intermediate_dim=1664,
        num_heads=8,
        feedforward_expansion_factor=4,
        use_swiglu_activation=False,
        max_position_embeddings=194,
    )
    moonshine_encoder.build(input_shape=[(1, 50, 416), (1,)])
    try:
        convert_encoder_weights(encoder_weights_path, moonshine_encoder)
        verify_encoder_conversion(moonshine_encoder, test_input=(1, 50, 416))
    except Exception as e:
        print(e)

    # ----- Preprocessor Conversion -----
    print(
        "\n=========== Preprocessor Weights Conversion Script ===============\n"
    )
    preprocessor_weights_path = "pt_moonshine/base/preprocessor.weights.h5"
    DIM = 416
    try:
        converted_weights = convert_preprocessor_weights(
            preprocessor_weights_path, DIM
        )
        moonshine_preprocessor = MoonshinePreprocessor(dim=DIM)
        apply_converted_preprocessor_weights(
            moonshine_preprocessor, converted_weights
        )
        print("Preprocessor weight conversion completed successfully.")
    except Exception as e:
        print(f"Error during preprocessor conversion process: {e}")

    # ----- Decoder Conversion -----
    print(
        "\n================ Decoder Weights Conversion Script ===============\n"
    )
    decoder_weights_path = "pt_moonshine/base/decoder.weights.h5"
    moonshine_decoder = MoonshineDecoder(
        num_layers=8,
        hidden_dim=416,
        intermediate_dim=1664,
        num_heads=8,
        vocabulary_size=32768,
        feedforward_expansion_factor=4,
        use_swiglu_activation=True,
        max_position_embeddings=194,
    )
    dummy_token_ids = keras.random.uniform(
        (1, 32),
        minval=0,
        maxval=moonshine_decoder.vocabulary_size,
        dtype="float32",
    )
    dummy_context = keras.random.uniform((1, 40, 416), dtype="float32")
    dummy_seq_len = keras.ops.convert_to_tensor([32], dtype="int32")
    _ = moonshine_decoder(
        [dummy_token_ids, dummy_context, dummy_seq_len], training=False
    )
    try:
        convert_decoder_weights(decoder_weights_path, moonshine_decoder)
        verify_decoder_conversion(
            moonshine_decoder, test_input_shape=(1, 32, 416)
        )
    except Exception as e:
        print(e)

    # ----- Compare with Original HF Implementation -----
    compare_with_hf(
        moonshine_encoder, moonshine_decoder, moonshine_preprocessor
    )


if __name__ == "__main__":
    main()
