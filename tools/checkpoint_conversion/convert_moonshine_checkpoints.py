"""
Convert Moonshine checkpoints to KerasHub format and provide a complete
end-to-end example.

The weights are sourced from:
https://huggingface.co/UsefulSensors/moonshine/tree/main/base

The Hugging Face config is available at:
https://huggingface.co/UsefulSensors/moonshine-base/blob/main/config.json

Usage:
```shell
python -m tools.checkpoint_conversion.convert_moonshine_checkpoints
```
"""

import json
import os

import h5py
import keras

try:
    import librosa
except ImportError:
    raise ImportError(
        "Moonshine ASR system requires librosa as a dependency. Please install "
        "it using 'pip install librosa' before proceeding."
    )
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel

from keras_hub.src.models.moonshine.moonshine_audio_converter import (
    MoonshineAudioConverter,
)
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_for_conditional_generation import (  # noqa: E501
    MoonshineForConditionalGeneration,
)
from keras_hub.src.models.moonshine.moonshine_tokenizer import (
    MoonshineTokenizer,
)

# Set random seed for reproducibility.
keras.utils.set_random_seed(50)


# Utility function to convert tensors to NumPy based on backend.
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


# Init.
PRESET_NAME = "moonshine-base"
PRESET = "UsefulSensors/moonshine-base"
EXTRACT_DIR = "./{}"

extract_dir = EXTRACT_DIR.format(PRESET_NAME)
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Download and load config.
config_path = os.path.join(extract_dir, "config.json")
response = requests.get(f"https://huggingface.co/{PRESET}/raw/main/config.json")
open(config_path, "wb").write(response.content)

cfg = {}
with open(config_path, "r") as pt_cfg_handler:
    pt_cfg = json.load(pt_cfg_handler)

# Setup Moonshine config.
cfg["vocabulary_size"] = pt_cfg["vocab_size"]
cfg["num_layers"] = pt_cfg["encoder_num_hidden_layers"]
cfg["filter_dim"] = pt_cfg["hidden_size"]
cfg["hidden_dim"] = pt_cfg["hidden_size"]
cfg["intermediate_dim"] = pt_cfg["intermediate_size"]
cfg["max_sequence_length"] = pt_cfg["max_position_embeddings"]
cfg["partial_rotary_factor"] = pt_cfg["partial_rotary_factor"]
cfg["rope_theta"] = pt_cfg["rope_theta"]
cfg["encoder_num_layers"] = pt_cfg["encoder_num_hidden_layers"]
cfg["decoder_num_layers"] = pt_cfg["decoder_num_hidden_layers"]
cfg["encoder_num_heads"] = pt_cfg.get("encoder_num_attention_heads", 8)
cfg["decoder_num_heads"] = pt_cfg.get("decoder_num_attention_heads", 8)
cfg["feedforward_expansion_factor"] = 4
cfg["attention_bias"] = pt_cfg["attention_bias"]
cfg["attention_dropout"] = pt_cfg["attention_dropout"]
cfg["dtype"] = pt_cfg["torch_dtype"]
cfg["decoder_use_swiglu_activation"] = pt_cfg["decoder_hidden_act"] == "silu"
cfg["encoder_use_swiglu_activation"] = pt_cfg["encoder_hidden_act"] == "silu"
cfg["initializer_range"] = pt_cfg["initializer_range"]
cfg["rope_scaling"] = pt_cfg["rope_scaling"]

# Taken from: https://huggingface.co/UsefulSensors/moonshine-base/blob/main/preprocessor_config.json.
cfg["sampling_rate"] = 16000
cfg["padding_value"] = 0.0
cfg["do_normalize"] = False
cfg["return_attention_mask"] = True

# Download weights.
weights_dir = os.path.join(extract_dir, "weights")
repo_id = "UsefulSensors/moonshine"
files = ["encoder.weights.h5", "preprocessor.weights.h5", "decoder.weights.h5"]
for fname in files:
    file_path = os.path.join(weights_dir, fname)
    if not os.path.exists(file_path):
        print(f"Downloading {fname} to {file_path}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=f"base/{fname}",
            local_dir=weights_dir,
        )
weights_dir = os.path.join(weights_dir, "base")

encoder_weights_path = os.path.join(weights_dir, "encoder.weights.h5")
preprocessor_weights_path = os.path.join(weights_dir, "preprocessor.weights.h5")
decoder_weights_path = os.path.join(weights_dir, "decoder.weights.h5")

# Load Hugging Face model.
hf_model = AutoModel.from_pretrained(PRESET)
hf_model.eval()


# Load H5 weights into dictionaries.
def load_h5_weights(filepath):
    with h5py.File(filepath, "r") as f:
        weights = {}

        def recursive_load(group, prefix=""):
            for key in group.keys():
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(group[key], h5py.Dataset):
                    weights[path] = np.array(group[key])
                else:
                    recursive_load(group[key], path)

        recursive_load(f)
    return weights


hf_wts_encoder = load_h5_weights(encoder_weights_path)
hf_wts_preprocessor = load_h5_weights(preprocessor_weights_path)
hf_wts_decoder = load_h5_weights(decoder_weights_path)

# Build Keras models.
backbone = MoonshineBackbone(
    vocabulary_size=cfg["vocabulary_size"],
    encoder_num_layers=cfg["encoder_num_layers"],
    decoder_num_layers=cfg["decoder_num_layers"],
    hidden_dim=cfg["hidden_dim"],
    intermediate_dim=cfg["intermediate_dim"],
    encoder_num_heads=cfg["encoder_num_heads"],
    decoder_num_heads=cfg["decoder_num_heads"],
    feedforward_expansion_factor=cfg["feedforward_expansion_factor"],
    decoder_use_swiglu_activation=cfg["decoder_use_swiglu_activation"],
    encoder_use_swiglu_activation=cfg["encoder_use_swiglu_activation"],
    max_position_embeddings=cfg["max_sequence_length"],
    partial_rotary_factor=cfg["partial_rotary_factor"],
    dropout=cfg["attention_dropout"],
    initializer_range=cfg["initializer_range"],
    rope_theta=cfg["rope_theta"],
    attention_bias=cfg["attention_bias"],
    attention_dropout=cfg["attention_dropout"],
    rope_scaling=cfg["rope_scaling"],
    dtype=cfg["dtype"],
)

# Build preprocessor.
keras_audio_converter = MoonshineAudioConverter(
    filter_dim=cfg["filter_dim"],
    initializer_range=cfg["initializer_range"],
    sampling_rate=cfg["sampling_rate"],
    padding_value=cfg["padding_value"],
    do_normalize=cfg["do_normalize"],
    return_attention_mask=cfg["return_attention_mask"],
)
tokenizer = MoonshineTokenizer(
    proto="keras_hub/src/tests/test_data/llama2_tokenizer_full.spm"
)
keras_model = MoonshineForConditionalGeneration(
    backbone=backbone,
    audio_converter=keras_audio_converter,
    tokenizer=tokenizer,
)

# Build the model with dummy data.
dummy_audio = np.zeros((1, 16000), dtype="float32")
dummy_token_ids = np.zeros((1, 32), dtype="int32")
dummy_inputs = {"audio": dummy_audio, "token_ids": dummy_token_ids}
keras_model(dummy_inputs)

# Assign preprocessor weights.
base_path = "layers/sequential/layers/"
weights = [
    hf_wts_preprocessor[f"{base_path}conv1d/vars/0"],  # conv1 kernel
    hf_wts_preprocessor[f"{base_path}group_normalization/vars/0"],  # gamma
    hf_wts_preprocessor[f"{base_path}group_normalization/vars/1"],  # beta
    hf_wts_preprocessor[f"{base_path}conv1d_1/vars/0"],  # conv2 kernel
    hf_wts_preprocessor[f"{base_path}conv1d_1/vars/1"],  # conv2 bias
    hf_wts_preprocessor[f"{base_path}conv1d_2/vars/0"],  # conv3 kernel
    hf_wts_preprocessor[f"{base_path}conv1d_2/vars/1"],  # conv3 bias
]
keras_model.audio_converter.set_weights(weights)

# Assign encoder weights.
keras_model.backbone.encoder_rotary_embedding.inv_freq.assign(
    hf_wts_encoder["layers/rotary_embedding/vars/0"]
)

for layer_index in range(cfg["encoder_num_layers"]):
    if layer_index == 0:
        base_prefix = "layers/functional/layers"
    else:
        base_prefix = f"layers/functional_{layer_index}/layers"
    attention_prefix = f"{base_prefix}/mha_with_rope"
    ff_prefix = f"{base_prefix}/functional/layers/sequential/layers"

    # Attention weights.
    keras_model.backbone.encoder_blocks[
        layer_index
    ].self_attention_layer._query_dense.kernel.assign(
        hf_wts_encoder[f"{attention_prefix}/query_dense/vars/0"]
    )
    keras_model.backbone.encoder_blocks[
        layer_index
    ].self_attention_layer._key_dense.kernel.assign(
        hf_wts_encoder[f"{attention_prefix}/key_dense/vars/0"]
    )
    keras_model.backbone.encoder_blocks[
        layer_index
    ].self_attention_layer._value_dense.kernel.assign(
        hf_wts_encoder[f"{attention_prefix}/value_dense/vars/0"]
    )
    keras_model.backbone.encoder_blocks[
        layer_index
    ].self_attention_layer._output_dense.kernel.assign(
        hf_wts_encoder[f"{attention_prefix}/output_dense/vars/0"]
    )

    # Layer norms.
    keras_model.backbone.encoder_blocks[
        layer_index
    ].self_attention_layer_norm.gamma.assign(
        hf_wts_encoder[f"{base_prefix}/layer_normalization/vars/0"]
    )
    keras_model.backbone.encoder_blocks[
        layer_index
    ].feedforward_layer_norm.gamma.assign(
        hf_wts_encoder[f"{base_prefix}/layer_normalization_1/vars/0"]
    )

    # Feedforward weights.
    keras_model.backbone.encoder_blocks[
        layer_index
    ].feedforward.dense_1.kernel.assign(
        hf_wts_encoder[f"{ff_prefix}/dense/vars/0"]
    )
    keras_model.backbone.encoder_blocks[
        layer_index
    ].feedforward.dense_1.bias.assign(
        hf_wts_encoder[f"{ff_prefix}/dense/vars/1"]
    )
    keras_model.backbone.encoder_blocks[
        layer_index
    ].feedforward.dense_2.kernel.assign(
        hf_wts_encoder[f"{ff_prefix}/dense_1/vars/0"]
    )
    keras_model.backbone.encoder_blocks[
        layer_index
    ].feedforward.dense_2.bias.assign(
        hf_wts_encoder[f"{ff_prefix}/dense_1/vars/1"]
    )

keras_model.backbone.encoder_final_layer_norm.gamma.assign(
    hf_wts_encoder["layers/layer_normalization/vars/0"]
)

# Assign decoder weights.
keras_model.backbone.embedding_layer.embeddings.assign(
    hf_wts_decoder["layers/reversible_embedding/vars/0"]
)
keras_model.backbone.decoder_rotary_embedding.inv_freq.assign(
    hf_wts_decoder["layers/rotary_embedding/vars/0"]
)

for layer_index in range(cfg["decoder_num_layers"]):
    if layer_index == 0:
        base_prefix = "layers/functional/layers"
    else:
        base_prefix = f"layers/functional_{layer_index}/layers"
    self_attention_prefix = f"{base_prefix}/mha_causal_with_rope"
    cross_attention_prefix = f"{base_prefix}/mha_precomputed_kv"
    ff_prefix = f"{base_prefix}/functional/layers"

    # Self-attention weights.
    keras_model.backbone.decoder_blocks[
        layer_index
    ].self_attention._query_dense.kernel.assign(
        hf_wts_decoder[f"{self_attention_prefix}/query_dense/vars/0"]
    )
    keras_model.backbone.decoder_blocks[
        layer_index
    ].self_attention._key_dense.kernel.assign(
        hf_wts_decoder[f"{self_attention_prefix}/key_dense/vars/0"]
    )
    keras_model.backbone.decoder_blocks[
        layer_index
    ].self_attention._value_dense.kernel.assign(
        hf_wts_decoder[f"{self_attention_prefix}/value_dense/vars/0"]
    )
    keras_model.backbone.decoder_blocks[
        layer_index
    ].self_attention._output_dense.kernel.assign(
        hf_wts_decoder[f"{self_attention_prefix}/output_dense/vars/0"]
    )

    # Cross-attention weights.
    keras_model.backbone.decoder_blocks[
        layer_index
    ].cross_attention._query_dense.kernel.assign(
        hf_wts_decoder[f"{cross_attention_prefix}/query_dense/vars/0"]
    )
    keras_model.backbone.decoder_blocks[
        layer_index
    ].cross_attention._key_dense.kernel.assign(
        hf_wts_decoder[f"{cross_attention_prefix}/key_dense/vars/0"]
    )
    keras_model.backbone.decoder_blocks[
        layer_index
    ].cross_attention._value_dense.kernel.assign(
        hf_wts_decoder[f"{cross_attention_prefix}/value_dense/vars/0"]
    )
    keras_model.backbone.decoder_blocks[
        layer_index
    ].cross_attention._output_dense.kernel.assign(
        hf_wts_decoder[f"{cross_attention_prefix}/output_dense/vars/0"]
    )

    # Layer norms.
    keras_model.backbone.decoder_blocks[layer_index].norm1.gamma.assign(
        hf_wts_decoder[f"{base_prefix}/layer_normalization/vars/0"]
    )
    keras_model.backbone.decoder_blocks[layer_index].norm2.gamma.assign(
        hf_wts_decoder[f"{base_prefix}/layer_normalization_1/vars/0"]
    )
    keras_model.backbone.decoder_blocks[layer_index].norm3.gamma.assign(
        hf_wts_decoder[f"{base_prefix}/layer_normalization_2/vars/0"]
    )

    # Feedforward weights.
    keras_model.backbone.decoder_blocks[layer_index].ff.dense_1.kernel.assign(
        hf_wts_decoder[f"{ff_prefix}/dense/vars/0"]
    )
    keras_model.backbone.decoder_blocks[layer_index].ff.dense_1.bias.assign(
        hf_wts_decoder[f"{ff_prefix}/dense/vars/1"]
    )
    keras_model.backbone.decoder_blocks[layer_index].ff.dense_2.kernel.assign(
        hf_wts_decoder[f"{ff_prefix}/dense_1/vars/0"]
    )
    keras_model.backbone.decoder_blocks[layer_index].ff.dense_2.bias.assign(
        hf_wts_decoder[f"{ff_prefix}/dense_1/vars/1"]
    )

keras_model.backbone.decoder_post_norm.gamma.assign(
    hf_wts_decoder["layers/layer_normalization/vars/0"]
)
output_dir = f"{extract_dir}/model.weights.h5"
keras_model.save_weights(output_dir)
print(f"Saved Keras model weights to {output_dir}")

# Validation: Compare Hidden States.
print("\n--- Validating Hidden States ---")
# Prepare inputs.
sample_text = [np.random.randn(16000).astype("float32")]  # Random audio sample
hf_inputs = {
    "input_values": torch.from_numpy(sample_text[0]).unsqueeze(0),
    "decoder_input_ids": torch.randint(
        0, cfg["vocabulary_size"], (1, 32), dtype=torch.int32
    ),
}
position_ids = torch.arange(0, 32, dtype=torch.long).unsqueeze(0)

# Prepare Keras inputs for backbone.
keras_preprocessed_inputs = keras_model.audio_converter(
    keras.ops.convert_to_tensor(sample_text), padding="longest"
)
encoder_input_values = keras_preprocessed_inputs["input_values"]
encoder_padding_mask = keras_preprocessed_inputs["attention_mask"]
decoder_token_ids = keras.ops.convert_to_tensor(hf_inputs["decoder_input_ids"])
decoder_padding_mask = keras.ops.cast(
    keras.ops.not_equal(decoder_token_ids, 0), "bool"
)

# Run Keras backbone.
keras_backbone_outputs = keras_model.backbone(
    {
        "encoder_input_values": encoder_input_values,
        "decoder_token_ids": decoder_token_ids,
        "encoder_padding_mask": encoder_padding_mask,
        "decoder_padding_mask": decoder_padding_mask,
    },
    training=False,
)
keras_encoder_output = to_numpy(
    keras_backbone_outputs["encoder_sequence_output"]
)
keras_decoder_output = to_numpy(
    keras_backbone_outputs["decoder_sequence_output"]
)

# Run Hugging Face model and compute outputs.
with torch.no_grad():
    hf_outputs = hf_model(
        input_values=hf_inputs["input_values"],
        decoder_input_ids=hf_inputs["decoder_input_ids"],
        decoder_position_ids=position_ids,
        output_hidden_states=True,
    )
    hf_encoder_hidden_states = hf_outputs.encoder_hidden_states[-1]
    hf_encoder_output_np = hf_encoder_hidden_states.numpy()
    hf_decoder_hidden_states = hf_outputs.last_hidden_state
    hf_decoder_output_np = hf_decoder_hidden_states.numpy()

# Compare encoder outputs.
print("Keras encoder output shape:", keras_encoder_output.shape)
print("HF encoder output shape:", hf_encoder_output_np.shape)
print("Keras decoder output shape:", keras_decoder_output.shape)
print("HF decoder output shape:", hf_decoder_output_np.shape)

# Compare decoder hidden states.
encoder_diff = np.abs(keras_encoder_output - hf_encoder_output_np)
decoder_diff = np.abs(keras_decoder_output - hf_decoder_output_np)
print("Maximum encoder difference:", np.max(encoder_diff))
print("Maximum decoder difference:", np.max(decoder_diff))
if np.max(encoder_diff) < 1e-4 and np.max(decoder_diff) < 1e-4:
    print("✅ Hidden states match within 1e-4.")
else:
    print("❌ Hidden states do not match within 1e-4.")

# Overall validation.
# End-to-end ASR example with real audio files.
print("\n--- End-to-End ASR Example ---")

# Example audio files: Generated by ElevenLabs.
# Male Voice: Bill, Eleven Multilingual v2. Speed: 1.0, Stability: 50%,
# Similarity: 75%, Style Exaggeration: None.
# Original Text Input: "Intelligence is what you use when you don't know what to
# do."
print("\nTest: Male Voice (3 Seconds)")
audio_path = "keras_hub/src/tests/test_data/male_voice_clip_3sec.wav"
audio, sr = librosa.load(audio_path, sr=cfg["sampling_rate"])
audio = audio.reshape(1, -1)
generated_ids = keras_model.generate(audio)
transcription = keras_model.tokenizer.detokenize(generated_ids[0])
print("Generated Token IDs:", generated_ids)
print("Transcription:", transcription)

# Female Voice: Rachel, Eleven Multilingual v2. Speed: 1.0, Stability: 50%,
# Similarity: 75%, Style Exaggeration: None.
# Original Text Input: "Intelligence is a multifaceted ability encompassing
# reasoning, learning, problem-solving, abstraction, creativity, and adaptation,
# allowing organisms or systems to process information, recognize patterns, form
# connections, and navigate complex environments efficiently."
print("\nTest: Female Voice (17 Seconds)")
audio_path = "keras_hub/src/tests/test_data/female_voice_clip_17sec.wav"
audio, sr = librosa.load(audio_path, sr=cfg["sampling_rate"])
audio = audio.reshape(1, -1)
generated_ids = keras_model.generate(audio)
transcription = keras_model.tokenizer.detokenize(generated_ids[0])
print("Generated Token IDs:", generated_ids)
print("Transcription:", transcription)
