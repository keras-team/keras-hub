"""Convert BAAI/bge-*-en-v1.5 checkpoints to KerasHub format.

Usage:
    python tools/checkpoint_conversion/convert_bge_checkpoints.py \
        --preset bge_small_en_v1.5

    # To upload after conversion:
    python tools/checkpoint_conversion/convert_bge_checkpoints.py \
        --preset bge_small_en_v1.5 \
        --upload_uri kaggle://keras/bge/keras/bge_small_en_v1.5
"""

import json
import os

import keras
import numpy as np
import requests
import torch
import torch.nn.functional as F
import transformers
from absl import app
from absl import flags

import keras_hub
from tools.checkpoint_conversion.checkpoint_conversion_utils import (
    get_md5_checksum,
)

# Maps KerasHub preset name → HuggingFace model ID.
PRESET_MAP = {
    "bge_small_en_v1.5": "BAAI/bge-small-en-v1.5",
    "bge_base_en_v1.5": "BAAI/bge-base-en-v1.5",
    "bge_large_en_v1.5": "BAAI/bge-large-en-v1.5",
}

EXTRACT_DIR = "./{}"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {', '.join(PRESET_MAP.keys())}",
)
flags.DEFINE_string(
    "upload_uri",
    None,
    "Optional URI to upload the preset (e.g. kaggle://keras/bge/keras/bge_small_en_v1.5).",
)


def download_files(hf_model_name):
    print("-> Download original vocab and config.")
    extract_dir = EXTRACT_DIR.format(FLAGS.preset)
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Config.
    config_path = os.path.join(extract_dir, "config.json")
    response = requests.get(
        f"https://huggingface.co/{hf_model_name}/raw/main/config.json"
    )
    open(config_path, "wb").write(response.content)
    print(f"  `{config_path}`")

    # Vocab.
    vocab_path = os.path.join(extract_dir, "vocab.txt")
    response = requests.get(
        f"https://huggingface.co/{hf_model_name}/raw/main/vocab.txt"
    )
    open(vocab_path, "wb").write(response.content)
    print(f"  `{vocab_path}`")


def define_tokenizer(hf_model_name):
    print("\n-> Define tokenizers.")
    extract_dir = EXTRACT_DIR.format(FLAGS.preset)
    vocab_path = os.path.join(extract_dir, "vocab.txt")

    keras_hub_tokenizer = keras_hub.models.BgeTokenizer(
        vocabulary=vocab_path,
        lowercase=True,
    )
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    print("\n-> MD5 checksum of the vocab file:")
    print(f"  `{vocab_path}` md5sum: {get_md5_checksum(vocab_path)}")

    return keras_hub_tokenizer, hf_tokenizer


def convert_checkpoints(keras_hub_model, hf_model):
    """Assign HuggingFace weights to the KerasHub BgeBackbone."""
    print("\n-> Convert original weights to KerasHub format.")

    extract_dir = EXTRACT_DIR.format(FLAGS.preset)
    config_path = os.path.join(extract_dir, "config.json")

    # Build backbone config from HF config.json.
    with open(config_path, "r") as f:
        pt_cfg = json.load(f)

    cfg = {
        "vocabulary_size": pt_cfg["vocab_size"],
        "num_layers": pt_cfg["num_hidden_layers"],
        "num_heads": pt_cfg["num_attention_heads"],
        "hidden_dim": pt_cfg["hidden_size"],
        "intermediate_dim": pt_cfg["intermediate_size"],
        "dropout": pt_cfg.get("hidden_dropout_prob", 0.1),
        "max_sequence_length": pt_cfg["max_position_embeddings"],
        "num_segments": pt_cfg["type_vocab_size"],
    }
    print("Backbone config:", cfg)

    hf_wts = hf_model.state_dict()
    print("\nHuggingFace weight keys:")
    for k in hf_wts.keys():
        print(f"  {k}  {tuple(hf_wts[k].shape)}")

    # ── Embeddings ──────────────────────────────────────────────────────────
    keras_hub_model.get_layer("token_embedding").embeddings.assign(
        hf_wts["embeddings.word_embeddings.weight"].numpy()
    )
    keras_hub_model.get_layer("position_embedding").position_embeddings.assign(
        hf_wts["embeddings.position_embeddings.weight"].numpy()
    )
    keras_hub_model.get_layer("segment_embedding").embeddings.assign(
        hf_wts["embeddings.token_type_embeddings.weight"].numpy()
    )
    keras_hub_model.get_layer("embeddings_layer_norm").gamma.assign(
        hf_wts["embeddings.LayerNorm.weight"].numpy()
    )
    keras_hub_model.get_layer("embeddings_layer_norm").beta.assign(
        hf_wts["embeddings.LayerNorm.bias"].numpy()
    )

    # ── Transformer layers ───────────────────────────────────────────────────
    for i in range(cfg["num_layers"]):
        num_heads = cfg["num_heads"]
        hidden_dim = cfg["hidden_dim"]
        head_dim = hidden_dim // num_heads

        # --- Self-attention: query ---
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.attention.self.query.weight"]
            .numpy()
            .T.reshape(hidden_dim, num_heads, head_dim)
        )
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.attention.self.query.bias"]
            .numpy()
            .reshape(num_heads, head_dim)
        )

        # --- Self-attention: key ---
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.attention.self.key.weight"]
            .numpy()
            .T.reshape(hidden_dim, num_heads, head_dim)
        )
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.attention.self.key.bias"]
            .numpy()
            .reshape(num_heads, head_dim)
        )

        # --- Self-attention: value ---
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.attention.self.value.weight"]
            .numpy()
            .T.reshape(hidden_dim, num_heads, head_dim)
        )
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.attention.self.value.bias"]
            .numpy()
            .reshape(num_heads, head_dim)
        )

        # --- Self-attention: output projection ---
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.attention.output.dense.weight"]
            .numpy()
            .T.reshape(num_heads, head_dim, hidden_dim)
        )
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.attention.output.dense.bias"].numpy()
        )

        # --- Self-attention layer norm ---
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer_norm.gamma.assign(
            hf_wts[
                f"encoder.layer.{i}.attention.output.LayerNorm.weight"
            ].numpy()
        )
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer_norm.beta.assign(
            hf_wts[f"encoder.layer.{i}.attention.output.LayerNorm.bias"].numpy()
        )

        # --- FFN: intermediate (up-projection) ---
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.intermediate.dense.weight"].numpy().T
        )
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.intermediate.dense.bias"].numpy()
        )

        # --- FFN: output (down-projection) ---
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.output.dense.weight"].numpy().T
        )
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.output.dense.bias"].numpy()
        )

        # --- FFN layer norm ---
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layer_norm.gamma.assign(
            hf_wts[f"encoder.layer.{i}.output.LayerNorm.weight"].numpy()
        )
        keras_hub_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layer_norm.beta.assign(
            hf_wts[f"encoder.layer.{i}.output.LayerNorm.bias"].numpy()
        )

    # ── Pooler ───────────────────────────────────────────────────────────────
    # NOTE: Pooler weights are converted for completeness.  For BGE-style
    # sentence embeddings, use sequence_output[:, 0, :] + L2 normalization
    # rather than pooled_output (which is Tanh-activated).
    keras_hub_model.get_layer("pooled_dense").kernel.assign(
        hf_wts["pooler.dense.weight"].numpy().T
    )
    keras_hub_model.get_layer("pooled_dense").bias.assign(
        hf_wts["pooler.dense.bias"].numpy()
    )


def validate_output(
    keras_hub_model,
    hf_model,
    keras_hub_tokenizer,
    hf_tokenizer,
):
    """Cross-framework numerical validation (atol=1e-4)."""

    print("\n-> Validate outputs numerically.")
    test_sentence = "I love machine learning and nlp"

    # HuggingFace reference.
    hf_inputs = hf_tokenizer(
        [test_sentence],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
    hf_cls = hf_outputs.last_hidden_state[:, 0, :].numpy()
    hf_emb = F.normalize(torch.tensor(hf_cls), p=2, dim=1).numpy()

    # KerasHub.
    preprocessor = keras_hub.models.BgeTextEmbedderPreprocessor(
        keras_hub_tokenizer, sequence_length=512
    )
    kh_inputs = preprocessor([test_sentence])
    kh_outputs = keras_hub_model(kh_inputs)
    kh_cls = kh_outputs["sequence_output"][:, 0, :].numpy()
    kh_emb = keras.ops.normalize(
        keras.ops.convert_to_tensor(kh_cls), axis=-1
    ).numpy()

    print(f"  HF embedding (first 5 dims):  {hf_emb[0, :5]}")
    print(f"  KerasHub embedding (first 5):  {kh_emb[0, :5]}")

    if np.allclose(hf_emb, kh_emb, atol=1e-4):
        print("  ✓ Outputs match within atol=1e-4")
    else:
        max_diff = np.abs(hf_emb - kh_emb).max()
        print(f"  ✗ Max absolute difference: {max_diff:.6f}")
        print(
            "  WARNING: Outputs differ beyond tolerance. "
            "Inspect weight assignment and activation function."
        )


def main(_):
    if FLAGS.preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{FLAGS.preset}'. "
            f"Choose from: {list(PRESET_MAP.keys())}"
        )

    hf_model_name = PRESET_MAP[FLAGS.preset]
    print(f"Converting preset: {FLAGS.preset}  (HF: {hf_model_name})")

    # 1. Download vocab and config.
    download_files(hf_model_name)

    # 2. Load HF model.
    print("\n-> Load HuggingFace model.")
    hf_model = transformers.AutoModel.from_pretrained(hf_model_name)
    hf_model.eval()

    # 3. Define KerasHub tokenizer.
    keras_hub_tokenizer, hf_tokenizer = define_tokenizer(hf_model_name)

    # 4. Build KerasHub backbone from HF config.
    extract_dir = EXTRACT_DIR.format(FLAGS.preset)
    with open(os.path.join(extract_dir, "config.json"), "r") as f:
        pt_cfg = json.load(f)

    keras_hub_backbone = keras_hub.models.BgeBackbone(
        vocabulary_size=pt_cfg["vocab_size"],
        num_layers=pt_cfg["num_hidden_layers"],
        num_heads=pt_cfg["num_attention_heads"],
        hidden_dim=pt_cfg["hidden_size"],
        intermediate_dim=pt_cfg["intermediate_size"],
        dropout=pt_cfg.get("hidden_dropout_prob", 0.1),
        max_sequence_length=pt_cfg["max_position_embeddings"],
        num_segments=pt_cfg["type_vocab_size"],
    )

    # Run a dummy forward pass to build all weights.
    dummy = {
        "token_ids": keras.ops.ones((1, 8), dtype="int32"),
        "segment_ids": keras.ops.zeros((1, 8), dtype="int32"),
        "padding_mask": keras.ops.ones((1, 8), dtype="int32"),
    }
    keras_hub_backbone(dummy)

    print(
        f"\n  KerasHub parameter count: {keras_hub_backbone.count_params():,}"
    )

    # 5. Convert weights.
    convert_checkpoints(keras_hub_backbone, hf_model)

    # 6. Validate.
    validate_output(
        keras_hub_backbone, hf_model, keras_hub_tokenizer, hf_tokenizer
    )

    # 7. Save preset.
    print("\n-> Save KerasHub preset.")
    preset_path = os.path.join(extract_dir, "preset")
    keras_hub_backbone.save_to_preset(preset_path)
    keras_hub_tokenizer.save_to_preset(preset_path)
    print(f"  Preset saved to: `{preset_path}`")

    # 8. Upload if requested.
    if FLAGS.upload_uri:
        print(f"\n-> Upload preset to: {FLAGS.upload_uri}")
        keras_hub.upload_preset(FLAGS.upload_uri, preset_path)
        print("  Upload complete.")


if __name__ == "__main__":
    app.run(main)
