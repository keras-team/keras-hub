"""
Convert HuggingFace BAAI/bge-*-en-v1.5 checkpoints to KerasHub format.

BGE (BAAI General Embedding) models are standard BERT encoders fine-tuned for
dense retrieval. Embeddings are computed as the CLS token output followed by
L2 normalization. This maps exactly to:
    BertTextEmbedder(
        backbone=backbone,
        preprocessor=preprocessor,
        pooling_mode="cls",
        normalize=True,
    )

Setup:
```shell
pip install keras-hub keras transformers safetensors huggingface_hub torch
```

Usage:
```shell
cd tools/checkpoint_conversion
python convert_bge_checkpoints.py --preset bge_small_en_v1.5
python convert_bge_checkpoints.py \
    --preset bge_small_en_v1.5 \
    --upload_uri kaggle://keras/bge/keras/bge_small_en_v1.5
```
"""

import json
import os
import tempfile

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from transformers import AutoModel
from transformers import AutoTokenizer

import keras_hub
from keras_hub.src.utils.transformers.convert_bert import (
    convert_backbone_config,
)
from keras_hub.src.utils.transformers.convert_bert import convert_tokenizer
from keras_hub.src.utils.transformers.convert_bert import convert_weights
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "preset",
    None,
    "Preset name to convert. Must be one of the keys in PRESET_MAP.",
)
flags.DEFINE_string(
    "upload_uri",
    None,
    "Kaggle URI to upload the preset to, e.g. "
    '"kaggle://keras/bge/keras/{preset}". Optional.',
)

PRESET_MAP = {
    "bge_small_en_v1.5": "BAAI/bge-small-en-v1.5",
    "bge_base_en_v1.5": "BAAI/bge-base-en-v1.5",
    "bge_large_en_v1.5": "BAAI/bge-large-en-v1.5",
}

# BGE standardizes on sequence_length=512.
SEQUENCE_LENGTH = 512


def _hf_encode(hf_model, hf_tokenizer, texts):
    """
    Encode texts with a HuggingFace BERT model using CLS + L2 normalization,
    matching BGE's reference implementation exactly.
    """
    encoded = hf_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=SEQUENCE_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = hf_model(**encoded)
    # CLS token (index 0), then L2 normalize.
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
    return cls_embeddings / np.maximum(norms, 1e-12)


def validate_output(keras_embedder, hf_model_id):
    """
    Validate numerical parity between the converted KerasHub model and the
    HuggingFace reference implementation.

    Performs five checks:
    1. Parameter count verification.
    2. Embedding output comparison (max/mean absolute difference).
    3. L2 norm check (embeddings must be unit-length).
    4. Cosine similarity ranking consistency (intra-batch).
    5. Semantic search ranking consistency.

    Args:
        keras_embedder: Converted KerasHub BertTextEmbedder.
        hf_model_id: HuggingFace model ID to compare against.

    Returns:
        bool: True if all checks pass.
    """
    print("\n" + "=" * 60)
    print("NUMERICAL VERIFICATION")
    print("=" * 60)

    print(f"\nLoading HuggingFace model: {hf_model_id}")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    hf_model = AutoModel.from_pretrained(hf_model_id)
    hf_model.eval()

    # =========================================
    # PARAMETER COUNT
    # =========================================
    print("\n--- Parameter Count ---")
    keras_params = keras_embedder.backbone.count_params()
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"KerasHub params:    {keras_params:,}")
    print(f"HuggingFace params: {hf_params:,}")
    param_diff = abs(keras_params - hf_params)
    if param_diff == 0:
        print("✅ Exact match")
    else:
        print(f"❌ Mismatch — diff: {param_diff:,}")

    # =========================================
    # EMBEDDING PARITY
    # =========================================
    print("\n--- Embedding Parity ---")
    test_texts = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    print(f"Test inputs: {test_texts}")

    print("\nComputing HuggingFace embeddings (CLS + L2 norm)...")
    hf_embeddings = _hf_encode(hf_model, hf_tokenizer, test_texts)
    print(f"HF shape: {hf_embeddings.shape}")
    print(f"HF[0][:5]: {hf_embeddings[0][:5]}")

    print("\nComputing KerasHub embeddings...")
    keras_embeddings = np.array(keras_embedder.predict(test_texts))
    print(f"KerasHub shape: {keras_embeddings.shape}")
    print(f"KerasHub[0][:5]: {keras_embeddings[0][:5]}")

    max_diff = np.max(np.abs(hf_embeddings - keras_embeddings))
    mean_diff = np.mean(np.abs(hf_embeddings - keras_embeddings))
    print(f"\nMax absolute diff:  {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    # Cosine similarity ranking check.
    keras_batch_sims = keras_embeddings @ keras_embeddings.T
    hf_batch_sims = hf_embeddings @ hf_embeddings.T
    # Sentences 0 and 1 should be more similar to each other than to 2.
    keras_ranking_ok = keras_batch_sims[0, 1] > keras_batch_sims[0, 2]
    hf_ranking_ok = hf_batch_sims[0, 1] > hf_batch_sims[0, 2]
    print(
        f"\nRanking consistency: KerasHub={keras_ranking_ok}, "
        f"HF={hf_ranking_ok}"
    )

    # =========================================
    # L2 NORM CHECK
    # =========================================
    print("\n--- L2 Norm Check (should be ~1.0) ---")
    keras_norms = np.linalg.norm(keras_embeddings, axis=1)
    hf_norms = np.linalg.norm(hf_embeddings, axis=1)
    print(f"KerasHub norms: {keras_norms}")
    print(f"HF norms:       {hf_norms}")
    norms_ok = np.allclose(keras_norms, 1.0, atol=1e-5)

    # =========================================
    # SEMANTIC SEARCH RANKING
    # =========================================
    print("\n--- Semantic Search Ranking ---")
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin.",
        "Mars is often referred to as the Red Planet.",
        "Jupiter has a prominent red spot.",
    ]
    print(f"Query: {query}")
    print(f"Documents: {documents}")

    keras_q = keras_embedder.encode_text([query])
    keras_d = keras_embedder.encode_documents(documents)
    keras_sims = keras_embedder.similarity(keras_q, keras_d)
    keras_best = int(np.argmax(keras_sims))

    hf_q = _hf_encode(hf_model, hf_tokenizer, [query])
    hf_d = _hf_encode(hf_model, hf_tokenizer, documents)
    hf_sims = hf_q @ hf_d.T
    hf_best = int(np.argmax(hf_sims))

    print(f"\nKerasHub sims: {keras_sims} -> Best: {documents[keras_best]}")
    print(f"HF sims:       {hf_sims[0]} -> Best: {documents[hf_best]}")
    search_ok = keras_best == hf_best

    # =========================================
    # RESULT
    # =========================================
    print("\n--- Result ---")
    passed = True

    if param_diff != 0:
        print("❌ FAILED: Parameter count mismatch")
        passed = False

    if mean_diff > 5e-4:
        print(f"❌ FAILED: Mean diff {mean_diff:.2e} exceeds 5e-4")
        passed = False
    elif mean_diff > 1e-4:
        print(f"⚠️  WARN: Mean diff {mean_diff:.2e} > 1e-4 (FP32 variance)")

    if max_diff > 1e-3:
        print(f"❌ FAILED: Max diff {max_diff:.2e} exceeds 1e-3")
        passed = False

    if not norms_ok:
        print("❌ FAILED: Embeddings are not unit-length")
        passed = False

    if not keras_ranking_ok:
        print("❌ FAILED: Ranking inconsistency")
        passed = False

    if not search_ok:
        print("❌ FAILED: Semantic search ranking mismatch")
        passed = False

    if passed:
        print("✅ ALL CHECKS PASSED")

    return passed


def main(_):
    """
    Main entry point: convert, validate, and save preset.
    """
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. "
            f"Must be one of: {list(PRESET_MAP.keys())}"
        )

    hf_model_id = PRESET_MAP[preset]

    print(f"\n{'=' * 60}")
    print(f"Converting: {hf_model_id} -> {preset}")
    print("Pooling: CLS token + L2 normalization")
    print(f"{'=' * 60}\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Download and parse config.
        print("Downloading config.json...")
        config_path = hf_hub_download(
            hf_model_id, "config.json", local_dir=temp_dir
        )
        with open(config_path, "r") as f:
            transformers_config = json.load(f)

        # Build KerasHub backbone from HF config.
        keras_config = convert_backbone_config(transformers_config)
        print(f"\nBackbone config: {keras_config}")
        backbone = keras_hub.models.BertBackbone(
            **keras_config, dtype="float32"
        )
        print(f"Backbone parameters: {backbone.count_params():,}")

        # Download and load weights.
        print("\nDownloading model.safetensors...")
        hf_hub_download(hf_model_id, "model.safetensors", local_dir=temp_dir)
        print("Converting weights...")
        with SafetensorLoader(temp_dir) as loader:
            convert_weights(backbone, loader, transformers_config)

        # Build tokenizer.
        print("\nDownloading tokenizer files...")
        hf_hub_download(hf_model_id, "vocab.txt", local_dir=temp_dir)
        hf_hub_download(
            hf_model_id, "tokenizer_config.json", local_dir=temp_dir
        )
        tokenizer = convert_tokenizer(keras_hub.models.BertTokenizer, temp_dir)

        # Assemble BertTextEmbedder with BGE settings.
        preprocessor = keras_hub.models.BertTextEmbedderPreprocessor(
            tokenizer=tokenizer,
            sequence_length=SEQUENCE_LENGTH,
        )
        embedder = keras_hub.models.BertTextEmbedder(
            backbone=backbone,
            preprocessor=preprocessor,
            pooling_mode="cls",
            normalize=True,
        )

        # Validate.
        passed = validate_output(embedder, hf_model_id)
        if not passed:
            print("\n⚠️  Verification failed. Preset not saved.")
            return

        # Save.
        print(f"\nSaving to preset: ./{preset}")
        embedder.save_to_preset(preset)
        print(f"\n✅ Successfully converted and saved to: ./{preset}")

        upload_uri = FLAGS.upload_uri
        if upload_uri:
            keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
            print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
