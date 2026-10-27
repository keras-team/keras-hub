"""
Convert HuggingFace sentence-transformer BERT checkpoints to KerasHub format.

This script loads weights from HuggingFace sentence-transformer models
(e.g., all-MiniLM-L6-v2) and converts them to KerasHub's BertTextEmbedder
format with numerical parity verification.

Setup:
```shell
pip install keras-hub keras sentence-transformers safetensors huggingface_hub
```

Usage:
```shell
cd tools/checkpoint_conversion
python convert_bert_sentence_transformer_checkpoints.py \
    --preset all_minilm_l6_v2_en
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
from sentence_transformers import SentenceTransformer

import keras_hub
from keras_hub.src.models.bert.bert_text_embedder import BertTextEmbedder
from keras_hub.src.utils.transformers.convert_bert import (
    convert_backbone_config,
)
from keras_hub.src.utils.transformers.convert_bert import convert_tokenizer
from keras_hub.src.utils.transformers.convert_bert import convert_weights
from keras_hub.src.utils.transformers.convert_bert import (
    load_preprocessor_config,
)
from keras_hub.src.utils.transformers.convert_bert import load_task_config
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "preset",
    None,
    "Preset name for output. Must be one of the keys in PRESET_MAP.",
)

# Standard BERT presets (backbone-only, no embedder head).
BERT_PRESET_MAP = {
    "bert_tiny_en_uncased": "google/bert_uncased_L-2_H-128_A-2",
    "bert_small_en_uncased": "google/bert_uncased_L-4_H-512_A-8",
    "bert_medium_en_uncased": "google/bert_uncased_L-8_H-512_A-8",
    "bert_base_en_uncased": "google-bert/bert-base-uncased",
    "bert_base_en": "google-bert/bert-base-cased",
    "bert_base_zh": "google-bert/bert-base-chinese",
    "bert_base_multi": "google-bert/bert-base-multilingual-cased",
    "bert_large_en_uncased": "google-bert/bert-large-uncased",
    "bert_large_en": "google-bert/bert-large-cased",
}

# Sentence-transformer presets (full BertTextEmbedder with pooling head).
SENTENCE_TRANSFORMER_PRESET_MAP = {
    # "all-*" family: general-purpose sentence embedding models.
    "all_minilm_l6_v2_en": "sentence-transformers/all-MiniLM-L6-v2",
    "all_minilm_l6_v1_en": "sentence-transformers/all-MiniLM-L6-v1",
    "all_minilm_l12_v2_en": "sentence-transformers/all-MiniLM-L12-v2",
    # "paraphrase-*" family: optimized for paraphrase detection.
    "paraphrase_minilm_l3_v2_en": (
        "sentence-transformers/paraphrase-MiniLM-L3-v2"
    ),
    "paraphrase_minilm_l6_v2_en": (
        "sentence-transformers/paraphrase-MiniLM-L6-v2"
    ),
    "paraphrase_minilm_l12_v2_en": (
        "sentence-transformers/paraphrase-MiniLM-L12-v2"
    ),
    # "multi-qa-*" family: optimized for semantic search.
    "multi_qa_minilm_l6_cos_v1_en": (
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    ),
    "multi_qa_minilm_l6_dot_v1_en": (
        "sentence-transformers/multi-qa-MiniLM-L6-dot-v1"
    ),
    # "msmarco-*" family: optimized for information retrieval.
    "msmarco_minilm_l6_cos_v5_en": (
        "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
    ),
    "msmarco_minilm_l12_cos_v5_en": (
        "sentence-transformers/msmarco-MiniLM-L12-cos-v5"
    ),
}

# Combined map for --preset lookup.
PRESET_MAP = {**BERT_PRESET_MAP, **SENTENCE_TRANSFORMER_PRESET_MAP}


def validate_output(keras_model, hf_model_id):
    """
    Validate numerical parity between KerasHub and HF SentenceTransformer.

    Performs four checks:
    1. Parameter count verification.
    2. Embedding output comparison (max absolute difference).
    3. Cosine similarity ranking consistency.
    4. Semantic search parity (encode_text / encode_documents / similarity).

    Args:
        keras_model: The converted KerasHub BertTextEmbedder.
        hf_model_id: The HuggingFace model ID to compare against.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print("NUMERICAL VERIFICATION")
    print("=" * 60)

    # Load HuggingFace model.
    print(f"\nLoading HuggingFace model: {hf_model_id}")
    hf_model = SentenceTransformer(hf_model_id)
    hf_model.eval()

    # =========================================
    # PARAMETER COUNT VERIFICATION
    # =========================================
    print("\n--- Parameter Count Check ---")
    keras_params = keras_model.count_params()

    hf_modules = list(hf_model._modules.values())
    hf_transformer = hf_modules[0]
    hf_params = sum(p.numel() for p in hf_transformer.auto_model.parameters())

    print(f"KerasHub params:      {keras_params:,}")
    print(f"HuggingFace params:   {hf_params:,}")

    param_diff = abs(keras_params - hf_params)
    if param_diff == 0:
        print("✅ Parameter count EXACT MATCH!")
    else:
        print(f"❌ Parameter count mismatch! Diff: {param_diff:,}")

    # =========================================
    # EMBEDDING VERIFICATION
    # =========================================
    print("\n--- Embedding Verification ---")
    test_texts = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    print(f"Test inputs: {test_texts}")

    # HuggingFace embeddings.
    print("\nComputing HF embeddings...")
    with torch.no_grad():
        hf_embeddings = hf_model.encode(test_texts, convert_to_numpy=True)
    print(f"HF output shape: {hf_embeddings.shape}")
    print(f"HF embedding[0][:5]: {hf_embeddings[0][:5]}")

    # KerasHub embeddings.
    print("\nComputing KerasHub embeddings...")
    keras_embeddings = keras_model.predict(test_texts)
    keras_embeddings = np.array(keras_embeddings)
    print(f"KerasHub output shape: {keras_embeddings.shape}")
    print(f"KerasHub embedding[0][:5]: {keras_embeddings[0][:5]}")

    # Compare.
    print("\n--- Comparison ---")
    max_diff = np.max(np.abs(hf_embeddings - keras_embeddings))
    mean_diff = np.mean(np.abs(hf_embeddings - keras_embeddings))
    print(f"Max absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    # Norm check (should be ~1.0 for L2 normalized).
    keras_norms = np.linalg.norm(keras_embeddings, axis=1)
    hf_norms = np.linalg.norm(hf_embeddings, axis=1)
    print(f"KerasHub norms: {keras_norms}")
    print(f"HF norms:       {hf_norms}")

    # Cosine similarity ranking check.
    keras_sims = keras_embeddings @ keras_embeddings.T
    hf_sims = hf_embeddings @ hf_embeddings.T
    # Sentences 0 and 1 should be more similar to each other than to 2.
    keras_ranking_ok = keras_sims[0, 1] > keras_sims[0, 2]
    hf_ranking_ok = hf_sims[0, 1] > hf_sims[0, 2]
    print(
        f"\nRanking consistency: KerasHub={keras_ranking_ok}, "
        f"HF={hf_ranking_ok}"
    )

    # =========================================
    # SEMANTIC SEARCH VERIFICATION
    # =========================================
    print("\n--- Semantic Search (encode_text / encode_documents) ---")
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin.",
        "Mars is often referred to as the Red Planet.",
        "Jupiter has a prominent red spot.",
    ]
    print(f"Query: {query}")
    print(f"Documents: {documents}")

    # KerasHub: use convenience methods.
    keras_q = keras_model.encode_text(query)
    keras_d = keras_model.encode_documents(documents)
    keras_search_sims = keras_model.similarity(keras_q, keras_d)
    keras_best = int(np.argmax(keras_search_sims))

    # HuggingFace: equivalent flow.
    with torch.no_grad():
        hf_q = hf_model.encode([query], convert_to_numpy=True)
        hf_d = hf_model.encode(documents, convert_to_numpy=True)
    hf_search_sims = hf_q @ hf_d.T
    hf_best = int(np.argmax(hf_search_sims))

    print(
        f"\nKerasHub sims: {keras_search_sims[0]} -> "
        f"Best: {documents[keras_best]}"
    )
    print(f"HF sims:       {hf_search_sims[0]} -> Best: {documents[hf_best]}")
    search_ranking_ok = keras_best == hf_best
    print(f"Search ranking match: {search_ranking_ok}")

    # Final result.
    print("\n--- Result ---")
    passed = True

    if param_diff != 0:
        print("❌ FAILED: Parameter count mismatch")
        passed = False

    if mean_diff > 5e-4:
        print(f"❌ FAILED: Mean diff {mean_diff:.2e} > 5e-4")
        passed = False
    elif mean_diff > 1e-4:
        print(
            f"⚠️  WARN: Mean diff {mean_diff:.2e} > 1e-4 "
            "(acceptable FP32 variance)"
        )

    if not keras_ranking_ok:
        print("❌ FAILED: Ranking inconsistency")
        passed = False

    if not search_ranking_ok:
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
    is_sentence_transformer = preset in SENTENCE_TRANSFORMER_PRESET_MAP

    print(f"\n{'=' * 60}")
    print(f"Converting: {hf_model_id} -> {preset}")
    st_type = (
        "sentence-transformer" if is_sentence_transformer else "standard BERT"
    )
    print(f"Type: {st_type}")
    print(f"{'=' * 60}\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Download and parse config.
        print("Downloading config...")
        config_path = hf_hub_download(
            hf_model_id, "config.json", local_dir=temp_dir
        )
        with open(config_path, "r") as f:
            transformers_config = json.load(f)

        # Create KerasHub backbone.
        keras_config = convert_backbone_config(transformers_config)
        print(f"\nBackbone config: {keras_config}")
        backbone = keras_hub.models.BertBackbone(
            **keras_config, dtype="float32"
        )
        print(f"Backbone parameters: {backbone.count_params():,}")

        # Download and convert weights.
        print("\nDownloading weights...")
        hf_hub_download(hf_model_id, "model.safetensors", local_dir=temp_dir)

        print("Converting weights...")
        with SafetensorLoader(temp_dir) as loader:
            convert_weights(backbone, loader, transformers_config)

        # Convert tokenizer.
        print("\nConverting tokenizer...")
        hf_hub_download(hf_model_id, "vocab.txt", local_dir=temp_dir)
        hf_hub_download(
            hf_model_id, "tokenizer_config.json", local_dir=temp_dir
        )
        tokenizer = convert_tokenizer(keras_hub.models.BertTokenizer, temp_dir)

        if is_sentence_transformer:
            # Download sentence-transformer config files.
            hf_hub_download(hf_model_id, "modules.json", local_dir=temp_dir)
            hf_hub_download(
                hf_model_id, "1_Pooling/config.json", local_dir=temp_dir
            )
            hf_hub_download(
                hf_model_id,
                "sentence_bert_config.json",
                local_dir=temp_dir,
            )
            task_config = load_task_config(temp_dir, transformers_config)
            preprocessor_config = load_preprocessor_config(
                temp_dir, transformers_config
            )
            print(f"Task config: {task_config}")
            print(f"Preprocessor config: {preprocessor_config}")

            # Build full BertTextEmbedder with preprocessor.
            preprocessor = keras_hub.models.BertTextEmbedderPreprocessor(
                tokenizer=tokenizer,
                **preprocessor_config,
            )
            embedder = BertTextEmbedder(
                backbone=backbone,
                preprocessor=preprocessor,
                **task_config,
            )

            # Validate with sentence-transformers reference.
            passed = validate_output(embedder, hf_model_id)
            if not passed:
                print("\n⚠️  Verification failed. Check weight mapping.")
                return

            # Save full embedder preset.
            print(f"\nSaving to preset: ./{preset}")
            embedder.save_to_preset(preset)
        else:
            # Standard BERT: save backbone + tokenizer only.
            print(f"\nSaving backbone to preset: ./{preset}")
            backbone.save_to_preset(preset)
            tokenizer.save_to_preset(preset)

        print(f"\n✅ Successfully converted and saved to: ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
