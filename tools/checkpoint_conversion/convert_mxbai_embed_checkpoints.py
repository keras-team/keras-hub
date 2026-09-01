"""
Convert HuggingFace mixedbread-ai mxbai-embed checkpoints to KerasHub format.

Supports three BERT-based presets (BertTextEmbedder):
- mxbai-embed-large-v1   (CLS pooling, no normalization)
- mxbai-embed-2d-large-v1 (CLS pooling, no normalization)
- mxbai-embed-xsmall-v1  (mean pooling, no normalization)

Setup:
```shell
pip install keras-hub keras sentence-transformers safetensors huggingface_hub
```

Usage:
```shell
cd tools/checkpoint_conversion
python convert_mxbai_embed_checkpoints.py \
    --preset mxbai_embed_large_v1_en
```
"""

import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
from absl import app
from absl import flags
from keras import ops
from sentence_transformers import SentenceTransformer

from keras_hub.src.models.bert.bert_text_embedder import BertTextEmbedder

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "preset",
    None,
    "Preset name for output. Must be one of the keys in PRESET_MAP.",
)

# BERT-based presets.
PRESET_MAP = {
    "mxbai_embed_large_v1_en": "mixedbread-ai/mxbai-embed-large-v1",
    "mxbai_embed_2d_large_v1_en": "mixedbread-ai/mxbai-embed-2d-large-v1",
    "mxbai_embed_xsmall_v1_en": "mixedbread-ai/mxbai-embed-xsmall-v1",
}


def validate_output(keras_model, hf_model_id):
    """Print embedding diagnostics for manual review.

    Prints parameter count, embedding statistics (mean logits),
    cosine similarity rankings, and semantic search results.
    Does NOT gate on pass/fail — always returns.

    Args:
        keras_model: The converted KerasHub text embedder.
        hf_model_id: The HuggingFace model ID to compare against.
    """
    print("\n" + "=" * 60)
    print("NUMERICAL VERIFICATION")
    print("=" * 60)

    # Load HuggingFace model for comparison.
    print(f"\nLoading HuggingFace model: {hf_model_id}")
    hf_model = SentenceTransformer(hf_model_id)
    hf_model.eval()

    # =========================================
    # PARAMETER COUNT
    # =========================================
    print("\n--- Parameter Count ---")
    keras_params = keras_model.count_params()
    hf_modules = list(hf_model._modules.values())
    hf_transformer = hf_modules[0]
    hf_params = sum(p.numel() for p in hf_transformer.auto_model.parameters())

    print(f"KerasHub params:    {keras_params:,}")
    print(f"HuggingFace params: {hf_params:,}")
    param_diff = abs(keras_params - hf_params)
    if param_diff == 0:
        print("✅ Parameter count EXACT MATCH")
    else:
        print(f"⚠️  Parameter count diff: {param_diff:,}")

    # =========================================
    # EMBEDDING COMPARISON
    # =========================================
    print("\n--- Embedding Comparison ---")
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
    print(f"HF embedding mean:   {np.mean(hf_embeddings):.6e}")

    # KerasHub embeddings.
    print("\nComputing KerasHub embeddings...")
    keras_embeddings = ops.convert_to_numpy(keras_model.predict(test_texts))
    print(f"KerasHub output shape: {keras_embeddings.shape}")
    print(f"KerasHub embedding[0][:5]: {keras_embeddings[0][:5]}")
    print(f"KerasHub embedding mean:   {np.mean(keras_embeddings):.6e}")

    # Differences.
    print("\n--- Mean Logits Comparison ---")
    max_diff = np.max(np.abs(hf_embeddings - keras_embeddings))
    mean_diff = np.mean(np.abs(hf_embeddings - keras_embeddings))
    print(f"Max absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    # Norm check.
    keras_norms = np.linalg.norm(keras_embeddings, axis=1)
    hf_norms = np.linalg.norm(hf_embeddings, axis=1)
    print(f"KerasHub norms: {keras_norms}")
    print(f"HF norms:       {hf_norms}")

    # =========================================
    # COSINE SIMILARITY RANKING
    # =========================================
    print("\n--- Cosine Similarity Matrix ---")
    keras_sims = keras_embeddings @ keras_embeddings.T
    hf_sims = hf_embeddings @ hf_embeddings.T
    print(
        f"KerasHub sim[0,1]={keras_sims[0, 1]:.4f}  "
        f"sim[0,2]={keras_sims[0, 2]:.4f}"
    )
    print(
        f"HF       sim[0,1]={hf_sims[0, 1]:.4f}  sim[0,2]={hf_sims[0, 2]:.4f}"
    )

    keras_ranking_ok = keras_sims[0, 1] > keras_sims[0, 2]
    hf_ranking_ok = hf_sims[0, 1] > hf_sims[0, 2]
    print(
        f"Ranking consistency: KerasHub={keras_ranking_ok}, HF={hf_ranking_ok}"
    )

    # =========================================
    # SEMANTIC SEARCH
    # =========================================
    print("\n--- Semantic Search ---")
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin.",
        "Mars is often referred to as the Red Planet.",
        "Jupiter has a prominent red spot.",
    ]
    print(f"Query: {query}")
    print(f"Documents: {documents}")

    # KerasHub search.
    keras_q = keras_model.encode_text(query)
    keras_d = keras_model.encode_documents(documents)
    keras_search_sims = ops.convert_to_numpy(
        keras_model.similarity(keras_q, keras_d)
    )
    keras_best = int(np.argmax(keras_search_sims))

    # HuggingFace search.
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
    search_match = keras_best == hf_best
    print(f"Search ranking match: {search_match}")

    print("\n" + "=" * 60)
    print("NUMERICAL VERIFICATION COMPLETE")
    print("=" * 60)


def main(_):
    """Main entry point: convert, validate, and save preset."""
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. "
            f"Must be one of: {list(PRESET_MAP.keys())}"
        )

    hf_model_id = PRESET_MAP[preset]

    print(f"\n{'=' * 60}")
    print(f"Converting: {hf_model_id} -> {preset}")
    print("Architecture: BERT")
    print(f"{'=' * 60}\n")

    # Load and convert using the existing KerasHub HF converters.
    hf_uri = f"hf://{hf_model_id}"
    print(f"Loading from preset: {hf_uri}")
    embedder = BertTextEmbedder.from_preset(hf_uri)

    print(f"Backbone parameters: {embedder.backbone.count_params():,}")

    # Validate embeddings for manual review.
    validate_output(embedder, hf_model_id)

    # Always save the preset.
    print(f"\nSaving to preset: ./{preset}")
    embedder.save_to_preset(preset)
    print(f"\n✅ Successfully converted and saved to: ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
