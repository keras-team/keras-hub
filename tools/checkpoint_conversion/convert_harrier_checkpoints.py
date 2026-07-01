"""Convert and validate Harrier text embedding checkpoints.

This script handles HuggingFace checkpoints for the Microsoft harrier-oss
family. Weight loading uses the standard KerasHub ``from_preset("hf://...")``
path:
  - ``harrier_embedding_oss_06b`` loads via ``Qwen3TextEmbedder``
  - ``harrier_embedding_oss_270m`` loads via ``Gemma3TextEmbedder``

Validation compares the converted embedder against the HF AutoModel reference
using last-token pool + L2 norm.

Usage::

    python -m tools.checkpoint_conversion.convert_harrier_checkpoints \
        --preset harrier_embedding_oss_06b

    python -m tools.checkpoint_conversion.convert_harrier_checkpoints \
        --preset harrier_embedding_oss_270m
"""

import os
import random

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
from absl import app
from absl import flags

random.seed(123)
torch.manual_seed(123)
device = torch.device("cpu")
torch.set_default_device(device)

from transformers import AutoModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import keras_hub  # noqa: E402

PRESET_MAP = {
    "harrier_embedding_oss_06b": "microsoft/harrier-oss-v1-0.6b",
    "harrier_embedding_oss_270m": "microsoft/harrier-oss-v1-270m",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_string(
    "upload_uri",
    None,
    "Kaggle URI to upload the preset to, e.g. "
    '"kaggle://keras/qwen3/keras/harrier_embedding_oss_06b". Optional.',
)


# =============================================================================
# HF reference embedding
# =============================================================================


def _hf_embed(texts, hf_tokenizer, hf_model, eos_token=None):
    """Tokenize, run AutoModel, last-token pool, L2 norm.

    Processes one sequence at a time to avoid batch-padding effects.

    Args:
        texts: List of input strings.
        hf_tokenizer: HuggingFace tokenizer.
        hf_model: HuggingFace AutoModel.
        eos_token: Optional string token to append after tokenization. When
            provided, `add_special_tokens=False` is used so the tokenizer
            does not inject its own special tokens (Qwen3 path — no BOS,
            manual `<|im_end|>` append). When `None`, the tokenizer's
            default special-token behaviour is used (Gemma3 path — BOS added
            automatically by the tokenizer).

    Returns:
        np.ndarray of shape `(len(texts), hidden_dim)` with L2-normalized
        embeddings.
    """
    results = []
    for text in texts:
        enc = hf_tokenizer(
            [text],
            padding=False,
            truncation=True,
            max_length=32767,
            return_tensors="pt",
            add_special_tokens=(eos_token is None),
        )
        if eos_token is not None:
            eos_id = hf_tokenizer.convert_tokens_to_ids(eos_token)
            eos_col = torch.full((1, 1), eos_id, dtype=torch.long)
            ones_col = torch.ones((1, 1), dtype=torch.long)
            enc["input_ids"] = torch.cat([enc["input_ids"], eos_col], dim=1)
            enc["attention_mask"] = torch.cat(
                [enc["attention_mask"], ones_col], dim=1
            )
        with torch.no_grad():
            out = hf_model(**enc)
        # Last real token: attention_mask.sum() - 1. Works for both paths
        # since we process one sequence at a time with no padding.
        seq_len = int(enc["attention_mask"].sum()) - 1
        last = out.last_hidden_state[0, seq_len, :].float().numpy()
        norm = np.linalg.norm(last)
        results.append(last / norm)
    return np.stack(results)


# =============================================================================
# Validation
# =============================================================================


def validate_output(embedder, hf_model_id, eos_token=None):
    """Validate embedder parity against the HF AutoModel reference.

    Performs:
    1. Parameter count check.
    2. Embedding output parity (mean / max absolute difference).
    3. L2 norm check (output must be unit-length).
    4. Cosine similarity ranking consistency.
    5. Semantic search ranking consistency.

    Args:
        embedder: Converted `TextEmbedder` instance (Qwen3 or Gemma3).
        hf_model_id: HuggingFace model ID string.
        eos_token: Passed through to `_hf_embed`. `None` uses the
            tokenizer's default special tokens (Gemma3); a token string such
            as `"<|im_end|>"` disables auto-special-tokens and appends the
            EOS manually (Qwen3).

    Returns:
        bool: True if all checks pass.
    """
    print(f"\nLoading AutoModel + AutoTokenizer: {hf_model_id} (float32)")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    hf_model = AutoModel.from_pretrained(hf_model_id, dtype=torch.float32)
    hf_model.eval()

    print("\n" + "=" * 60)
    print("NUMERICAL VERIFICATION (Embedding)")
    print("=" * 60)

    # =========================================
    # PARAMETER COUNT
    # =========================================
    print("\n--- Parameter Count ---")
    keras_params = embedder.count_params()
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

    print("\nComputing HF embeddings (AutoModel, last-token + L2 norm)...")
    hf_embeddings = _hf_embed(test_texts, hf_tokenizer, hf_model, eos_token)
    print(f"HF shape: {hf_embeddings.shape}")
    print(f"HF[0][:5]: {hf_embeddings[0][:5]}")

    embedder_cls = type(embedder).__name__
    print(f"\nComputing KerasHub embeddings ({embedder_cls}.predict)...")
    keras_embeddings = np.array(embedder.predict(test_texts))
    print(f"KerasHub shape: {keras_embeddings.shape}")
    print(f"KerasHub[0][:5]: {keras_embeddings[0][:5]}")

    max_diff = np.max(np.abs(hf_embeddings - keras_embeddings))
    mean_diff = np.mean(np.abs(hf_embeddings - keras_embeddings))
    print(f"\nMax absolute diff:  {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    # =========================================
    # L2 NORM CHECK
    # =========================================
    print("\n--- L2 Norm Check (should be ~1.0) ---")
    keras_norms = np.linalg.norm(keras_embeddings, axis=1)
    hf_norms = np.linalg.norm(hf_embeddings, axis=1)
    print(f"KerasHub norms: {keras_norms}")
    print(f"HF norms:       {hf_norms}")
    norms_ok = np.allclose(keras_norms, 1.0, atol=1e-5)

    # Cosine similarity ranking (sentences 0 & 1 should be closer than 0 & 2).
    keras_batch_sims = keras_embeddings @ keras_embeddings.T
    hf_batch_sims = hf_embeddings @ hf_embeddings.T
    keras_ranking_ok = bool(keras_batch_sims[0, 1] > keras_batch_sims[0, 2])
    hf_ranking_ok = bool(hf_batch_sims[0, 1] > hf_batch_sims[0, 2])
    print(
        f"\nRanking consistency: KerasHub={keras_ranking_ok}, "
        f"HF={hf_ranking_ok}"
    )

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

    keras_q = np.array(embedder.encode_text(query))
    keras_d = np.array(embedder.encode_text(documents))
    keras_sims = np.array(embedder.similarity(keras_q, keras_d))
    keras_best = int(np.argmax(keras_sims))

    hf_q = _hf_embed([query], hf_tokenizer, hf_model, eos_token)
    hf_d = _hf_embed(documents, hf_tokenizer, hf_model, eos_token)
    hf_sims = hf_q @ hf_d.T
    hf_best = int(np.argmax(hf_sims))

    print(f"\nKerasHub sims: {keras_sims[0]} -> Best: {documents[keras_best]}")
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


# =============================================================================
# Main
# =============================================================================


def main(_):
    if FLAGS.preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{FLAGS.preset}'. "
            f"Must be one of: {', '.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    if preset == "harrier_embedding_oss_06b":
        print(f"\nLoading Qwen3TextEmbedder from hf://{hf_preset} ...")
        embedder = keras_hub.models.Qwen3TextEmbedder.from_preset(
            f"hf://{hf_preset}"
        )
        eos_token = "<|im_end|>"
    else:
        print(f"\nLoading Gemma3TextEmbedder from hf://{hf_preset} ...")
        embedder = keras_hub.models.Gemma3TextEmbedder.from_preset(
            f"hf://{hf_preset}"
        )
        eos_token = None
    print("\n-> Embedder loaded")

    passed = validate_output(embedder, hf_preset, eos_token=eos_token)
    if not passed:
        print("\n⚠️  Verification failed. Preset not saved.")
        return

    embedder.save_to_preset(f"./{preset}")
    print(f"\n✅ Embedder preset saved to ./{preset}/")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
