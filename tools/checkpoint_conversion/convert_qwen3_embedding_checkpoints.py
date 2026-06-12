"""Convert and validate Qwen3-based text embedding checkpoints.

This script handles HuggingFace checkpoints that declare
``"architectures": ["Qwen3Model"]``, such as the Microsoft harrier-oss
family.  Weight loading goes through
``keras_hub.src.utils.transformers.convert_qwen3_embedding`` via the
standard KerasHub ``from_preset("hf://...")`` path.

Validation reproduces the reference embedding pipeline entirely from the
``Qwen3Backbone`` without requiring the ``Qwen3TextEmbedder`` class (which
lives on a separate PR):

  token_ids/padding_mask → Qwen3Backbone → last-token pool → L2 norm

Usage::

    python -m tools.checkpoint_conversion.convert_qwen3_embedding_checkpoints \
        --preset harrier_oss_v1_0.6b_en --upload_uri \
        "kaggle://kerashub/qwen-3-embedding/keras/harrier_oss_v1_0.6b_en"
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

from keras import ops  # noqa: E402
from transformers import AutoModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import keras_hub  # noqa: E402

PRESET_MAP = {
    "harrier_oss_v1_0.6b_en": "microsoft/harrier-oss-v1-0.6b",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_string(
    "upload_uri",
    None,
    "Kaggle URI to upload the preset to, e.g. "
    "kaggle://kerashub/qwen-3-embedding/keras/harrier_oss_v1_0.6b_en",
)


# =============================================================================
# Embedding helpers (no Qwen3TextEmbedder dependency)
# =============================================================================


def _last_token_pool(sequence_output, padding_mask):
    """Last non-padding token pooling.

    Identical to ``Qwen3TextEmbedder._last_token_pooling`` so that the
    backbone-only validation produces the same result as the full embedder.
    """
    mask = ops.cast(padding_mask, sequence_output.dtype)
    mask_shifted = ops.pad(mask, [[0, 0], [0, 1]])[:, 1:]
    last_token_mask = mask * (1.0 - mask_shifted)
    return ops.sum(
        sequence_output * ops.expand_dims(last_token_mask, axis=-1),
        axis=1,
    )


def _l2_normalize(embeddings):
    return ops.nn.normalize(embeddings, axis=-1, order=2)


def _keras_embed(backbone, token_ids_np, padding_mask_np):
    """Run backbone + pool + normalize and return a numpy array."""
    inputs = {
        "token_ids": token_ids_np,
        "padding_mask": padding_mask_np,
    }
    seq_out = backbone(inputs)
    pooled = _last_token_pool(seq_out, padding_mask_np)
    normed = _l2_normalize(pooled)
    return ops.convert_to_numpy(normed)


def _build_inputs(texts, keras_tokenizer, sequence_length=256):
    """Tokenize texts with the KerasHub tokenizer and build backbone inputs.

    Appends ``end_token_id`` (<|im_end|>, 151645) to each sequence and pads
    to ``sequence_length`` with ``pad_token_id`` (<|endoftext|>, 151643),
    matching what ``Qwen3TextEmbedderPreprocessor`` produces.
    """
    eos_id = keras_tokenizer.end_token_id
    pad_id = keras_tokenizer.pad_token_id

    token_ids_batch = []
    padding_mask_batch = []

    for text in texts:
        # Tokenize without any special tokens; KerasHub tokenizer returns
        # a ragged/dense tensor of content token ids.
        ids = (
            ops.convert_to_numpy(keras_tokenizer(text))
            .astype(np.int32)
            .flatten()
        )
        # Truncate to leave room for EOS, then append it.
        ids = ids[: sequence_length - 1]
        ids = np.concatenate([ids, [eos_id]]).astype(np.int32)
        content_len = len(ids)

        padded_ids = np.full(sequence_length, pad_id, dtype=np.int32)
        padded_mask = np.zeros(sequence_length, dtype=np.int32)
        padded_ids[:content_len] = ids
        padded_mask[:content_len] = 1

        token_ids_batch.append(padded_ids)
        padding_mask_batch.append(padded_mask)

    return np.stack(token_ids_batch), np.stack(padding_mask_batch)


# =============================================================================
# HF reference embedding
# =============================================================================


def _hf_embed(texts, hf_tokenizer, hf_model):
    """Tokenize, run AutoModel, last-token pool, L2 norm.

    Processes one sequence at a time to avoid batch-padding effects.
    Uses ``add_special_tokens=False`` + explicit ``<|im_end|>`` append to
    produce an identical token sequence to ``_build_inputs`` above.
    """
    eos_id = hf_tokenizer.convert_tokens_to_ids("<|im_end|>")
    results = []
    for text in texts:
        enc = hf_tokenizer(
            [text],
            padding=False,
            truncation=True,
            max_length=32767,
            return_tensors="pt",
            add_special_tokens=False,
        )
        eos_col = torch.full((1, 1), eos_id)
        ones_col = torch.ones((1, 1), dtype=torch.long)
        input_ids = torch.cat([enc["input_ids"], eos_col], dim=1)
        attention_mask = torch.cat([enc["attention_mask"], ones_col], dim=1)
        with torch.no_grad():
            out = hf_model(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state[0, -1, :].float().numpy()
        norm = np.linalg.norm(last)
        results.append(last / norm)
    return np.stack(results)


# =============================================================================
# Validation
# =============================================================================


def validate_embedding_output(backbone, keras_tokenizer, hf_model_id):
    """Validate backbone embedding parity against the HF AutoModel reference.

    Performs:
    1. Parameter count check.
    2. Embedding output parity (mean / max absolute difference).
    3. L2 norm check (output must be unit-length).
    4. Cosine similarity ranking consistency.
    5. Semantic search ranking consistency.

    Args:
        backbone: Converted ``Qwen3Backbone`` instance.
        hf_model_id: HuggingFace model ID string.

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
    keras_params = backbone.count_params()
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
    hf_embeddings = _hf_embed(test_texts, hf_tokenizer, hf_model)
    print(f"HF shape: {hf_embeddings.shape}")
    print(f"HF[0][:5]: {hf_embeddings[0][:5]}")

    print("\nComputing KerasHub embeddings (backbone + pool + norm)...")
    token_ids, padding_mask = _build_inputs(test_texts, keras_tokenizer)
    keras_embeddings = _keras_embed(backbone, token_ids, padding_mask)
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

    q_ids, q_mask = _build_inputs([query], keras_tokenizer)
    d_ids, d_mask = _build_inputs(documents, keras_tokenizer)
    keras_q = _keras_embed(backbone, q_ids, q_mask)
    keras_d = _keras_embed(backbone, d_ids, d_mask)
    keras_sims = keras_q @ keras_d.T
    keras_best = int(np.argmax(keras_sims))

    hf_q = _hf_embed([query], hf_tokenizer, hf_model)
    hf_d = _hf_embed(documents, hf_tokenizer, hf_model)
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

    # Load backbone via KerasHub's from_preset.  The preset_loader routes
    # Qwen3Model-architecture checkpoints to convert_qwen3_embedding.py
    # which handles the harrier weight-key prefix automatically.
    print(f"\nLoading Qwen3Backbone from hf://{hf_preset} ...")
    backbone = keras_hub.models.Qwen3Backbone.from_preset(f"hf://{hf_preset}")
    keras_tokenizer = keras_hub.models.Qwen3Tokenizer.from_preset(
        f"hf://{hf_preset}"
    )

    print("\n-> Backbone loaded")

    passed = validate_embedding_output(backbone, keras_tokenizer, hf_preset)
    if not passed:
        print("\n⚠️  Verification failed. Preset not saved.")
        return

    # Save backbone preset.
    # TODO: once the Qwen3TextEmbedder PR is merged, save a
    # Qwen3TextEmbedder preset instead so the preprocessor and pooling
    # configuration are included.
    backbone.save_to_preset(f"./{preset}")
    print(f"\n✅ Backbone preset saved to ./{preset}/")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
