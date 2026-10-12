"""Convert ModernBERT sentence-embedding checkpoints to KerasHub presets.

Supported checkpoints:

- `nomic-ai/modernbert-embed-base`: mean pooling with L2 normalization.
  Prefix queries with `"search_query: "` and documents with
  `"search_document: "`.
- `Alibaba-NLP/gte-modernbert-base`: `[CLS]` pooling, without
  normalization.

The checkpoint is loaded through `ModernBertTextEmbedder.from_preset()` with
an `hf://` handle, which reads the pooling strategy, normalization and
sequence length from the checkpoint's sentence-transformers config files.
The converted model is then verified against
`sentence_transformers.SentenceTransformer` before the preset is saved.

Setup:
```shell
pip install -e .
pip install sentence-transformers
```

Usage (from the repository root):
```shell
python \
    tools/checkpoint_conversion/convert_modern_bert_embedding_checkpoints.py \
    --preset modern_bert_embed_base_en
```

"""

import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_TORCH_DEVICE"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from absl import app
from absl import flags
from keras import ops
from sentence_transformers import SentenceTransformer

import keras_hub

PRESET_MAP = {
    "modern_bert_embed_base_en": "nomic-ai/modernbert-embed-base",
    "gte_modern_bert_base_en": "Alibaba-NLP/gte-modernbert-base",
}

# Task prefixes each checkpoint expects on queries and documents.
PROMPTS = {
    "modern_bert_embed_base_en": ("search_query: ", "search_document: "),
    "gte_modern_bert_base_en": ("", ""),
}

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "preset",
    None,
    f"Preset name to convert. Must be one of {', '.join(PRESET_MAP)}.",
)

# Validation runs at a shorter sequence length than the checkpoint default
# (up to 8192) to keep memory use reasonable. All validation texts fit, so no
# truncation happens on either side. The original value is restored before
# the preset is saved.
VALIDATION_SEQUENCE_LENGTH = 512

QUERY = "Which planet is known as the Red Planet?"
DOCUMENTS = [
    "Venus is often called Earth's twin.",
    "Mars is often referred to as the Red Planet.",
    # Longer than ModernBERT's 128-token local attention window, so both the
    # local and the global attention layers are exercised, and the batch
    # contains padding.
    " ".join(
        [
            "Jupiter is the largest planet in the Solar System, and its "
            "Great Red Spot is a giant storm that has raged for centuries."
        ]
        * 8
    ),
]

# Parity thresholds. Embeddings are compared at a relative scale so that the
# same thresholds work for normalized and unnormalized checkpoints.
MAX_RELATIVE_DIFF = 1e-3
MIN_COSINE_SIMILARITY = 0.99999


def _cosine_similarity(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


def validate_output(keras_embedder, st_model, query_prefix, document_prefix):
    """Check the converted model against sentence-transformers.

    Checks the parameter count, tokenization, embedding parity and semantic
    search ranking on a padded batch of mixed-length inputs.

    Returns:
        bool: True if all checks pass.
    """
    query = query_prefix + QUERY
    documents = [document_prefix + document for document in DOCUMENTS]
    texts = [query] + documents
    passed = True

    print("\n Parameter count ")
    keras_params = keras_embedder.backbone.count_params()
    st_params = sum(p.numel() for p in st_model.parameters())
    print(f"KerasHub: {keras_params:,}")
    print(f"sentence-transformers: {st_params:,}")
    if keras_params != st_params:
        print("❌ Parameter count mismatch")
        passed = False

    print("\n Tokenization ")
    keras_inputs = keras_embedder.preprocessor(texts)
    keras_ids = ops.convert_to_numpy(keras_inputs["token_ids"])
    keras_mask = ops.convert_to_numpy(keras_inputs["padding_mask"])
    st_inputs = st_model.tokenize(texts)
    st_ids = st_inputs["input_ids"].numpy()
    st_mask = st_inputs["attention_mask"].numpy()
    for i, text in enumerate(texts):
        keras_tokens = keras_ids[i][keras_mask[i].astype(bool)]
        st_tokens = st_ids[i][st_mask[i].astype(bool)]
        if not np.array_equal(keras_tokens, st_tokens):
            print(f"❌ Token ids differ for {text[:40]!r}...")
            passed = False
    print(f"Sequence lengths: {keras_mask.sum(axis=-1).tolist()}")

    print("\n Embedding parity ")
    st_embeddings = st_model.encode(texts, convert_to_numpy=True)
    keras_embeddings = np.asarray(keras_embedder.encode_text(texts))
    scale = np.abs(st_embeddings).max()
    max_relative_diff = np.abs(st_embeddings - keras_embeddings).max() / scale
    cosine = np.diag(_cosine_similarity(keras_embeddings, st_embeddings))
    print(f"Embedding shape: {keras_embeddings.shape}")
    print(f"Max |diff| / max |value|: {max_relative_diff:.2e}")
    print(f"Min cosine(KerasHub, sentence-transformers): {cosine.min():.7f}")
    print(f"Norms: {np.linalg.norm(keras_embeddings, axis=-1)}")
    if max_relative_diff > MAX_RELATIVE_DIFF:
        print(f"❌ Relative difference exceeds {MAX_RELATIVE_DIFF}")
        passed = False
    if cosine.min() < MIN_COSINE_SIMILARITY:
        print(f"❌ Cosine similarity below {MIN_COSINE_SIMILARITY}")
        passed = False

    print("\nSemantic search ")
    # sentence-transformers scores with cosine similarity, which is also
    # correct for checkpoints that do not normalize their embeddings.
    keras_scores = _cosine_similarity(
        keras_embeddings[:1], keras_embeddings[1:]
    )[0]
    st_scores = _cosine_similarity(st_embeddings[:1], st_embeddings[1:])[0]
    print(f"KerasHub scores: {keras_scores}")
    print(f"sentence-transformers scores: {st_scores}")
    if np.argmax(keras_scores) != np.argmax(st_scores):
        print("❌ Best match differs")
        passed = False
    else:
        print(f"Best match: {documents[np.argmax(keras_scores)]!r}")

    print("\n✅ All checks passed" if passed else "\n❌ Verification failed")
    return passed


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {preset!r}. Must be one of {list(PRESET_MAP)}."
        )
    hf_model_id = PRESET_MAP[preset]
    print(f"Converting {hf_model_id} -> {preset}")

    keras_embedder = keras_hub.models.ModernBertTextEmbedder.from_preset(
        f"hf://{hf_model_id}",
        dtype="float32",
    )
    print(
        f"pooling_mode={keras_embedder.pooling_mode}, "
        f"normalize={keras_embedder.normalize}, "
        f"sequence_length={keras_embedder.preprocessor.sequence_length}"
    )

    st_model = SentenceTransformer(hf_model_id, device="cpu")
    # Compare in float32. Recent `transformers` versions run checkpoints in
    # their stored dtype, and some are stored in half precision (e.g.
    # `Alibaba-NLP/gte-modernbert-base` is float16), which would make the
    # reference less precise than the float32 KerasHub model.
    st_model = st_model.float()
    st_model.eval()
    print(f"sentence-transformers max_seq_length: {st_model.max_seq_length}")

    original_sequence_length = keras_embedder.preprocessor.sequence_length
    keras_embedder.preprocessor.sequence_length = VALIDATION_SEQUENCE_LENGTH
    passed = validate_output(keras_embedder, st_model, *PROMPTS[preset])
    keras_embedder.preprocessor.sequence_length = original_sequence_length
    if not passed:
        print("Preset not saved.")
        return

    keras_embedder.save_to_preset(f"./{preset}")
    print(f"Saved preset to ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
