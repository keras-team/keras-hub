"""
Convert HuggingFace intfloat/multilingual-e5-* checkpoints to KerasHub format.

Multilingual-E5 models are BERT encoders with an XLM-RoBERTa SentencePiece
tokenizer, fine-tuned for multilingual dense retrieval using weakly-supervised
contrastive pre-training. Embeddings are computed as mean-pooled token outputs
followed by L2 normalization. This maps to:

    BertTextEmbedder(
        backbone=backbone,
        preprocessor=preprocessor,
        pooling_mode="mean",
        normalize=True,
    )

Usage convention: prefix queries with "query: " and documents with "passage: ".

Setup:
```shell
pip install keras-hub keras transformers safetensors huggingface_hub \
    torch sentencepiece
```

Usage:
```shell
cd tools/checkpoint_conversion
python convert_multilingual_e5_checkpoints.py --preset multilingual_e5_small
python convert_multilingual_e5_checkpoints.py \\
    --preset multilingual_e5_small \\
    --upload_uri kaggle://keras/multilingual-e5/keras/multilingual_e5_small
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
from keras import ops
from transformers import AutoModel
from transformers import AutoTokenizer

import keras_hub
from keras_hub.src.models.bert.bert_text_embedder import BertTextEmbedder
from keras_hub.src.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)
from keras_hub.src.utils.transformers.convert_bert import (
    convert_backbone_config,
)
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
    "Preset name to convert. Must be one of the keys in PRESET_MAP.",
)
flags.DEFINE_string(
    "upload_uri",
    None,
    "Kaggle URI to upload the preset to, e.g. "
    '"kaggle://keras/multilingual-e5/keras/{preset}". Optional.',
)

PRESET_MAP = {"multilingual_e5_small": "intfloat/multilingual-e5-small"}

# Multilingual test sentences covering three language families.
TEST_TEXTS = [
    "query: What is the capital of France?",
    "query: Quelle est la capitale de la France?",
    "query: 法国的首都是哪里？",
]
QUERY_DOC_PAIRS = (
    "query: Which planet is known as the Red Planet?",
    [
        "passage: Venus is often called Earth's twin.",
        "passage: Mars is often referred to as the Red Planet.",
        "passage: Jupiter has a prominent red spot.",
    ],
)

# Each triplet: (anchor_EN, positive_other_lang, negative_other_lang).
# sim(anchor, positive) must exceed sim(anchor, negative) to pass.
CROSS_LINGUAL_TRIPLETS = [
    (
        "query: What is the capital of France?",
        "query: Quelle est la capitale de la France?",  # FR — same meaning
        "query: Quel temps fait-il aujourd'hui?",  # FR — different meaning
    ),
    (
        "query: What is the capital of France?",
        "query: 法国的首都是哪里？",  # ZH — same meaning
        "query: 今天天气怎么样？",  # ZH — different meaning
    ),
]


def _hf_mean_pool(outputs, attention_mask):
    """Mean-pool HuggingFace token embeddings weighted by attention mask."""
    token_embeds = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    sum_embeds = torch.sum(token_embeds * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    embeddings = sum_embeds / sum_mask
    # L2 normalize.
    return (embeddings / embeddings.norm(dim=-1, keepdim=True)).numpy()


def _hf_encode(hf_model, hf_tokenizer, texts):
    """Encode texts using HuggingFace model with mean pooling + L2 norm."""
    encoded = hf_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = hf_model(**encoded)
    return _hf_mean_pool(outputs, encoded["attention_mask"])


def validate_output(keras_embedder, hf_model_id):
    """Validate numerical parity between the converted KerasHub model and HF.

    Performs five checks:
    1. Parameter count verification.
    2. Embedding output comparison (max/mean absolute difference).
    3. L2 norm check (embeddings must be unit-length).
    4. Cosine similarity ranking consistency across multilingual inputs.
    5. Semantic search ranking consistency (query vs. passages).

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
    # EMBEDDING PARITY (MULTILINGUAL)
    # =========================================
    print("\n--- Embedding Parity (multilingual) ---")
    print(f"Test inputs: {TEST_TEXTS}")

    print("\nComputing HuggingFace embeddings (mean pool + L2 norm)...")
    hf_embeddings = _hf_encode(hf_model, hf_tokenizer, TEST_TEXTS)
    print(f"HF shape: {hf_embeddings.shape}")
    print(f"HF[0][:5]: {hf_embeddings[0][:5]}")

    print("\nComputing KerasHub embeddings...")
    keras_embeddings = np.array(keras_embedder.encode_text(TEST_TEXTS))
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

    # =========================================
    # CROSS-LINGUAL RANKING
    # =========================================
    print("\n--- Cross-lingual Ranking ---")
    print("sim(anchor, same-meaning in other lang) must exceed")
    print("sim(anchor, different-meaning in other lang)")
    lang_labels = ["EN↔FR", "EN↔ZH"]
    cross_lingual_ok = True
    for label, (anchor, positive, negative) in zip(
        lang_labels, CROSS_LINGUAL_TRIPLETS
    ):
        embs = np.array(
            keras_embedder.encode_text([anchor, positive, negative])
        )
        # similarity returns shape (1, 2): [anchor↔positive, anchor↔negative]
        sims = ops.convert_to_numpy(
            keras_embedder.similarity(embs[0:1], embs[1:])
        )
        sim_pos, sim_neg = float(sims[0, 0]), float(sims[0, 1])
        ok = sim_pos > sim_neg
        cross_lingual_ok = cross_lingual_ok and ok
        status = "✅" if ok else "❌"
        print(
            f"{label}: sim(same)={sim_pos:.4f} > sim(diff)={sim_neg:.4f} "
            f"{status}"
        )

    # =========================================
    # SEMANTIC SEARCH RANKING
    # =========================================
    print("\n--- Semantic Search Ranking ---")
    query, documents = QUERY_DOC_PAIRS
    print(f"Query: {query}")
    print(f"Documents: {documents}")

    keras_q = np.array(keras_embedder.encode_text(query))
    keras_d = np.array(keras_embedder.encode_documents(documents))
    keras_sims = ops.convert_to_numpy(
        keras_embedder.similarity(keras_q, keras_d)
    )
    keras_best = int(np.argmax(keras_sims))

    hf_q = _hf_encode(hf_model, hf_tokenizer, [query])
    hf_d = _hf_encode(hf_model, hf_tokenizer, documents)
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

    if not cross_lingual_ok:
        print(
            "❌ FAILED: Cross-lingual ranking — same-meaning pair scored "
            "below unrelated pair"
        )
        passed = False

    if not search_ok:
        print("❌ FAILED: Semantic search ranking mismatch")
        passed = False

    if passed:
        print("✅ ALL CHECKS PASSED")

    return passed


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

        # Build XLM-RoBERTa tokenizer from SentencePiece proto.
        print("\nDownloading sentencepiece.bpe.model...")
        hf_hub_download(
            hf_model_id, "sentencepiece.bpe.model", local_dir=temp_dir
        )
        tokenizer = XLMRobertaTokenizer(
            proto=os.path.join(temp_dir, "sentencepiece.bpe.model")
        )

        # Download sentence-transformer config files to derive pooling,
        # normalization, and sequence length dynamically.
        for fname in (
            "modules.json",
            "1_Pooling/config.json",
            "sentence_bert_config.json",
        ):
            hf_hub_download(hf_model_id, fname, local_dir=temp_dir)

        task_config = load_task_config(temp_dir, transformers_config)
        preprocessor_config = load_preprocessor_config(
            temp_dir, transformers_config
        )
        print(f"Task config: {task_config}")
        print(f"Preprocessor config: {preprocessor_config}")

        # Assemble BertTextEmbedder from config-derived settings.
        preprocessor = keras_hub.models.BertTextEmbedderPreprocessor(
            tokenizer=tokenizer,
            **preprocessor_config,
        )
        embedder = BertTextEmbedder(
            backbone=backbone,
            preprocessor=preprocessor,
            **task_config,
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
