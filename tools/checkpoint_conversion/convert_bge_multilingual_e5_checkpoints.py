"""
Convert HuggingFace BAAI/bge-* and intfloat/multilingual-e5-* checkpoints to
KerasHub format.

Three model families are supported:

**BGE English/Chinese models** (BAAI/bge-{small,base,large}-{en,zh}* and
BAAI/llm-embedder) are BERT encoders using a WordPiece tokenizer and CLS
token pooling followed by L2 normalization. They map to BertTextEmbedder.

**Multilingual-E5 small** (intfloat/multilingual-e5-small) is also a BERT
encoder but uses an XLM-RoBERTa SentencePiece tokenizer and mean pooling.
It maps to BertTextEmbedder.

**Multilingual-E5 base/large** (intfloat/multilingual-e5-{base,large}) and
**BGE-M3** (BAAI/bge-m3) are pure XLM-RoBERTa encoders that map to
XLMRobertaTextEmbedder. Multilingual-E5 uses mean pooling; BGE-M3 uses CLS
pooling. All support 100+ languages.

Usage convention for E5 models: prefix queries with "query: " and documents
with "passage: ".

Setup:
```shell
pip install keras-hub keras transformers safetensors huggingface_hub \
    torch sentencepiece
```

Usage:
```shell
cd tools/checkpoint_conversion
# BGE (BERT-based)
python convert_multilingual_e5_checkpoints.py --preset bge_small_en_v1.5
python convert_multilingual_e5_checkpoints.py \\
    --preset bge_small_en_v1.5 \\
    --upload_uri kaggle://keras/bge/keras/bge_small_en_v1.5

# BGE-M3 (XLM-RoBERTa-based)
python convert_multilingual_e5_checkpoints.py --preset bge_m3
python convert_multilingual_e5_checkpoints.py \\
    --preset bge_m3 \\
    --upload_uri kaggle://keras/bge/keras/bge_m3

# Multilingual-E5 small (BERT-based)
python convert_multilingual_e5_checkpoints.py --preset multilingual_e5_small
python convert_multilingual_e5_checkpoints.py \\
    --preset multilingual_e5_small \\
    --upload_uri kaggle://keras/multilingual-e5/keras/multilingual_e5_small

# Multilingual-E5 base/large (XLM-RoBERTa-based)
python convert_multilingual_e5_checkpoints.py --preset multilingual_e5_base
python convert_multilingual_e5_checkpoints.py \\
    --preset multilingual_e5_large \\
    --upload_uri kaggle://keras/multilingual-e5/keras/multilingual_e5_large
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
from keras_hub.src.models.xlm_roberta.xlm_roberta_text_embedder import (
    XLMRobertaTextEmbedder,
)
from keras_hub.src.models.xlm_roberta.xlm_roberta_text_embedder_preprocessor import (  # noqa: E501
    XLMRobertaTextEmbedderPreprocessor,
)
from keras_hub.src.utils.transformers import convert_bert as bert_converter
from keras_hub.src.utils.transformers import (
    convert_xlm_roberta as xlm_converter,
)
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

PRESET_MAP = {
    "bge_small_en": "BAAI/bge-small-en",
    "bge_base_en": "BAAI/bge-base-en",
    "bge_large_en": "BAAI/bge-large-en",
    "bge_small_en_v1.5": "BAAI/bge-small-en-v1.5",
    "bge_base_en_v1.5": "BAAI/bge-base-en-v1.5",
    "bge_large_en_v1.5": "BAAI/bge-large-en-v1.5",
    "bge_small_zh": "BAAI/bge-small-zh",
    "bge_base_zh": "BAAI/bge-base-zh",
    "bge_large_zh": "BAAI/bge-large-zh",
    "bge_small_zh_v1.5": "BAAI/bge-small-zh-v1.5",
    "bge_base_zh_v1.5": "BAAI/bge-base-zh-v1.5",
    "bge_large_zh_v1.5": "BAAI/bge-large-zh-v1.5",
    "llm_embedder": "BAAI/llm-embedder",
    # BGE-M3 (pure XLM-RoBERTa, multilingual).
    "bge_m3": "BAAI/bge-m3",
    # Multilingual-E5 small (BERT + XLM-R tokenizer).
    "multilingual_e5_small": "intfloat/multilingual-e5-small",
    # Multilingual-E5 base/large (pure XLM-RoBERTa).
    "multilingual_e5_base": "intfloat/multilingual-e5-base",
    "multilingual_e5_large": "intfloat/multilingual-e5-large",
}


# Pure XLM-RoBERTa architecture presets (excludes multilingual_e5_small which
# is BERT with an XLM-R tokenizer).
_XLM_ROBERTA_PRESETS = frozenset(
    {"bge_m3", "multilingual_e5_base", "multilingual_e5_large"}
)


def _is_xlm_roberta(preset):
    """Return True for pure XLM-RoBERTa architecture presets."""
    return preset in _XLM_ROBERTA_PRESETS


def _use_mean_pool(preset):
    """Return True for presets that use mean pooling (all ME5 variants)."""
    return preset.startswith("multilingual_e5")


def _is_multilingual(preset):
    """Return True for multilingual presets; enables cross-lingual checks."""
    return preset in _XLM_ROBERTA_PRESETS or preset == "multilingual_e5_small"


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

# English test sentences for BGE embedding parity check.
# Sentences 0 and 1 are semantically closer than 0 and 2.
BGE_TEST_TEXTS = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

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
    mask = attention_mask.unsqueeze(-1).float()
    sum_embeds = torch.sum(token_embeds * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    embeddings = sum_embeds / sum_mask
    # L2 normalize.
    return (embeddings / embeddings.norm(dim=-1, keepdim=True)).numpy()


def _hf_cls_pool(outputs):
    """CLS-pool HuggingFace token embeddings with L2 normalization."""
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
    return cls_embeddings / np.maximum(norms, 1e-12)


def _hf_encode(hf_model, hf_tokenizer, texts, mean_pool=False):
    """Encode texts using a HuggingFace model.

    Uses mean pooling + L2 norm when mean_pool=True (Multilingual-E5), or
    CLS token + L2 norm when mean_pool=False (BGE).
    """
    encoded = hf_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = hf_model(**encoded)
    if mean_pool:
        return _hf_mean_pool(outputs, encoded["attention_mask"])
    return _hf_cls_pool(outputs)


def validate_output(
    keras_embedder,
    hf_model_id,
    use_mean_pool=False,
    is_multilingual=False,
    is_xlm_roberta=False,
):
    """Validate numerical parity between the converted KerasHub model and HF.

    Performs checks appropriate for the model family:
    - All models: parameter count, embedding parity, L2 norm, semantic search.
    - English BGE: intra-batch cosine ranking consistency.
    - Multilingual (bge_m3, multilingual_e5_*): cross-lingual ranking
      consistency (EN↔FR, EN↔ZH).

    Args:
        keras_embedder: Converted KerasHub text embedder instance.
        hf_model_id: HuggingFace model ID to compare against.
        use_mean_pool: bool. True for mean pooling (multilingual_e5_*),
            False for CLS pooling (BGE / bge_m3).
        is_multilingual: bool. True for multilingual presets; enables
            cross-lingual ranking checks.
        is_xlm_roberta: bool. True for pure XLM-RoBERTa presets; adjusts
            param count to exclude token_type_embeddings.

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

    if is_xlm_roberta:
        hf_params = sum(
            p.numel()
            for name, p in hf_model.named_parameters()
            if not name.startswith("pooler.")
            and name != "embeddings.token_type_embeddings.weight"
        )
        hf_params -= 2 * hf_model.embeddings.position_embeddings.weight.shape[1]
    else:
        hf_params = sum(p.numel() for p in hf_model.parameters())

    parity_texts = TEST_TEXTS if is_multilingual else BGE_TEST_TEXTS
    query, documents = QUERY_DOC_PAIRS

    hf_embeddings = _hf_encode(
        hf_model, hf_tokenizer, parity_texts, mean_pool=use_mean_pool
    )
    hf_q = _hf_encode(hf_model, hf_tokenizer, [query], mean_pool=use_mean_pool)
    hf_d = _hf_encode(
        hf_model, hf_tokenizer, documents, mean_pool=use_mean_pool
    )
    hf_sims_search = hf_q @ hf_d.T
    hf_best = int(np.argmax(hf_sims_search))

    # =========================================
    # PARAMETER COUNT
    # =========================================
    print("\n--- Parameter Count ---")
    keras_params = keras_embedder.backbone.count_params()
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
    pool_desc = "mean pool" if use_mean_pool else "CLS pool"
    if is_multilingual:
        print(
            f"\n--- Embedding Parity (multilingual, {pool_desc} + L2 norm) ---"
        )
    else:
        print("\n--- Embedding Parity (CLS + L2 norm) ---")
    print(f"Test inputs: {parity_texts}")
    print(f"HF shape: {hf_embeddings.shape}")
    print(f"HF[0][:5]: {hf_embeddings[0][:5]}")

    del hf_model, hf_tokenizer

    print("\nComputing KerasHub embeddings...")
    if is_multilingual:
        keras_embeddings = np.array(keras_embedder.encode_text(parity_texts))
    else:
        keras_embeddings = np.array(keras_embedder.predict(parity_texts))

    print(f"KerasHub shape: {keras_embeddings.shape}")
    print(f"KerasHub[0][:5]: {keras_embeddings[0][:5]}")

    max_diff = np.max(np.abs(hf_embeddings - keras_embeddings))
    mean_diff = np.mean(np.abs(hf_embeddings - keras_embeddings))
    print(f"\nMax absolute diff:  {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    # =========================================
    # INTRA-BATCH RANKING (English BGE only)
    # Sentences 0 and 1 are semantically closer than 0 and 2.
    # =========================================
    ranking_ok = True
    if not is_multilingual:
        print("\n--- Ranking Consistency ---")
        keras_batch_sims = keras_embeddings @ keras_embeddings.T
        hf_batch_sims = hf_embeddings @ hf_embeddings.T
        keras_ranking_ok = bool(keras_batch_sims[0, 1] > keras_batch_sims[0, 2])
        hf_ranking_ok = bool(hf_batch_sims[0, 1] > hf_batch_sims[0, 2])
        ranking_ok = keras_ranking_ok
        print(
            f"Ranking consistency: KerasHub={keras_ranking_ok}, "
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
    # CROSS-LINGUAL RANKING (multilingual models)
    # =========================================
    cross_lingual_ok = True
    if is_multilingual:
        print("\n--- Cross-lingual Ranking ---")
        print("sim(anchor, same-meaning in other lang) must exceed")
        print("sim(anchor, different-meaning in other lang)")
        lang_labels = ["EN↔FR", "EN↔ZH"]
        for label, (anchor, positive, negative) in zip(
            lang_labels, CROSS_LINGUAL_TRIPLETS
        ):
            embs = np.array(
                keras_embedder.encode_text([anchor, positive, negative])
            )
            # similarity shape (1, 2): [anchor↔positive, anchor↔negative]
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
    print(f"Query: {query}")
    print(f"Documents: {documents}")

    keras_q = np.array(keras_embedder.encode_text(query))
    keras_d = np.array(keras_embedder.encode_documents(documents))
    keras_sims = ops.convert_to_numpy(
        keras_embedder.similarity(keras_q, keras_d)
    )
    keras_best = int(np.argmax(keras_sims))
    hf_sims = hf_sims_search

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

    if max_diff > 1e-3:
        print(f"❌ FAILED: Max diff {max_diff:.2e} exceeds 1e-3")
        passed = False

    if not norms_ok:
        print("❌ FAILED: Embeddings are not unit-length")
        passed = False

    if not is_multilingual and not ranking_ok:
        print("❌ FAILED: Ranking inconsistency")
        passed = False

    if is_multilingual and not cross_lingual_ok:
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
    is_xlm_r = _is_xlm_roberta(preset)
    use_mean_pool_flag = _use_mean_pool(preset)
    is_multi = _is_multilingual(preset)

    print(f"\n{'=' * 60}")
    print(f"Converting: {hf_model_id} -> {preset}")
    print(f"{'=' * 60}\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        print("Downloading config.json...")
        config_path = hf_hub_download(
            hf_model_id, "config.json", local_dir=temp_dir
        )
        with open(config_path, "r") as f:
            transformers_config = json.load(f)

        # Build KerasHub backbone from HF config.
        if is_xlm_r:
            keras_config = xlm_converter.convert_backbone_config(
                transformers_config
            )
        else:
            keras_config = bert_converter.convert_backbone_config(
                transformers_config
            )
        print(f"\nBackbone config: {keras_config}")
        if is_xlm_r:
            backbone = keras_hub.models.XLMRobertaBackbone(
                **keras_config, dtype="float32"
            )
        else:
            backbone = keras_hub.models.BertBackbone(
                **keras_config, dtype="float32"
            )
        print(f"Backbone parameters: {backbone.count_params():,}")

        print("\nDownloading model weights...")
        from huggingface_hub.errors import EntryNotFoundError

        try:
            hf_hub_download(
                hf_model_id, "model.safetensors", local_dir=temp_dir
            )
        except EntryNotFoundError:
            print(
                "model.safetensors not found on main — "
                "downloading pytorch_model.bin and converting to safetensors."
            )
            bin_path = hf_hub_download(
                hf_model_id, "pytorch_model.bin", local_dir=temp_dir
            )
            from safetensors.torch import save_file as _save_safetensors

            bin_state_dict = torch.load(
                bin_path, map_location="cpu", weights_only=True
            )
            _save_safetensors(
                bin_state_dict,
                os.path.join(temp_dir, "model.safetensors"),
            )
            del bin_state_dict

        print("Converting weights...")
        with SafetensorLoader(temp_dir) as loader:
            if is_xlm_r:
                xlm_converter.convert_weights(
                    backbone, loader, transformers_config
                )
            else:
                bert_converter.convert_weights(
                    backbone, loader, transformers_config
                )

        print("\nDownloading tokenizer files...")
        hf_hub_download(
            hf_model_id, "tokenizer_config.json", local_dir=temp_dir
        )
        if is_xlm_r:
            # Pure XLM-RoBERTa models always use SentencePiece.
            hf_hub_download(
                hf_model_id, "sentencepiece.bpe.model", local_dir=temp_dir
            )
            tokenizer = xlm_converter.convert_tokenizer(
                keras_hub.models.XLMRobertaTokenizer, temp_dir
            )
        else:
            # BERT-based models: download the appropriate vocab file.
            # multilingual_e5_small uses SentencePiece; others use WordPiece.
            if _use_mean_pool(preset):
                hf_hub_download(
                    hf_model_id,
                    "sentencepiece.bpe.model",
                    local_dir=temp_dir,
                )
            else:
                hf_hub_download(hf_model_id, "vocab.txt", local_dir=temp_dir)
            tokenizer = bert_converter.convert_tokenizer(
                keras_hub.models.BertTokenizer, temp_dir
            )

        for fname in (
            "modules.json",
            "1_Pooling/config.json",
            "sentence_bert_config.json",
        ):
            hf_hub_download(hf_model_id, fname, local_dir=temp_dir)

        if is_xlm_r:
            task_config = xlm_converter.load_task_config(
                temp_dir, transformers_config
            )
            preprocessor_config = xlm_converter.load_preprocessor_config(
                temp_dir, transformers_config
            )
        else:
            task_config = bert_converter.load_task_config(
                temp_dir, transformers_config
            )
            preprocessor_config = bert_converter.load_preprocessor_config(
                temp_dir, transformers_config
            )
        print(f"Task config: {task_config}")
        print(f"Preprocessor config: {preprocessor_config}")

        if is_xlm_r:
            preprocessor = XLMRobertaTextEmbedderPreprocessor(
                tokenizer=tokenizer,
                **preprocessor_config,
            )
            embedder = XLMRobertaTextEmbedder(
                backbone=backbone,
                preprocessor=preprocessor,
                **task_config,
            )
        else:
            preprocessor = keras_hub.models.BertTextEmbedderPreprocessor(
                tokenizer=tokenizer,
                **preprocessor_config,
            )
            embedder = BertTextEmbedder(
                backbone=backbone,
                preprocessor=preprocessor,
                **task_config,
            )
        # Cap sequence_length to 512 for validation inference to avoid OOM on
        # large-context models (e.g. bge_m3 uses sequence_length=8192).
        _orig_seq_len = embedder.preprocessor.sequence_length
        if _orig_seq_len > 512:
            print(
                f"\n[validation] Temporarily capping sequence_length "
                f"{_orig_seq_len} → 512 to avoid OOM (restored before save)."
            )
            embedder.preprocessor.sequence_length = 512
        passed = validate_output(
            embedder,
            hf_model_id,
            use_mean_pool=use_mean_pool_flag,
            is_multilingual=is_multi,
            is_xlm_roberta=is_xlm_r,
        )
        embedder.preprocessor.sequence_length = _orig_seq_len
        if not passed:
            print("\n⚠️  Verification failed. Preset not saved.")
            return

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
