"""Convert Hugging Face sentence-transformer MPNet checkpoints to KerasHub.

This script converts an MPNet sentence-transformer model (e.g.
`sentence-transformers/all-mpnet-base-v2`) to a KerasHub `MPNetTextEmbedder`,
verifies numerical parity against the reference `SentenceTransformer`
implementation, and saves the result as a KerasHub preset.

Setup (from the root of a keras-hub checkout):
```shell
pip install -e .
pip install sentence-transformers safetensors huggingface_hub
```

Usage:
```shell
python tools/checkpoint_conversion/convert_mpnet_checkpoints.py \
    --preset all_mpnet_base_v2_en
```

The Keras backend defaults to torch and can be set with `KERAS_BACKEND`
(e.g. `KERAS_BACKEND=jax`). The KerasHub model runs on a GPU when one is
available, while the reference model always runs on the CPU.
"""

import json
import os
import tempfile

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from keras import ops
from sentence_transformers import SentenceTransformer

from keras_hub.src.models.mpnet.mpnet_backbone import MPNetBackbone
from keras_hub.src.models.mpnet.mpnet_text_embedder import MPNetTextEmbedder
from keras_hub.src.models.mpnet.mpnet_text_embedder_preprocessor import (
    MPNetTextEmbedderPreprocessor,
)
from keras_hub.src.models.mpnet.mpnet_tokenizer import MPNetTokenizer
from keras_hub.src.utils.transformers import convert_mpnet
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

FLAGS = flags.FLAGS

PRESET_MAP = {
    "all_mpnet_base_v2_en": "sentence-transformers/all-mpnet-base-v2",
}

flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
)

HF_FILES = [
    "config.json",
    "model.safetensors",
    "vocab.txt",
    "tokenizer_config.json",
    "modules.json",
    "1_Pooling/config.json",
    "sentence_bert_config.json",
]

# A mix of regular sentences and tokenizer edge cases (casing, accents,
# punctuation, URLs, CJK, emoji, whitespace, empty input, and a text longer
# than the maximum sequence length).
TEST_TEXTS = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
    "Café naïve résumé, déjà vu!",
    "HELLO World, MiXeD CaSe.",
    "Prices rose 3.5% to $1,299.99 at https://example.com/a?x=1&y=2.",
    "中文分词测试和日本語のテキスト。",
    "Emoji test 😀 🚀 done.",
    "  multiple   spaces\tand\nnewlines  ",
    "",
    " ".join(["The quick brown fox jumps over the lazy dog."] * 60),
]

# The maximum absolute difference allowed between L2-normalized embeddings.
EMBEDDING_TOLERANCE = 1e-5
# The maximum absolute difference allowed between the final hidden states.
HIDDEN_STATE_TOLERANCE = 1e-4
# The maximum absolute difference allowed between the embeddings of the
# converted model and of the reloaded preset.
RELOAD_TOLERANCE = 1e-6


def convert_model(preset_dir):
    """Build a KerasHub `MPNetTextEmbedder` from downloaded HF files."""
    with open(os.path.join(preset_dir, "config.json")) as f:
        transformers_config = json.load(f)

    backbone = MPNetBackbone(
        **convert_mpnet.convert_backbone_config(transformers_config),
        dtype="float32",
    )
    with SafetensorLoader(preset_dir) as loader:
        convert_mpnet.convert_weights(backbone, loader, transformers_config)

    tokenizer = convert_mpnet.convert_tokenizer(MPNetTokenizer, preset_dir)
    preprocessor = MPNetTextEmbedderPreprocessor(
        tokenizer=tokenizer,
        **convert_mpnet.load_preprocessor_config(
            preset_dir, transformers_config
        ),
    )
    return MPNetTextEmbedder(
        backbone=backbone,
        preprocessor=preprocessor,
        **convert_mpnet.load_task_config(preset_dir, transformers_config),
    )


def check_parameter_count(keras_backbone, hf_model):
    """Reconcile the HF and KerasHub parameter counts.

    KerasHub intentionally does not port two groups of HF parameters, the same
    as `convert_bge_multilingual_e5_checkpoints.py` does for XLM-RoBERTa:
    - The pooler (`dense` + `tanh` on the first token). Sentence-transformers
      never uses it.
    - The first 2 rows of the position embedding table. They are reserved by
      `padding_idx=1`, and real positions start at row 2.
    """
    hf_params = sum(p.numel() for p in hf_model.parameters())
    pooler_params = 0
    if getattr(hf_model, "pooler", None) is not None:
        pooler_params = sum(p.numel() for p in hf_model.pooler.parameters())
    reserved_position_params = 2 * hf_model.config.hidden_size
    expected_params = hf_params - pooler_params - reserved_position_params
    keras_params = keras_backbone.count_params()
    print(f"HF params:                      {hf_params:>12,}")
    print(f"  - pooler (unused):            {-pooler_params:>12,}")
    print(f"  - 2 reserved position rows:   {-reserved_position_params:>12,}")
    print(f"Expected KerasHub params:       {expected_params:>12,}")
    print(f"KerasHub params:                {keras_params:>12,}")
    return keras_params == expected_params


def check_tokenizer(preprocessor, hf_tokenizer):
    """Compare token ids, including special tokens and truncation."""
    hf_ids = hf_tokenizer(
        TEST_TEXTS,
        truncation=True,
        max_length=preprocessor.sequence_length,
    )["input_ids"]
    keras_inputs = preprocessor(TEST_TEXTS)
    token_ids = ops.convert_to_numpy(keras_inputs["token_ids"])
    padding_mask = ops.convert_to_numpy(keras_inputs["padding_mask"])
    passed = True
    for i, text in enumerate(TEST_TEXTS):
        keras_ids = token_ids[i][padding_mask[i].astype(bool)].tolist()
        if keras_ids != hf_ids[i]:
            passed = False
            print(f"Token id mismatch for {text[:40]!r}:")
            print(f"  KerasHub: {keras_ids[:24]}")
            print(f"  HF:       {hf_ids[i][:24]}")
    print(f"Compared {len(TEST_TEXTS)} texts.")
    return passed


def check_backbone(keras_backbone, preprocessor, hf_model):
    """Compare final hidden states on non-padding positions."""
    keras_inputs = preprocessor(TEST_TEXTS)
    token_ids = ops.convert_to_numpy(keras_inputs["token_ids"])
    padding_mask = ops.convert_to_numpy(keras_inputs["padding_mask"])
    padding_mask = padding_mask.astype("int32")
    keras_hidden = ops.convert_to_numpy(
        keras_backbone({"token_ids": token_ids, "padding_mask": padding_mask})
    )
    with torch.no_grad():
        hf_hidden = hf_model(
            input_ids=torch.from_numpy(token_ids.astype("int64")),
            attention_mask=torch.from_numpy(padding_mask.astype("int64")),
        ).last_hidden_state.numpy()
    diff = np.abs(keras_hidden - hf_hidden)[padding_mask.astype(bool)]
    print(f"Max abs diff:  {diff.max():.2e}")
    print(f"Mean abs diff: {diff.mean():.2e}")
    return diff.max() <= HIDDEN_STATE_TOLERANCE


def check_embeddings(keras_model, hf_model):
    """Compare sentence embeddings."""
    with torch.no_grad():
        hf_embeddings = hf_model.encode(TEST_TEXTS, convert_to_numpy=True)
    keras_embeddings = np.asarray(keras_model.predict(TEST_TEXTS, verbose=0))
    diff = np.abs(hf_embeddings - keras_embeddings)
    print(f"Output shape:  {keras_embeddings.shape}")
    print(f"Max abs diff:  {diff.max():.2e}")
    print(f"Mean abs diff: {diff.mean():.2e}")
    cosine = np.sum(hf_embeddings * keras_embeddings, axis=-1) / (
        np.linalg.norm(hf_embeddings, axis=-1)
        * np.linalg.norm(keras_embeddings, axis=-1)
    )
    print(f"Min cosine similarity to reference: {cosine.min():.7f}")
    return diff.max() <= EMBEDDING_TOLERANCE


def check_semantic_search(keras_model, hf_model):
    """The best match must be the same for both implementations."""
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin.",
        "Mars is often referred to as the Red Planet.",
        "Jupiter has a prominent red spot.",
    ]
    keras_scores = ops.convert_to_numpy(
        keras_model.similarity(
            keras_model.encode_text(query, verbose=0),
            keras_model.encode_documents(documents, verbose=0),
        )
    )[0]
    with torch.no_grad():
        hf_query = hf_model.encode([query], convert_to_numpy=True)
        hf_documents = hf_model.encode(documents, convert_to_numpy=True)
    hf_scores = (hf_query @ hf_documents.T)[0]
    print(f"KerasHub scores: {np.round(keras_scores, 5)}")
    print(f"HF scores:       {np.round(hf_scores, 5)}")
    return np.argmax(keras_scores) == np.argmax(hf_scores)


def check_saved_preset(keras_model, preset):
    """Reload the saved preset and compare weights and outputs."""
    # Use a path, a bare name would resolve to the registered Kaggle preset.
    reloaded = MPNetTextEmbedder.from_preset(os.path.abspath(preset))
    weights_match = len(keras_model.weights) == len(reloaded.weights) and all(
        np.array_equal(ops.convert_to_numpy(a), ops.convert_to_numpy(b))
        for a, b in zip(keras_model.weights, reloaded.weights)
    )
    original = np.asarray(keras_model.predict(TEST_TEXTS, verbose=0))
    restored = np.asarray(reloaded.predict(TEST_TEXTS, verbose=0))
    max_diff = np.max(np.abs(original - restored))
    print(f"Weights identical after reload: {weights_match}")
    print(f"Max abs diff after reload: {max_diff:.2e}")
    # The weights must match exactly. GPU kernels are not guaranteed to be
    # bitwise reproducible across separately compiled models, so the outputs
    # get a tiny tolerance.
    return weights_match and max_diff <= RELOAD_TOLERANCE


def configure_devices():
    """Use full float32 matmuls on GPUs and keep TensorFlow on the CPU."""
    import tensorflow as tf

    # On Ampere and newer GPUs, TF32 matmuls would inflate the diffs against
    # the float32 reference to ~1e-3.
    torch.backends.cuda.matmul.allow_tf32 = False
    backend = keras.config.backend()
    if backend == "jax":
        import jax

        jax.config.update("jax_default_matmul_precision", "highest")
    if backend == "tensorflow":
        tf.config.experimental.enable_tensor_float_32_execution(False)
    else:
        # TensorFlow only runs the tokenizer's string ops. Keeping it off the
        # GPU stops it from reserving memory that torch or JAX needs.
        tf.config.set_visible_devices([], "GPU")


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {preset}. Must be one of "
            f"{','.join(PRESET_MAP.keys())}"
        )
    hf_model_id = PRESET_MAP[preset]
    backend = keras.config.backend()
    print(f"Converting {hf_model_id} -> {preset} ({backend} backend)")
    configure_devices()

    with tempfile.TemporaryDirectory() as temp_dir:
        for filename in HF_FILES:
            hf_hub_download(hf_model_id, filename, local_dir=temp_dir)
        keras_model = convert_model(temp_dir)
    device = keras_model.backbone.weights[0].value.device
    print(f"KerasHub model device: {device}")

    hf_model = SentenceTransformer(hf_model_id, device="cpu")
    hf_model.eval()
    hf_transformer = hf_model[0].auto_model

    checks = {
        "Parameter count": lambda: check_parameter_count(
            keras_model.backbone, hf_transformer
        ),
        "Tokenizer": lambda: check_tokenizer(
            keras_model.preprocessor, hf_model.tokenizer
        ),
        "Backbone hidden states": lambda: check_backbone(
            keras_model.backbone, keras_model.preprocessor, hf_transformer
        ),
        "Sentence embeddings": lambda: check_embeddings(keras_model, hf_model),
        "Semantic search": lambda: check_semantic_search(keras_model, hf_model),
    }
    results = {}
    for name, check in checks.items():
        print(f"\n--- {name} ---")
        results[name] = bool(check())

    print("\n--- Summary ---")
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if not all(results.values()):
        raise RuntimeError("Verification failed. The preset was not saved.")

    print("\n--- Saved preset ---")
    keras_model.save_to_preset(preset)
    if not check_saved_preset(keras_model, preset):
        raise RuntimeError("The saved preset does not reproduce the outputs.")
    print(f"Saved and verified preset: ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
