"""Numerically verify KerasHub ModernBERT against Hugging Face.

This script loads the official Hugging Face ModernBERT checkpoint and the
corresponding KerasHub preset, then compares embeddings, backbone hidden
states, masked-language-model logits, and top-k predictions.

Weight conversion is intentionally not implemented here. Hugging Face weight
porting is handled by `convert_modern_bert.py` through the standard
`from_preset("hf://...")` loading path.

To run:

    python tools/checkpoint_conversion/convert_modern_bert_checkpoints.py \
        --preset modernbert_base_en
"""

import argparse
import os
import sys

# This must run before `keras_hub` is imported, otherwise running the
# script directly (`python tools/checkpoint_conversion/...`) picks up an
# installed `keras_hub` instead of this checkout.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForMaskedLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from keras_hub.src.models.modernbert.modern_bert_masked_lm import (  # noqa: E402
    ModernBertMaskedLM,
)

PRESET_MAP = {
    "modernbert_base_en": "answerdotai/ModernBERT-base",
    "modernbert_large_en": "answerdotai/ModernBERT-large",
}

# Tolerances for float32 CPU comparisons against the PyTorch reference.
# These are deliberately well above the observed values so the checks are
# not flaky, while still being tight enough to catch a real regression.
EMBEDDING_ATOL = 1e-5
BACKBONE_ATOL = 5e-2
LOGITS_ATOL = 1e-3


def get_huggingface_model_and_tokenizer(hf_repo):
    """Load the Hugging Face model and tokenizer."""
    print(f"Loading Hugging Face checkpoint: {hf_repo}")

    hf_model = AutoModelForMaskedLM.from_pretrained(hf_repo)
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_repo)

    hf_model.eval()

    return hf_model, hf_tokenizer


def get_keras_model(preset):
    """Load the KerasHub model through the standard HF preset path."""
    hf_repo = PRESET_MAP.get(preset, preset)

    print(f"Loading KerasHub model from: hf://{hf_repo}")

    keras_lm = ModernBertMaskedLM.from_preset(
        f"hf://{hf_repo}",
    )

    return keras_lm


def tokenize(hf_tokenizer, text):
    """Tokenize text for both Hugging Face and Keras."""
    hf_inputs = hf_tokenizer(
        text,
        return_tensors="pt",
    )

    hf_inputs.pop("token_type_ids", None)

    input_ids = hf_inputs["input_ids"].cpu().numpy().astype("int32")
    padding_mask = hf_inputs["attention_mask"].cpu().numpy().astype("int32")

    return hf_inputs, input_ids, padding_mask


def get_mask_positions(input_ids, mask_token_id):
    """Return masked-token positions in KerasHub format."""
    return np.argwhere(input_ids == mask_token_id).astype("int32")


def verify_embeddings(
    keras_lm,
    hf_model,
    hf_inputs,
):
    """Compare Hugging Face and KerasHub embedding outputs."""
    print("\nEmbedding verification")

    with torch.no_grad():
        hf_embedding = (
            hf_model.model.embeddings(
                hf_inputs["input_ids"],
            )
            .cpu()
            .numpy()
        )

    input_ids = hf_inputs["input_ids"].cpu().numpy().astype("int32")

    backbone = keras_lm.backbone

    keras_embedding = backbone.token_embedding(input_ids)

    if backbone.embedding_norm is not None:
        keras_embedding = backbone.embedding_norm(keras_embedding)

    keras_embedding = keras.ops.convert_to_numpy(
        keras_embedding,
    )

    diff = np.abs(
        hf_embedding.astype("float32") - keras_embedding.astype("float32")
    )

    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Embedding max diff : {max_diff:.6e}")
    print(f"Embedding mean diff: {mean_diff:.6e}")

    print(f"HF embedding mean    : {hf_embedding.mean():.6e}")
    print(f"Keras embedding mean : {keras_embedding.mean():.6e}")
    print(f"HF embedding std     : {hf_embedding.std():.6e}")
    print(f"Keras embedding std  : {keras_embedding.std():.6e}")

    np.testing.assert_allclose(
        hf_embedding,
        keras_embedding,
        atol=EMBEDDING_ATOL,
        rtol=EMBEDDING_ATOL,
    )

    print("✅ Embedding verification passed.")

    return max_diff


def verify_backbone(
    keras_lm,
    hf_model,
    hf_inputs,
    input_ids,
    padding_mask,
):
    """Compare Hugging Face and KerasHub backbone outputs."""
    print("\nBackbone verification")

    with torch.no_grad():
        hf_hidden = (
            hf_model.model(
                input_ids=hf_inputs["input_ids"],
                attention_mask=hf_inputs["attention_mask"],
            )
            .last_hidden_state.cpu()
            .numpy()
        )

    keras_hidden = keras_lm.backbone(
        {
            "token_ids": input_ids,
            "padding_mask": padding_mask,
        },
        training=False,
    )

    keras_hidden = keras.ops.convert_to_numpy(
        keras_hidden,
    )

    diff = np.abs(hf_hidden.astype("float32") - keras_hidden.astype("float32"))

    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Backbone max diff : {max_diff:.6e}")
    print(f"Backbone mean diff: {mean_diff:.6e}")

    print(f"HF hidden mean    : {hf_hidden.mean():.6e}")
    print(f"Keras hidden mean : {keras_hidden.mean():.6e}")
    print(f"HF hidden std     : {hf_hidden.std():.6e}")
    print(f"Keras hidden std  : {keras_hidden.std():.6e}")

    if max_diff > BACKBONE_ATOL:
        raise ValueError(
            f"Backbone max diff {max_diff:.6e} exceeds tolerance "
            f"{BACKBONE_ATOL:.6e}. Mean diff was {mean_diff:.6e}."
        )

    print("✅ Backbone verification passed.")

    return max_diff


def verify_masked_lm(
    keras_lm,
    hf_model,
    hf_tokenizer,
    hf_inputs,
    input_ids,
    padding_mask,
):
    """Compare Hugging Face and Keras ModernBERT MLM logits."""

    print("\nMaskedLM verification")

    mask_positions = get_mask_positions(
        input_ids,
        hf_tokenizer.mask_token_id,
    )

    mask_positions = np.asarray(
        mask_positions,
        dtype="int32",
    )

    if mask_positions.size == 0:
        print("No [MASK] tokens found; skipping MLM verification.")
        return None, 0, 0, 0

    mask_positions = mask_positions.reshape(-1, 2)

    print(f"Mask positions: {mask_positions.tolist()}")

    # Run Hugging Face model.
    #
    # Keep everything as torch tensors until the HF logits have been
    # extracted at the mask positions.

    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=hf_inputs["input_ids"],
            attention_mask=hf_inputs["attention_mask"],
        )

        hf_logits = hf_outputs.logits

    hf_mask_logits = []

    with torch.no_grad():
        for batch_index, sequence_index in mask_positions:
            batch_index = int(batch_index)
            sequence_index = int(sequence_index)

            hf_mask_logits.append(
                hf_logits[
                    batch_index,
                    sequence_index,
                ]
            )

        hf_mask_logits = torch.stack(
            hf_mask_logits,
            dim=0,
        )

    hf_mask_logits = hf_mask_logits.detach().cpu().numpy().astype("float32")

    batch_size = input_ids.shape[0]

    batch_mask_positions = []

    for batch_index in range(batch_size):
        positions = mask_positions[
            mask_positions[:, 0] == batch_index,
            1,
        ]

        batch_mask_positions.append(positions)

    # ModernBertMaskedLM currently expects the same number of masks
    # for every example in the batch.
    num_masks_per_batch = [len(positions) for positions in batch_mask_positions]

    if len(set(num_masks_per_batch)) != 1:
        raise ValueError(
            "ModernBertMaskedLM verification requires the same "
            "number of mask positions for every batch element. "
            f"Got {num_masks_per_batch}. "
            f"Original mask positions: "
            f"{mask_positions.tolist()}"
        )

    batch_mask_positions = np.asarray(
        batch_mask_positions,
        dtype="int32",
    )

    print(f"Keras mask_positions shape: {batch_mask_positions.shape}")
    print(f"Keras mask_positions: {batch_mask_positions.tolist()}")

    keras_logits = keras_lm(
        {
            "token_ids": input_ids,
            "padding_mask": padding_mask,
            "mask_positions": batch_mask_positions,
        },
        training=False,
    )

    if isinstance(keras_logits, dict):
        keras_logits = keras_logits["logits"]

    keras_logits = keras.ops.convert_to_numpy(keras_logits)

    keras_logits = np.asarray(
        keras_logits,
        dtype="float32",
    )

    print(f"Raw Keras logits shape: {keras_logits.shape}")

    if keras_logits.ndim == 3:
        keras_logits = keras_logits.reshape(
            -1,
            keras_logits.shape[-1],
        )

    elif keras_logits.ndim == 2:
        # (num_masks, vocab_size)
        pass

    else:
        raise ValueError(
            f"Unexpected Keras MLM output shape: {keras_logits.shape}"
        )

    # Verify shape.
    print(f"HF masked logits shape    : {hf_mask_logits.shape}")

    print(f"Keras masked logits shape : {keras_logits.shape}")

    if hf_mask_logits.shape != keras_logits.shape:
        raise ValueError(
            "HF and Keras masked-logit shapes differ: "
            f"HF={hf_mask_logits.shape}, "
            f"Keras={keras_logits.shape}. "
            f"Original mask positions="
            f"{mask_positions.tolist()}, "
            f"Keras mask positions="
            f"{batch_mask_positions.tolist()}"
        )

    # Numerical comparison.
    diff = np.abs(hf_mask_logits - keras_logits)

    max_diff = float(np.max(diff))

    mean_diff = float(np.mean(diff))

    print(f"MaskedLM max diff : {max_diff:.6e}\n")

    print(f"MaskedLM mean diff: {mean_diff:.6e}\n")

    print(f"HF logits mean    : {hf_mask_logits.mean():.6e}")

    print(f"Keras logits mean : {keras_logits.mean():.6e}")

    print(f"HF logits std     : {hf_mask_logits.std():.6e}")

    print(f"Keras logits std  : {keras_logits.std():.6e}")

    if max_diff > LOGITS_ATOL:
        raise ValueError(
            f"MaskedLM logits max diff {max_diff:.6e} exceeds tolerance "
            f"{LOGITS_ATOL:.6e}. Mean diff was {mean_diff:.6e}."
        )

    # Top-1 / Top-5 verification.
    top1_matches = 0
    top5_matches = 0

    for i in range(hf_mask_logits.shape[0]):
        hf_top1 = int(np.argmax(hf_mask_logits[i]))

        keras_top1 = int(np.argmax(keras_logits[i]))

        hf_top5 = np.argsort(hf_mask_logits[i])[-5:]

        keras_top5 = np.argsort(keras_logits[i])[-5:]

        top1_match = hf_top1 == keras_top1

        top5_match = np.array_equal(
            np.sort(hf_top5),
            np.sort(keras_top5),
        )

        top1_matches += int(top1_match)

        top5_matches += int(top5_match)

        hf_word = hf_tokenizer.decode(
            [hf_top1],
            clean_up_tokenization_spaces=False,
        )

        keras_word = hf_tokenizer.decode(
            [keras_top1],
            clean_up_tokenization_spaces=False,
        )
        print(
            f"  HF: '{hf_word}' | Keras: '{keras_word}' | "
            f"top-1 match: {top1_match} | top-5 match: {top5_match}"
        )

    return (
        max_diff,
        mean_diff,
        top1_matches,
        top5_matches,
    )


def verify_tokenizer(keras_lm, hf_tokenizer):
    """Compare `ModernBertTokenizer` against HF's `AutoTokenizer`."""
    print("\nTokenizer verification")

    preprocessor = keras_lm.preprocessor

    if preprocessor is None or preprocessor.tokenizer is None:
        print("No KerasHub tokenizer attached; skipping.")
        return

    keras_tokenizer = preprocessor.tokenizer

    for name in ("cls_token_id", "sep_token_id", "pad_token_id"):
        keras_id = getattr(keras_tokenizer, name)
        hf_id = getattr(hf_tokenizer, name)
        if keras_id != hf_id:
            raise ValueError(
                f"Special token mismatch for {name}: "
                f"KerasHub={keras_id}, Hugging Face={hf_id}."
            )

    if keras_tokenizer.mask_token_id != hf_tokenizer.mask_token_id:
        raise ValueError(
            "Special token mismatch for mask_token_id: "
            f"KerasHub={keras_tokenizer.mask_token_id}, "
            f"Hugging Face={hf_tokenizer.mask_token_id}."
        )

    print("Special token ids match Hugging Face.")

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "ModernBERT uses local-global alternating attention.",
        "Numbers like 1969 and punctuation -- all of it.",
    ]

    for text in texts:
        hf_ids = hf_tokenizer(text)["input_ids"]

        # HF wraps the sequence in [CLS] ... [SEP]; the KerasHub tokenizer
        # emits raw BPE ids and leaves the boundary tokens to the
        # preprocessor, so strip them before comparing.
        hf_body_ids = [
            token_id
            for token_id in hf_ids
            if token_id
            not in (
                hf_tokenizer.cls_token_id,
                hf_tokenizer.sep_token_id,
            )
        ]

        keras_ids = [int(i) for i in keras_tokenizer([text])[0]]

        if keras_ids != hf_body_ids:
            raise ValueError(
                f"Tokenizer mismatch for {text!r}: "
                f"KerasHub={keras_ids}, Hugging Face={hf_body_ids}."
            )

    print(f"✅ Tokenizer matches Hugging Face on {len(texts)} strings.")


def verify_padded_batch(keras_lm, hf_model, hf_tokenizer, texts):
    """Compare backbone outputs for a right-padded batch.

    The single-sequence cases all produce an all-ones attention mask, so
    the padding path is otherwise never exercised.
    """
    print("\nPadded batch verification")

    hf_inputs = hf_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    )

    hf_inputs.pop("token_type_ids", None)

    input_ids = hf_inputs["input_ids"].cpu().numpy().astype("int32")
    padding_mask = hf_inputs["attention_mask"].cpu().numpy().astype("int32")

    with torch.no_grad():
        hf_hidden = (
            hf_model.model(
                input_ids=hf_inputs["input_ids"],
                attention_mask=hf_inputs["attention_mask"],
            )
            .last_hidden_state.cpu()
            .numpy()
        )

    keras_hidden = keras_lm.backbone(
        {
            "token_ids": input_ids,
            "padding_mask": padding_mask,
        },
        training=False,
    )

    keras_hidden = keras.ops.convert_to_numpy(keras_hidden)

    if np.isnan(keras_hidden).any():
        raise ValueError("KerasHub produced NaNs on a padded batch.")

    # Padded positions are meaningless in both frameworks, so only compare
    # real tokens.
    valid = padding_mask[..., None].astype("float32")

    diff = (
        np.abs(hf_hidden.astype("float32") - keras_hidden.astype("float32"))
        * valid
    )

    max_diff = float(np.max(diff))

    print(f"Batch shape: {input_ids.shape}")

    for index in range(input_ids.shape[0]):
        real_tokens = int(padding_mask[index].sum())
        pad_tokens = int(input_ids.shape[1] - real_tokens)
        print(
            f"  row {index}: real={real_tokens} pad={pad_tokens} "
            f"max diff={float(np.max(diff[index])):.6e}"
        )

    if max_diff > BACKBONE_ATOL:
        raise ValueError(
            f"Padded batch backbone diff {max_diff:.6e} exceeds "
            f"tolerance {BACKBONE_ATOL:.6e}."
        )

    print(f"✅ Padded batch verification passed (max diff {max_diff:.6e}).")

    return max_diff


def verify_text(keras_lm, hf_model, hf_tokenizer, text):
    """Run numerical verification for one input string."""

    hf_inputs, input_ids, padding_mask = tokenize(
        hf_tokenizer,
        text,
    )

    seq_len = int(input_ids.shape[1])

    # The sliding-window mask keeps tokens within `window // 2` of each
    # other, so it only removes edges once the sequence is longer than
    # `2 * radius + 1`. Below that, local layers == global layers and the
    # local attention path is effectively untested.
    window = getattr(keras_lm.backbone, "local_attention_window", None)

    if window is None:
        window_active = False
    else:
        window_active = seq_len > (2 * (window // 2) + 1)

    print(
        f"\nInput: seq_len={seq_len}, "
        f"local sliding-window attention active: {window_active}"
    )

    embedding_diff = verify_embeddings(
        keras_lm,
        hf_model,
        hf_inputs,
    )

    backbone_diff = verify_backbone(
        keras_lm,
        hf_model,
        hf_inputs,
        input_ids,
        padding_mask,
    )

    mlm_result = verify_masked_lm(
        keras_lm,
        hf_model,
        hf_tokenizer,
        hf_inputs,
        input_ids,
        padding_mask,
    )

    if mlm_result[0] is None:
        logits_diff = None
        logits_mean_diff = None
        top1_matches = 0
        top5_matches = 0
    else:
        (
            logits_diff,
            logits_mean_diff,
            top1_matches,
            top5_matches,
        ) = mlm_result

    return {
        "seq_len": seq_len,
        "window_active": window_active,
        "embedding_max_diff": embedding_diff,
        "backbone_max_diff": backbone_diff,
        "logits_max_diff": logits_diff,
        "logits_mean_diff": logits_mean_diff,
        "top1_matches": top1_matches,
        "top5_matches": top5_matches,
    }


def save_preset(keras_lm, preset_name):
    """Save the verified ModernBERT model as a KerasHub preset."""
    print(f"\nSaving to preset: ./{preset_name}")
    keras_lm.save_to_preset(preset_name)
    print(f"✅ Successfully saved and verified preset: ./{preset_name}\n")


def main(preset, skip_save=False):
    """Run numerical verification."""
    hf_repo = PRESET_MAP.get(preset, preset)

    hf_model, hf_tokenizer = get_huggingface_model_and_tokenizer(
        hf_repo,
    )

    keras_lm = get_keras_model(preset)

    # Verify the KerasHub tokenizer itself. The rest of this script feeds
    # Hugging Face token ids straight into Keras, so without this check
    # `ModernBertTokenizer` would never be exercised at all.
    verify_tokenizer(keras_lm, hf_tokenizer)

    # A long passage is required to make local (sliding-window) attention
    # do anything: the window radius is `local_attention // 2` (64 for the
    # released checkpoints), so anything shorter than ~65 tokens makes the
    # local layers behave exactly like global ones.
    long_context = (
        "The quick brown fox jumps over the lazy dog while the bird "
        "watches quietly from a nearby tree. " * 20
    )

    test_cases = [
        "The capital of France is [MASK].",
        "Hello, my name is [MASK] and I live in [MASK].",
        "The [MASK] barked loudly at the mailman.",
        "In 1969, humans first landed on the [MASK].",
        (
            "The quick brown fox jumps over the lazy dog while the " * 5
            + "[MASK] watches."
        ),
        # Exercises local sliding-window attention.
        long_context + "The [MASK] watches from the tree.",
    ]

    results = []

    for text in test_cases:
        results.append(
            verify_text(
                keras_lm,
                hf_model,
                hf_tokenizer,
                text,
            )
        )

    # Padding is never exercised by the single-sequence cases above, since
    # a lone string tokenizes to an all-ones attention mask.
    verify_padded_batch(
        keras_lm,
        hf_model,
        hf_tokenizer,
        [
            "The capital of France is Paris.",
            long_context,
        ],
    )

    print("\n")
    print("NUMERICAL VERIFICATION SUMMARY")

    valid_logits_results = [
        result for result in results if result["logits_max_diff"] is not None
    ]

    total_top1 = sum(result["top1_matches"] for result in valid_logits_results)

    total_top5 = sum(result["top5_matches"] for result in valid_logits_results)

    print(f"Test cases run: {len(test_cases)}")

    windowed = sum(1 for result in results if result["window_active"])
    print(
        f"Cases activating local sliding-window attention: "
        f"{windowed}/{len(test_cases)}"
    )

    max_embedding_diff = max(result["embedding_max_diff"] for result in results)
    max_backbone_diff = max(result["backbone_max_diff"] for result in results)
    max_mlm_logits_diff = max(
        result["logits_max_diff"] for result in valid_logits_results
    )

    print(f"✅ Max embedding diff: {max_embedding_diff:.6e}")
    print(f"✅ Max backbone diff: {max_backbone_diff:.6e}")
    print(f"✅ Max MLM logits diff: {max_mlm_logits_diff:.6e}")

    # Count masks directly from the tokenizer inputs.
    total_masks = 0
    for text in test_cases:
        _, input_ids, _ = tokenize(
            hf_tokenizer,
            text,
        )
        total_masks += np.sum(input_ids == hf_tokenizer.mask_token_id)

    print(f"✅ Top-1 prediction matches: {total_top1}/{total_masks}")

    print(f"✅ Top-5 prediction matches: {total_top5}/{total_masks}")

    assert windowed > 0, (
        "No test case was long enough to activate local sliding-window "
        "attention, so the local attention path went unverified."
    )

    assert total_top1 == total_masks, (
        "At least one top-1 prediction differs from Hugging Face."
    )

    assert total_top5 == total_masks, (
        "At least one top-5 prediction differs from Hugging Face."
    )

    if not skip_save:
        save_preset(keras_lm, preset)

    print("✅ All numerical verification checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preset",
        type=str,
        default="modernbert_base_en",
    )

    parser.add_argument(
        "--skip_save",
        action="store_true",
        help=(
            "Skip writing the converted preset to ./<preset>. The saved "
            "preset is large and lands in the repo root, which dirties "
            "`git status` and trips the api_gen pre-commit hook."
        ),
    )

    args = parser.parse_args()

    main(args.preset, skip_save=args.skip_save)
