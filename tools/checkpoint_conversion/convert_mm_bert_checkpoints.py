"""Convert the mmBERT Hugging Face checkpoint into a KerasHub preset.

mmBERT only ships `pytorch_model.bin` weights, while the KerasHub preset
loader only reads `safetensors`. This script downloads the original
checkpoint, rewrites its weights as `safetensors` in a temporary directory,
ports them into the `keras_hub.models.MMBert*` classes with the shared
ModernBERT converter, verifies the port numerically against Hugging Face, and
saves the result as a Keras preset that can be loaded with
`keras_hub.models.MMBertMaskedLM.from_preset(...)`.

To run (from a checkout installed with `pip install -e .`):

    python tools/checkpoint_conversion/convert_mm_bert_checkpoints.py \
        --preset mm_bert_base_multi
"""

import argparse
import os
import shutil
import tempfile

import keras
import numpy as np
import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

from keras_hub.src.models.mm_bert.mm_bert_backbone import MMBertBackbone
from keras_hub.src.models.mm_bert.mm_bert_masked_lm import MMBertMaskedLM
from keras_hub.src.models.mm_bert.mm_bert_masked_lm_preprocessor import (
    MMBertMaskedLMPreprocessor,
)
from keras_hub.src.models.mm_bert.mm_bert_tokenizer import MMBertTokenizer
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers import convert_modern_bert
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

PRESET_MAP = {
    "mm_bert_base_multi": "jhu-clsp/mmBERT-base",
    "mm_bert_small_multi": "jhu-clsp/mmBERT-small",
}

PYTORCH_WEIGHTS_FILE = "pytorch_model.bin"
SAFETENSOR_WEIGHTS_FILE = "model.safetensors"

# The checkpoint files that the Keras preset is built from.
CHECKPOINT_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

# Tolerances for float32 CPU comparisons against the PyTorch reference.
EMBEDDING_ATOL = 1e-5
BACKBONE_ATOL = 5e-2
LOGITS_ATOL = 1e-3

# `MaskedLMPreprocessor` needs a sequence length. mmBERT handles much longer
# sequences, and a user can always pass another value when loading the preset.
SEQUENCE_LENGTH = 512

TOKENIZER_TEXTS = [
    "The quick brown fox jumped over the lazy dog.",
    "  leading and trailing spaces  ",
    "a\nb\tc",
    "<bos>hello<eos>",
    "Mixed<mask>inline",
    "The capital of <mask> is Paris.",
    "Das ist ein Test. Ça va? これはテストです。",
    "你好，世界！",
    " ᐊᖏᔪᖅ",
    "café naïve résumé",
]


def download_checkpoint(hf_repo, checkpoint_dir):
    """Download the checkpoint files into a local Hugging Face directory."""
    from huggingface_hub import hf_hub_download

    print(f"Downloading the Hugging Face checkpoint: {hf_repo}")

    for filename in CHECKPOINT_FILES + [PYTORCH_WEIGHTS_FILE]:
        hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            local_dir=checkpoint_dir,
        )


def convert_pytorch_weights(checkpoint_dir, remove_original=True):
    """Rewrite `pytorch_model.bin` as `model.safetensors`."""
    import safetensors.torch

    bin_path = os.path.join(checkpoint_dir, PYTORCH_WEIGHTS_FILE)
    safetensor_path = os.path.join(checkpoint_dir, SAFETENSOR_WEIGHTS_FILE)

    print(f"Rewriting {PYTORCH_WEIGHTS_FILE} as {SAFETENSOR_WEIGHTS_FILE}")

    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    # Training checkpoints wrap the weights in a `state_dict`/`model` key.
    for key in ("state_dict", "model"):
        inner = state_dict.get(key)
        if isinstance(inner, dict):
            state_dict = inner
            break

    # `safetensors` only stores tensors, and requires them to be contiguous.
    state_dict = {
        key: value.contiguous()
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }

    safetensors.torch.save_file(state_dict, safetensor_path)

    # Drop the original weights, they are 1.2GB and no longer needed. A
    # caller provided checkpoint directory is left untouched.
    if remove_original:
        os.remove(bin_path)


def build_keras_model(checkpoint_dir):
    """Port the checkpoint into the KerasHub `MMBert*` classes."""
    config = load_json(checkpoint_dir, "config.json")
    backbone = MMBertBackbone(
        **convert_modern_bert.convert_backbone_config(config)
    )
    tokenizer = convert_modern_bert.convert_tokenizer(
        MMBertTokenizer,
        checkpoint_dir,
    )
    preprocessor = MMBertMaskedLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH,
    )
    keras_lm = MMBertMaskedLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )

    # The weights to port are the ones written by
    # `convert_pytorch_weights()`, so load them through the `safetensors`
    # loader that the ModernBERT converter is written against.
    with SafetensorLoader(checkpoint_dir, prefix="") as loader:
        convert_modern_bert.convert_weights(backbone, loader, config)
        convert_modern_bert.convert_head(keras_lm, loader, config)

    return keras_lm


def tokenize(hf_tokenizer, text, keras_tokenizer=None):
    """Tokenize text independently for Hugging Face and KerasHub."""
    hf_inputs = hf_tokenizer(text, return_tensors="pt")
    hf_inputs.pop("token_type_ids", None)
    hf_input_ids = hf_inputs["input_ids"].cpu().numpy().astype("int32")

    if keras_tokenizer is None:
        input_ids = hf_input_ids
        padding_mask = hf_inputs["attention_mask"].cpu().numpy().astype("int32")
        return hf_inputs, input_ids, padding_mask

    # KerasHub tokenizers do not add the `<bos>`/`<eos>` pair that the
    # checkpoint's `post_processor` adds, so add it here.
    keras_body_ids = [int(i) for i in keras_tokenizer([text])[0]]
    keras_seq = (
        [int(keras_tokenizer.bos_token_id)]
        + keras_body_ids
        + [int(keras_tokenizer.eos_token_id)]
    )
    input_ids = np.asarray([keras_seq], dtype="int32")
    padding_mask = np.ones_like(input_ids, dtype="int32")

    if not np.array_equal(input_ids, hf_input_ids):
        raise ValueError(
            f"Token IDs diverged between MMBertTokenizer and the Hugging Face "
            f"tokenizer for {text!r}: KerasHub={input_ids.tolist()}, "
            f"HF={hf_input_ids.tolist()}."
        )

    return hf_inputs, input_ids, padding_mask


def get_mask_positions(input_ids, mask_token_id):
    """Return masked-token positions in KerasHub format."""
    return np.argwhere(input_ids == mask_token_id).astype("int32")


def verify_tokenizer(keras_tokenizer, hf_tokenizer):
    """Compare `MMBertTokenizer` against the Hugging Face tokenizer."""
    print("\nTokenizer verification")

    for text in TOKENIZER_TEXTS:
        hf_ids = hf_tokenizer(text, add_special_tokens=False)["input_ids"]
        keras_ids = [
            int(i)
            for i in np.reshape(np.asarray(keras_tokenizer([text])[0]), -1)
        ]
        if hf_ids != keras_ids:
            raise ValueError(
                f"Token IDs diverged between MMBertTokenizer and the Hugging "
                f"Face tokenizer for {text!r}: KerasHub={keras_ids}, "
                f"HF={hf_ids}."
            )

    print(
        f"✅ Tokenizer matches Hugging Face on {len(TOKENIZER_TEXTS)} strings."
    )

    return len(TOKENIZER_TEXTS)


def verify_embeddings(keras_lm, hf_model, hf_inputs):
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

    keras_embedding = keras.ops.convert_to_numpy(keras_embedding)

    max_diff = np.max(
        np.abs(
            hf_embedding.astype("float32") - keras_embedding.astype("float32")
        )
    )

    print(f"Embedding max diff : {max_diff:.6e}")

    np.testing.assert_allclose(
        hf_embedding,
        keras_embedding,
        atol=EMBEDDING_ATOL,
        rtol=EMBEDDING_ATOL,
    )

    print("✅ Embedding verification passed.")

    return max_diff


def verify_backbone(keras_lm, hf_model, hf_inputs, input_ids, padding_mask):
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

    keras_hidden = keras.ops.convert_to_numpy(
        keras_lm.backbone(
            {
                "token_ids": input_ids,
                "padding_mask": padding_mask,
            },
            training=False,
        )
    )

    diff = np.abs(hf_hidden.astype("float32") - keras_hidden.astype("float32"))
    max_diff = float(np.max(diff))

    print(f"Backbone max diff : {max_diff:.6e}")
    print(f"Backbone mean diff: {np.mean(diff):.6e}")

    if max_diff > BACKBONE_ATOL:
        raise ValueError(
            f"Backbone max diff {max_diff:.6e} exceeds tolerance "
            f"{BACKBONE_ATOL:.6e}."
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
    """Compare Hugging Face and KerasHub masked language model logits."""
    print("\nMaskedLM verification")

    mask_positions = get_mask_positions(input_ids, hf_tokenizer.mask_token_id)

    if mask_positions.size == 0:
        print("No <mask> tokens found; skipping MLM verification.")
        return None, 0, 0

    mask_positions = mask_positions.reshape(-1, 2)
    print(f"Mask positions: {mask_positions.tolist()}")

    with torch.no_grad():
        hf_logits = hf_model(
            input_ids=hf_inputs["input_ids"],
            attention_mask=hf_inputs["attention_mask"],
        ).logits

    hf_mask_logits = np.asarray(
        [
            hf_logits[int(batch_index), int(seq_index)].cpu().numpy()
            for batch_index, seq_index in mask_positions
        ]
    )

    keras_logits = keras.ops.convert_to_numpy(
        keras_lm(
            {
                "token_ids": input_ids,
                "padding_mask": padding_mask,
                "mask_positions": mask_positions,
            },
            training=False,
        )
    )

    diff = np.abs(
        hf_mask_logits.astype("float32") - keras_logits.astype("float32")
    )
    max_diff = float(np.max(diff))

    print(f"MLM logits max diff : {max_diff:.6e}")
    print(f"MLM logits mean diff: {np.mean(diff):.6e}")

    if max_diff > LOGITS_ATOL:
        raise ValueError(
            f"MLM logits max diff {max_diff:.6e} exceeds tolerance "
            f"{LOGITS_ATOL:.6e}."
        )

    top1_matches = int(
        np.sum(
            np.argmax(hf_mask_logits, axis=-1)
            == np.argmax(keras_logits, axis=-1)
        )
    )
    hf_top5 = np.argsort(hf_mask_logits, axis=-1)[:, -5:]
    keras_top5 = np.argsort(keras_logits, axis=-1)[:, -5:]
    top5_matches = int(
        np.sum([len(set(a) & set(b)) for a, b in zip(hf_top5, keras_top5)])
    )

    print(f"Top-1 matches: {top1_matches}/{len(mask_positions)}")
    print(f"Top-5 matches: {top5_matches}/{len(mask_positions)}")
    print("✅ MaskedLM verification passed.")

    return max_diff, top1_matches, top5_matches


def verify_padded_batch(keras_lm, hf_model, hf_tokenizer, texts):
    """Compare the backbone on a right-padded batch."""
    print("\nPadded batch verification")

    hf_inputs = hf_tokenizer(texts, return_tensors="pt", padding=True)
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

    keras_hidden = keras.ops.convert_to_numpy(
        keras_lm.backbone(
            {"token_ids": input_ids, "padding_mask": padding_mask},
            training=False,
        )
    )

    valid = padding_mask.astype(bool)
    diff = np.abs(
        hf_hidden.astype("float32")[valid]
        - keras_hidden.astype("float32")[valid]
    )
    max_diff = float(np.max(diff))

    print(f"Padded batch max diff: {max_diff:.6e}")

    if max_diff > BACKBONE_ATOL:
        raise ValueError(
            f"Padded batch max diff {max_diff:.6e} exceeds tolerance "
            f"{BACKBONE_ATOL:.6e}."
        )

    print("✅ Padded batch verification passed.")

    return max_diff


def save_preset(keras_lm, preset_name):
    """Save the verified mmBERT model as a KerasHub preset."""
    print(f"\nSaving to preset: ./{preset_name}")
    keras_lm.save_to_preset(preset_name)
    print(f"✅ Successfully saved and verified preset: ./{preset_name}\n")


def main(
    preset,
    skip_save=False,
    keep_checkpoint=False,
    checkpoint_dir=None,
):
    """Convert, verify and save the mmBERT checkpoint."""
    hf_repo = PRESET_MAP.get(preset, preset)

    owns_checkpoint = checkpoint_dir is None

    if owns_checkpoint:
        checkpoint_dir = tempfile.mkdtemp(prefix="mm_bert_checkpoint_")
    else:
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        if not os.path.isdir(checkpoint_dir):
            raise ValueError(f"`{checkpoint_dir}` is not a directory.")

    try:
        download_checkpoint(hf_repo, checkpoint_dir)
        convert_pytorch_weights(
            checkpoint_dir,
            remove_original=owns_checkpoint,
        )

        keras_lm = build_keras_model(checkpoint_dir)
        tokenizer = keras_lm.preprocessor.tokenizer

        hf_model = AutoModelForMaskedLM.from_pretrained(checkpoint_dir)
        hf_model.eval()
        hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        verify_tokenizer(tokenizer, hf_tokenizer)

        # A long passage is required to make local (sliding-window) attention
        # do anything: the window radius is `local_attention // 2` (64 for the
        # released checkpoints), so anything shorter than ~65 tokens makes the
        # local layers behave exactly like global ones.
        long_context = (
            "The quick brown fox jumps over the lazy dog while the bird "
            "watches quietly from a nearby tree. " * 20
        )

        test_cases = [
            "The capital of France is <mask>.",
            "Hello, my name is <mask> and I live in <mask>.",
            "The <mask> barked loudly at the mailman.",
            "In 1969, humans first landed on the <mask>.",
            "Das ist ein <mask>. Ça va? これは<mask>です。",
            # Exercises local sliding-window attention.
            long_context + "The <mask> watches from the tree.",
        ]

        results = []

        for text in test_cases:
            print(f"\nText: {text[:70]!r}")
            hf_inputs, input_ids, padding_mask = tokenize(
                hf_tokenizer,
                text,
                tokenizer,
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
            logits_diff, top1, top5 = verify_masked_lm(
                keras_lm,
                hf_model,
                hf_tokenizer,
                hf_inputs,
                input_ids,
                padding_mask,
            )
            results.append(
                {
                    "embedding": embedding_diff,
                    "backbone": backbone_diff,
                    "logits": logits_diff,
                    "top1": top1,
                    "top5": top5,
                }
            )

        verify_padded_batch(
            keras_lm,
            hf_model,
            hf_tokenizer,
            [
                "The capital of France is Paris.",
                long_context,
            ],
        )

        valid = [result for result in results if result["logits"] is not None]
        total_masks = sum(result["top1"] for result in valid)

        print("\nNUMERICAL VERIFICATION SUMMARY")
        print(f"Test cases run: {len(test_cases)}")
        print(
            "✅ Max embedding diff: "
            f"{max(result['embedding'] for result in results):.6e}"
        )
        print(
            "✅ Max backbone diff: "
            f"{max(result['backbone'] for result in results):.6e}"
        )
        print(
            "✅ Max MLM logits diff: "
            f"{max(result['logits'] for result in valid):.6e}"
        )
        print(
            f"✅ Top-1 prediction matches: "
            f"{sum(result['top1'] for result in valid)}/{total_masks}"
        )
        print(
            f"✅ Top-5 prediction matches: "
            f"{sum(result['top5'] for result in valid)}/{total_masks}"
        )

        if not skip_save:
            save_preset(keras_lm, preset)

        print("✅ All numerical verification checks passed.")
    finally:
        if keep_checkpoint or not owns_checkpoint:
            print(f"\nKeeping the converted checkpoint at {checkpoint_dir}")
        else:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preset",
        type=str,
        default="mm_bert_base_multi",
        help="The preset name to convert and verify.",
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

    parser.add_argument(
        "--keep_checkpoint",
        action="store_true",
        help="Keep the temporary safetensors checkpoint for debugging.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Reuse the checkpoint files in this directory instead of "
            "downloading them into a temporary directory."
        ),
    )

    args = parser.parse_args()

    main(
        args.preset,
        skip_save=args.skip_save,
        keep_checkpoint=args.keep_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
    )
