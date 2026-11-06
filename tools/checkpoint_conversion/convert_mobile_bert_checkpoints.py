"""Convert MobileBERT from Hugging Face to KerasHub.

Usage:
  python tools/checkpoint_conversion/convert_mobile_bert_checkpoints.py \
    --preset mobilebert_uncased \
    --output_dir /tmp/mobilebert \
    --hf_dir optional_local_dir
"""

import argparse
import os
import shutil
import tempfile

import keras
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import BertTokenizer
from transformers import MobileBertForPreTraining

from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)

try:
    from keras_hub.src.models.mobile_bert.mobile_bert_masked_lm import (
        MobileBertMaskedLM,
    )
except ImportError:
    MobileBertMaskedLM = None

PRESET_MAP = {"mobilebert_uncased": "google/mobilebert-uncased"}


def load_hf_model(hf_id, hf_dir=None, tmp_dir=None):
    if hf_dir:
        model = MobileBertForPreTraining.from_pretrained(hf_dir)
        tokenizer = BertTokenizer.from_pretrained(hf_dir)
        model.save_pretrained(tmp_dir, safe_serialization=True)
        tokenizer.save_pretrained(tmp_dir)

        if not os.path.exists(os.path.join(tmp_dir, "vocab.txt")):
            vocab_path = os.path.join(hf_dir, "vocab.txt")
            if os.path.exists(vocab_path):
                shutil.copy(vocab_path, os.path.join(tmp_dir, "vocab.txt"))

        if not os.path.exists(os.path.join(tmp_dir, "tokenizer_config.json")):
            tokenizer_config_path = os.path.join(
                hf_dir, "tokenizer_config.json"
            )
            if os.path.exists(tokenizer_config_path):
                shutil.copy(
                    tokenizer_config_path,
                    os.path.join(tmp_dir, "tokenizer_config.json"),
                )

        if not os.path.exists(os.path.join(tmp_dir, "vocab.txt")):
            raise FileNotFoundError(f"vocab.txt not found in {tmp_dir}")
        if not os.path.exists(os.path.join(tmp_dir, "tokenizer_config.json")):
            raise FileNotFoundError(
                f"tokenizer_config.json not found in {tmp_dir}"
            )

        return model, tokenizer
    else:
        model = MobileBertForPreTraining.from_pretrained(hf_id)
        tokenizer = BertTokenizer.from_pretrained(hf_id)
        model.save_pretrained(tmp_dir, safe_serialization=True)
        tokenizer.save_pretrained(tmp_dir)

        if not os.path.exists(os.path.join(tmp_dir, "vocab.txt")):
            vocab_path = hf_hub_download(hf_id, "vocab.txt")
            shutil.copy(vocab_path, os.path.join(tmp_dir, "vocab.txt"))

        if not os.path.exists(os.path.join(tmp_dir, "tokenizer_config.json")):
            tokenizer_config_path = hf_hub_download(
                hf_id, "tokenizer_config.json"
            )
            shutil.copy(
                tokenizer_config_path,
                os.path.join(tmp_dir, "tokenizer_config.json"),
            )

        if not os.path.exists(os.path.join(tmp_dir, "vocab.txt")):
            raise FileNotFoundError(f"vocab.txt not found in {tmp_dir}")
        if not os.path.exists(os.path.join(tmp_dir, "tokenizer_config.json")):
            raise FileNotFoundError(
                f"tokenizer_config.json not found in {tmp_dir}"
            )

        return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="mobilebert_uncased")
    parser.add_argument("--hf_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--upload_uri", type=str, default=None)
    args = parser.parse_args()

    hf_id = PRESET_MAP.get(args.preset, args.preset)

    with tempfile.TemporaryDirectory() as tmp_dir:
        hf_model, hf_tokenizer = load_hf_model(hf_id, args.hf_dir, tmp_dir)
        hf_model.eval()

        if MobileBertMaskedLM is not None:
            keras_model = MobileBertMaskedLM.from_preset(tmp_dir)
            keras_backbone = keras_model.backbone
        else:
            keras_model = MobileBertBackbone.from_preset(tmp_dir)
            keras_backbone = keras_model

        errors = []

        # Tokenizer test
        # 3 sentences
        sentences = [
            "This is a test.",
            "MobileBERT is a smaller BERT model.",
            "KerasHub is a library for Keras models.",
        ]

        print("Validating tokenizer...")
        keras_tokenizer = None
        if (
            hasattr(keras_model, "preprocessor")
            and keras_model.preprocessor is not None
        ):
            keras_tokenizer = keras_model.preprocessor.tokenizer
        else:
            try:
                from keras_hub.src.models.mobile_bert import (
                    mobile_bert_tokenizer as mbt,
                )

                MobileBertTokenizer = mbt.MobileBertTokenizer

                keras_tokenizer = MobileBertTokenizer.from_preset(tmp_dir)
            except ImportError:
                print("Skipped tokenizer (MobileBertTokenizer not found).")

        if keras_tokenizer is not None:
            for text in sentences:
                hf_ids = hf_tokenizer(text)["input_ids"]
                hf_body = [
                    x
                    for x in hf_ids
                    if x
                    not in (
                        hf_tokenizer.cls_token_id,
                        hf_tokenizer.sep_token_id,
                    )
                ]
                keras_ids = [int(i) for i in keras_tokenizer([text])[0]]
                if keras_ids != hf_body:
                    errors.append(
                        f"Mismatch '{text}': HF={hf_body}, K={keras_ids}"
                    )

            # Print success if no tokenizer errors were added
            if not any("Mismatch" in e for e in errors):
                print("Tokenizer ids match.")

        # Parameter count test
        print("Validating parameter count...")
        hf_state_dict = hf_model.state_dict()
        if MobileBertMaskedLM is None:
            # MaskedLM not available yet, only compare backbone weights.
            # Backbone weights lack "cls." in raw backbone models, or start
            # with "mobilebert."
            hf_numel = 0
            for n, p in hf_state_dict.items():
                if "position_ids" in n:
                    continue
                if n.startswith("mobilebert.") or not n.startswith("cls."):
                    hf_numel += p.numel()
        else:
            # Full model: exclude sequence relationship and duplicate
            # tied decoder weights.
            hf_numel = 0
            for n, p in hf_state_dict.items():
                if "position_ids" in n:
                    continue
                if "cls.seq_relationship." in n:
                    continue
                if (
                    "cls.predictions.decoder.weight" in n
                    or "cls.predictions.decoder.bias" in n
                    or n == "decoder.bias"
                ):
                    continue
                hf_numel += p.numel()

        keras_numel = sum(np.prod(w.shape) for w in keras_model.weights)
        if hf_numel != keras_numel:
            msg = f"Param count mismatch. HF={hf_numel} vs Keras={keras_numel}"
            errors.append(msg)
            print(msg)
        else:
            print(f"Parameter count matches exactly: {keras_numel}")

        # Backbone padded batch test
        print("Validating backbone padded batch...")
        hf_inputs = hf_tokenizer(sentences, padding=True, return_tensors="pt")
        with torch.no_grad():
            if hasattr(hf_model, "mobilebert"):
                hf_out = hf_model.mobilebert(
                    input_ids=hf_inputs["input_ids"],
                    attention_mask=hf_inputs["attention_mask"],
                )
            else:
                hf_out = hf_model(
                    input_ids=hf_inputs["input_ids"],
                    attention_mask=hf_inputs["attention_mask"],
                )
            hf_hidden = hf_out.last_hidden_state.cpu().numpy()

        input_ids = hf_inputs["input_ids"].cpu().numpy().astype("int32")
        padding_mask = hf_inputs["attention_mask"].cpu().numpy().astype("int32")

        segment_ids = np.zeros_like(input_ids)
        keras_hidden = keras_backbone(
            {
                "token_ids": input_ids,
                "segment_ids": segment_ids,
                "padding_mask": padding_mask,
            },
            training=False,
        )
        if isinstance(keras_hidden, dict):
            keras_hidden = keras_hidden["sequence_output"]
        keras_hidden = keras.ops.convert_to_numpy(keras_hidden)

        valid = padding_mask[..., None].astype("float32")
        diff = (
            np.abs(hf_hidden.astype("float32") - keras_hidden.astype("float32"))
            * valid
        )
        abs_max = float(np.max(diff))
        ref_max = float(np.max(np.abs(hf_hidden.astype("float32")) * valid))
        rel = abs_max / (ref_max if ref_max > 0 else 1.0)

        # MobileBERT uses NoNorm, so the final hidden state is not renormalized
        # and reaches ~7e7 in magnitude on google/mobilebert-uncased; HF's own
        # float32 forward differs from its float64 forward by ~30 absolute.
        # An absolute tolerance is meaningless at that scale; the MLM logits
        # below (after the head's LayerNorm) are gated absolutely.
        print(
            f"Backbone max diff: abs_max={abs_max:.6e}, "
            f"ref_max={ref_max:.6e}, rel={rel:.6e}"
        )
        if rel >= 1e-4:
            errors.append(f"Backbone max relative diff {rel:.6e} >= 1e-4")

        # MLM logits test
        if MobileBertMaskedLM is not None:
            print("Validating MLM logits...")
            with torch.no_grad():
                hf_mlm_out = hf_model(
                    input_ids=hf_inputs["input_ids"],
                    attention_mask=hf_inputs["attention_mask"],
                )
                hf_logits = hf_mlm_out.prediction_logits.cpu().numpy()

            # Keras MaskedLM expects mask_positions
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            mask_positions = np.arange(seq_len)[np.newaxis, :].repeat(
                batch_size, axis=0
            )

            keras_mlm_out = keras_model(
                {
                    "token_ids": input_ids,
                    "segment_ids": segment_ids,
                    "padding_mask": padding_mask,
                    "mask_positions": mask_positions,
                },
                training=False,
            )
            keras_logits = keras.ops.convert_to_numpy(keras_mlm_out)

            # HF outputs (batch, seq_len, vocab).
            # Keras outputs (batch, num_masks, vocab).
            # Since we set mask_positions = all positions, shapes should match.
            logits_diff = np.abs(
                hf_logits.astype("float32") - keras_logits.astype("float32")
            )
            max_logits_diff = float(np.max(logits_diff))
            print(f"MLM logits max diff: {max_logits_diff:.6e}")
            if max_logits_diff >= 1e-4:
                errors.append(
                    f"MLM logits max diff {max_logits_diff:.6e} >= 1e-4"
                )

        # Perturbation test
        print("Validating perturbation test...")
        w = keras_backbone.get_layer("token_embedding").embeddings
        orig = keras.ops.convert_to_numpy(w).copy()
        w.assign(orig + 1.0)

        keras_hidden_perturbed = keras_backbone(
            {
                "token_ids": input_ids,
                "segment_ids": segment_ids,
                "padding_mask": padding_mask,
            },
            training=False,
        )
        if isinstance(keras_hidden_perturbed, dict):
            keras_hidden_perturbed = keras_hidden_perturbed["sequence_output"]
        keras_hidden_perturbed = keras.ops.convert_to_numpy(
            keras_hidden_perturbed
        )

        diff_perturbed = (
            np.abs(
                hf_hidden.astype("float32")
                - keras_hidden_perturbed.astype("float32")
            )
            * valid
        )
        abs_max_perturbed = float(np.max(diff_perturbed))
        rel_perturbed = abs_max_perturbed / (ref_max if ref_max > 0 else 1.0)
        print(
            f"Backbone max diff after perturbation: "
            f"abs_max={abs_max_perturbed:.6e}, rel={rel_perturbed:.6e}"
        )
        if rel_perturbed <= rel:
            errors.append(
                "Perturbation did not increase the relative difference!"
            )
        w.assign(orig)  # restore

        if errors:
            print("\nValidation failed with errors:")
            for e in errors:
                print(f" - {e}")
            raise ValueError("Validators failed.")
        else:
            print(
                "\nAll validators finished successfully or skipped gracefully."
            )

        if args.output_dir:
            keras_model.save_to_preset(args.output_dir)
            print(f"Saved to {args.output_dir}")

        if args.upload_uri:
            import keras_hub

            keras_hub.upload_preset(args.upload_uri, args.output_dir)
            print(f"Uploaded to {args.upload_uri}")


if __name__ == "__main__":
    main()
