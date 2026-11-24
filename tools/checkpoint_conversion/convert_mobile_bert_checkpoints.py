"""Convert MobileBERT from Hugging Face to KerasHub.

Usage:
  python tools/checkpoint_conversion/convert_mobile_bert_checkpoints.py \
    --preset mobilebert_uncased \
    --output_dir /tmp/mobilebert
"""

import contextlib
import os
import tempfile

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch
import transformers
from absl import app
from absl import flags
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import MobileBertForMaskedLM
from transformers import MobileBertForPreTraining
from transformers import MobileBertModel
from transformers import MobileBertPreTrainedModel

from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)
from keras_hub.src.models.mobile_bert.mobile_bert_masked_lm import (
    MobileBertMaskedLM,
)
from keras_hub.src.models.mobile_bert.mobile_bert_masked_lm_preprocessor import (  # noqa: E501
    MobileBertMaskedLMPreprocessor,
)
from keras_hub.src.models.mobile_bert.mobile_bert_text_classifier_preprocessor import (  # noqa: E501
    MobileBertTextClassifierPreprocessor,
)
from keras_hub.src.models.mobile_bert.mobile_bert_tokenizer import (
    MobileBertTokenizer,
)
from keras_hub.src.utils.preset_utils import HF_PREFIX
from keras_hub.src.utils.preset_utils import SAFETENSOR_CONFIG_FILE
from keras_hub.src.utils.preset_utils import SAFETENSOR_FILE
from keras_hub.src.utils.preset_utils import check_file_exists

FLAGS = flags.FLAGS
flags.DEFINE_string("preset", "mobilebert_uncased", "Preset name.")
flags.DEFINE_string(
    "output_dir", None, "Output directory for the saved preset."
)

PRESET_MAP = {"mobilebert_uncased": "google/mobilebert-uncased"}


def _check_hf_loading_info(info):
    bad = {
        k: sorted(str(x) for x in info[k])
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys")
        if info[k]
    }
    if bad:
        raise ValueError(f"Incomplete HF checkpoint load: {bad}")


def _hf_mlm_logits(output):
    return (
        output.prediction_logits
        if hasattr(output, "prediction_logits")
        else output.logits
    )


def main(_):
    raw_preset = PRESET_MAP.get(FLAGS.preset, FLAGS.preset)
    hf_id = (
        raw_preset[len(HF_PREFIX) :]
        if raw_preset.startswith(HF_PREFIX)
        else raw_preset
    )
    preset_uri = hf_id if os.path.isdir(hf_id) else HF_PREFIX + hf_id

    hf_tokenizer = AutoTokenizer.from_pretrained(hf_id)
    hf_cfg = AutoConfig.from_pretrained(hf_id)
    arch = hf_cfg.architectures[0] if hf_cfg.architectures else None
    if arch is not None:
        hf_cls = getattr(transformers, arch, None)
        if not isinstance(hf_cls, type) or not issubclass(
            hf_cls, MobileBertPreTrainedModel
        ):
            raise ValueError(
                f"Unsupported MobileBERT architecture in config: {arch!r}."
            )
        hf_model, loading_info = hf_cls.from_pretrained(
            hf_id, output_loading_info=True
        )
        _check_hf_loading_info(loading_info)
    else:
        hf_model = None
        first_info = None
        for candidate_cls in (
            MobileBertForPreTraining,
            MobileBertForMaskedLM,
            MobileBertModel,
        ):
            model, info = candidate_cls.from_pretrained(
                hf_id, output_loading_info=True
            )
            if first_info is None:
                first_info = info
            if not (
                info["missing_keys"]
                or info["unexpected_keys"]
                or info["mismatched_keys"]
            ):
                hf_model = model
                break
        if hf_model is None:
            _check_hf_loading_info(first_info)
    hf_model.eval()
    hf_backbone = getattr(hf_model, "mobilebert", hf_model)
    has_mlm_head = hasattr(getattr(hf_model, "cls", None), "predictions")

    with contextlib.ExitStack() as stack:
        if check_file_exists(preset_uri, SAFETENSOR_FILE) or check_file_exists(
            preset_uri, SAFETENSOR_CONFIG_FILE
        ):
            load_preset = preset_uri
        else:
            staging_dir = stack.enter_context(tempfile.TemporaryDirectory())
            hf_model.save_pretrained(staging_dir)
            hf_tokenizer.save_pretrained(staging_dir)
            load_preset = staging_dir

        if has_mlm_head:
            keras_model = MobileBertMaskedLM.from_preset(load_preset)
            keras_backbone = keras_model.backbone
            keras_tokenizer = keras_model.preprocessor.tokenizer
            fill_tokenizer = MobileBertTokenizer.from_preset(
                load_preset, special_tokens_in_strings=True
            )
        else:
            keras_model = None
            keras_backbone = MobileBertBackbone.from_preset(load_preset)
            keras_tokenizer = MobileBertTokenizer.from_preset(load_preset)
            fill_tokenizer = None
    seq_len = keras_backbone.max_sequence_length

    errors = []

    sentences = [
        "This is a test.",
        "MobileBERT is a smaller BERT model.",
        "Café RÉSUMÉ naïve déjà vu",
    ]
    # A sentence pair exercises `segment_ids == 1`; a long input exercises
    # truncation at `seq_len` and every position embedding row.
    pairs = (["What is MobileBERT?"], ["A compact task-agnostic BERT."])
    long_sentence = ["word " * (seq_len + 40)]

    print("Validating tokenizer...")
    classifier_preprocessor = MobileBertTextClassifierPreprocessor(
        keras_tokenizer, sequence_length=seq_len
    )
    keras_batches = [
        classifier_preprocessor(sentences),
        # `y=None` keeps the tuple as two segments instead of `(x, y)`.
        classifier_preprocessor(pairs, y=None),
        classifier_preprocessor(long_sentence),
    ]
    keras_inputs = {
        k: np.concatenate(
            [keras.ops.convert_to_numpy(b[k]) for b in keras_batches]
        )
        for k in keras_batches[0]
    }
    keras_ids = keras_inputs["token_ids"]

    hf_kwargs = dict(
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    hf_batches = [
        hf_tokenizer(sentences, **hf_kwargs),
        hf_tokenizer(pairs[0], pairs[1], **hf_kwargs),
        hf_tokenizer(long_sentence, **hf_kwargs),
    ]
    hf_inputs = {
        k: torch.cat([b[k] for b in hf_batches]) for k in hf_batches[0]
    }

    tok_mismatches = 0
    for kh_key, hf_key in (
        ("token_ids", "input_ids"),
        ("segment_ids", "token_type_ids"),
        ("padding_mask", "attention_mask"),
    ):
        kh_arr = keras_inputs[kh_key].astype("int64")
        hf_arr = hf_inputs[hf_key].cpu().numpy().astype("int64")
        if not np.array_equal(kh_arr, hf_arr):
            tok_mismatches += 1
            row_idx = int(np.where(np.any(kh_arr != hf_arr, axis=1))[0][0])
            errors.append(
                f"Tokenizer {kh_key} mismatch at row={row_idx}. "
                f"HF: {hf_arr[row_idx]}, Keras: {kh_arr[row_idx]}"
            )
    if tok_mismatches == 0:
        print("Tokenizer token_ids, segment_ids, and padding_mask match.")

    print("Validating parameter count...")
    if has_mlm_head:
        hf_numel = sum(p.numel() for p in hf_model.parameters())
        seq_rel = getattr(hf_model.cls, "seq_relationship", None)
        if seq_rel is not None:
            hf_numel -= sum(p.numel() for p in seq_rel.parameters())
        keras_numel = keras_model.count_params()
    else:
        hf_numel = sum(p.numel() for p in hf_backbone.parameters())
        keras_numel = keras_backbone.count_params()

    if hf_numel != keras_numel:
        msg = f"Param count mismatch. HF={hf_numel} vs Keras={keras_numel}"
        errors.append(msg)
        print(msg)
    else:
        print(f"Parameter count matches exactly: {keras_numel}")

    print("Validating backbone padded batch...")
    with torch.no_grad():
        hf_out = hf_backbone(
            input_ids=hf_inputs["input_ids"],
            attention_mask=hf_inputs["attention_mask"],
            token_type_ids=hf_inputs["token_type_ids"],
        )
        hf_hidden = hf_out.last_hidden_state.cpu().numpy()

    padding_mask = keras.ops.convert_to_numpy(
        keras_inputs["padding_mask"]
    ).astype("int32")
    keras_backbone_out = keras_backbone(keras_inputs, training=False)
    keras_hidden = keras.ops.convert_to_numpy(
        keras_backbone_out["sequence_output"]
    )

    valid = padding_mask[..., None].astype("float32")
    diff = (
        np.abs(hf_hidden.astype("float32") - keras_hidden.astype("float32"))
        * valid
    )
    abs_max = float(np.max(diff))
    ref_max = float(np.max(np.abs(hf_hidden.astype("float32")) * valid))
    rel = abs_max / (ref_max if ref_max > 0 else 1.0)
    row_abs = diff.max(axis=(1, 2))
    row_ref = (np.abs(hf_hidden.astype("float32")) * valid).max(axis=(1, 2))
    per_row_rel = ", ".join(
        f"{(a / (r if r > 0 else 1.0)):.3e}" for a, r in zip(row_abs, row_ref)
    )
    print(
        f"Backbone max diff: abs_max={abs_max:.6e}, "
        f"ref_max={ref_max:.6e}, rel={rel:.6e}"
    )
    print(
        f"Backbone max rel diff per row (3 single, pair, long): {per_row_rel}"
    )
    # MobileBERT `no_norm` hidden states accumulate past ~7.4e7 across 24
    # unnormalized residual blocks, so float32 comparisons must be relative
    # (`rel < 1e-4`), whereas MLM logits sit near ~35 after the head LayerNorm
    # and therefore use an absolute `5e-4` bound below.
    if rel >= 1e-4:
        errors.append(f"Backbone max relative diff {rel:.6e} >= 1e-4")

    if hf_out.pooler_output is not None:
        hf_pooled = hf_out.pooler_output.cpu().numpy().astype("float32")
        keras_pooled = keras.ops.convert_to_numpy(
            keras_backbone_out["pooled_output"]
        ).astype("float32")
        pooled_abs_max = float(np.max(np.abs(hf_pooled - keras_pooled)))
        pooled_ref_max = float(np.max(np.abs(hf_pooled)))
        pooled_rel = pooled_abs_max / (
            pooled_ref_max if pooled_ref_max > 0 else 1.0
        )
        print(
            f"Backbone pooled_output max diff: abs_max={pooled_abs_max:.6e}, "
            f"ref_max={pooled_ref_max:.6e}, rel={pooled_rel:.6e}"
        )
        if pooled_rel >= 1e-4:
            errors.append(
                f"Backbone pooled_output max relative diff "
                f"{pooled_rel:.6e} >= 1e-4"
            )

    if has_mlm_head:
        print("Validating MLM logits...")
        with torch.no_grad():
            hf_mlm_out = hf_model(
                input_ids=hf_inputs["input_ids"],
                attention_mask=hf_inputs["attention_mask"],
                token_type_ids=hf_inputs["token_type_ids"],
            )
            hf_logits = _hf_mlm_logits(hf_mlm_out).cpu().numpy()

        batch_size = keras_ids.shape[0]
        mask_positions = np.arange(seq_len, dtype="int32")[
            np.newaxis, :
        ].repeat(batch_size, axis=0)
        mlm_inputs = {**keras_inputs, "mask_positions": mask_positions}
        keras_logits = keras_model(mlm_inputs, training=False)
        keras_logits = keras.ops.convert_to_numpy(keras_logits)

        valid_logits = padding_mask[..., None].astype("float32")
        logits_diff = (
            np.abs(hf_logits.astype("float32") - keras_logits.astype("float32"))
            * valid_logits
        )
        max_logits_diff = float(np.max(logits_diff))
        print(f"MLM logits max diff: {max_logits_diff:.6e}")
        per_row = ", ".join(f"{d:.3e}" for d in logits_diff.max(axis=(1, 2)))
        print(f"MLM logits max diff per row (3 single, pair, long): {per_row}")
        # Both sides are float32; HF's own float32 forward differs from its
        # float64 forward by up to ~4e-5 on these logits (hidden states reach
        # ~7e7 before the head's LayerNorm). So 1e-4 sits inside reordering
        # noise; 5e-4 is ~10x that floor, while a mapping error moves logits
        # by O(1).
        if max_logits_diff >= 5e-4:
            errors.append(f"MLM logits max diff {max_logits_diff:.6e} >= 5e-4")

        print("Validating fill-mask top-1...")
        test_sentence = "The capital of France is [MASK]."
        hf_fill_inputs = hf_tokenizer([test_sentence], return_tensors="pt")
        hf_mask_idx = (
            hf_fill_inputs["input_ids"] == hf_tokenizer.mask_token_id
        ).nonzero(as_tuple=True)[1][0]
        with torch.no_grad():
            hf_fill_logits = _hf_mlm_logits(hf_model(**hf_fill_inputs))
        hf_top1 = hf_fill_logits[0, hf_mask_idx].argmax().item()

        fill_preprocessor = MobileBertTextClassifierPreprocessor(
            fill_tokenizer, sequence_length=seq_len
        )
        keras_fill_inputs = fill_preprocessor([test_sentence])
        keras_fill_inputs = {
            k: keras.ops.convert_to_numpy(v)
            for k, v in keras_fill_inputs.items()
        }
        keras_fill_token_ids = keras_fill_inputs["token_ids"][0]
        mask_indices = (
            keras_fill_token_ids == keras_tokenizer.token_to_id("[MASK]")
        ).nonzero()[0]
        if len(mask_indices) == 0:
            raise ValueError("[MASK] token not found in Keras inputs.")
        keras_mask_idx = mask_indices[0]
        keras_fill_mask_positions = np.array([[keras_mask_idx]])
        keras_fill_logits = keras_model(
            {**keras_fill_inputs, "mask_positions": keras_fill_mask_positions},
            training=False,
        )
        keras_top1 = int(
            np.argmax(keras.ops.convert_to_numpy(keras_fill_logits[0, 0]))
        )

        print(f"HF top-1: {hf_tokenizer.convert_ids_to_tokens(hf_top1)}")
        print(f"Keras top-1: {keras_tokenizer.id_to_token(keras_top1)}")
        if hf_top1 != keras_top1:
            errors.append(
                f"Fill-mask mismatch: HF={hf_top1} vs Keras={keras_top1}"
            )

    print("Validating perturbation test...")
    w = keras_backbone.get_layer("token_embedding").embeddings
    orig = keras.ops.convert_to_numpy(w).copy()
    w.assign(orig + 1.0)

    keras_hidden_perturbed = keras_backbone(keras_inputs, training=False)[
        "sequence_output"
    ]
    keras_hidden_perturbed = keras.ops.convert_to_numpy(keras_hidden_perturbed)
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
        errors.append("Perturbation did not increase the relative difference!")
    w.assign(orig)

    if errors:
        print("\nValidation failed with errors:")
        for e in errors:
            print(f" - {e}")
        raise ValueError("Validators failed.")
    else:
        print("\nAll validators passed.")

    out_ctx = (
        contextlib.nullcontext(FLAGS.output_dir)
        if FLAGS.output_dir
        else tempfile.TemporaryDirectory()
    )
    with out_ctx as save_dir:
        if has_mlm_head:
            keras_model.save_to_preset(save_dir)
            reloaded_pre = MobileBertMaskedLMPreprocessor.from_preset(save_dir)
            if (
                reloaded_pre.mask_selection_rate != 0.15
                or reloaded_pre.mask_selection_length != 96
            ):
                raise ValueError(
                    "Saved preprocessor has unexpected masking config: "
                    f"mask_selection_rate={reloaded_pre.mask_selection_rate}, "
                    "mask_selection_length="
                    f"{reloaded_pre.mask_selection_length}"
                )
            reloaded = MobileBertMaskedLM.from_preset(save_dir)
            reloaded_logits = keras.ops.convert_to_numpy(
                reloaded(mlm_inputs, training=False)
            )
            reload_diff = float(np.max(np.abs(reloaded_logits - keras_logits)))
            print(f"Reloaded preset MLM logits max diff: {reload_diff:.6e}")
            if reload_diff > 1e-6:
                raise ValueError(f"Reloaded preset diverges: {reload_diff:.6e}")
        else:
            keras_backbone.save_to_preset(save_dir)
            keras_tokenizer.save_to_preset(save_dir)
            reloaded_bb = MobileBertBackbone.from_preset(save_dir)
            reloaded_tok = MobileBertTokenizer.from_preset(save_dir)
            reloaded_hidden = keras.ops.convert_to_numpy(
                reloaded_bb(keras_inputs, training=False)["sequence_output"]
            )
            reload_diff = float(np.max(np.abs(reloaded_hidden - keras_hidden)))
            print(f"Reloaded preset backbone max diff: {reload_diff:.6e}")
            if reload_diff > 1e-6:
                raise ValueError(f"Reloaded preset diverges: {reload_diff:.6e}")
            reloaded_rows = [
                [int(t) for t in row] for row in reloaded_tok(sentences)
            ]
            expected_rows = [
                [int(t) for t in row] for row in keras_tokenizer(sentences)
            ]
            if reloaded_rows != expected_rows:
                raise ValueError("Reloaded tokenizer output mismatch.")
        if FLAGS.output_dir:
            print(f"Saved to {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main)
