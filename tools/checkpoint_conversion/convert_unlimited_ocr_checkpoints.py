"""Convert Unlimited-OCR HuggingFace checkpoints to KerasHub presets.

The script compares KerasHub against the native `transformers` class
`UnlimitedOcrForConditionalGeneration`. It runs four cases: a page and a
receipt in gundam mode (`crop_to_patches=True`), a page in base mode
(`crop_to_patches=False`), and a two-page prompt in base mode. Each side
preprocesses the raw input with its own code. The script saves the preset
only when every check passes.

Usage:
    python tools/checkpoint_conversion/convert_unlimited_ocr_checkpoints.py \
        --preset unlimited_ocr
"""

import contextlib
import gc
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from keras import ops
from PIL import Image
from transformers import AutoProcessor
from transformers import UnlimitedOcrForConditionalGeneration

import keras_hub
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_presets import (  # noqa: E402
    backbone_presets,
)

PRESET_MAP = {
    "unlimited_ocr": "baidu/Unlimited-OCR",
}

IMAGE_REPO = "huggingface/documentation-images"
# The two images of the HF modeling docs. The page is larger than one tile,
# so it gets local tiles. The receipt has no tiles.
PAGE_FILE = "transformers/model_doc/ocr_suggestion_form.jpg"
RECEIPT_FILE = "transformers/model_doc/ocr_receipt.jpeg"

IMAGE_PROMPT = "<image>document parsing."
MULTI_PAGE_PROMPT = "<image><image>Multi page parsing."
NUM_NEW_TOKENS = 256
ATOL = 1e-3
RTOL = 1e-3
# The logits at an image placeholder row are never used, because the next
# token is another placeholder. These rows carry the float32 noise of the
# vision towers. The text rows use `ATOL`. The image rows use `IMAGE_ATOL`.
# Measured on CPU in float32, the image rows of the gundam page (12 tiles)
# need an atol of at least 7.0e-3. The image rows of the other three cases
# pass with `ATOL`. `IMAGE_ATOL` is 7.0e-3 plus about 20%.
IMAGE_ATOL = 8.5e-3
# Rows per step when the check compares the logits.
CHECK_ROWS = 128

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_bool(
    "skip_generate",
    False,
    "If True, skip the greedy generation checks.",
)


def load_images():
    """Returns the page and the receipt as `uint8` arrays."""

    def load(file):
        path = hf_hub_download(IMAGE_REPO, file, repo_type="dataset")
        return np.array(Image.open(path).convert("RGB"))

    return load(PAGE_FILE), load(RECEIPT_FILE)


def get_hf_stripped_tokens(tokenizer):
    """Returns the token strings that HF removes with `skip_special_tokens`.

    The HF tokenizer marks these tokens as special. KerasHub keeps some of
    them in the text, for example `<|det|>` and `<td>`. Tokens such as
    `<PAGE>` and `<table>` are not special, so HF keeps them.
    """
    return sorted(
        (
            str(token)
            for token in tokenizer.added_tokens_decoder.values()
            if token.special
        ),
        key=len,
        reverse=True,
    )


def strip_tokens(text, tokens):
    """Removes every string in `tokens` from `text`.

    Pass `tokens` longest first, so a token never loses part of a longer one.
    """
    for token in tokens:
        text = text.replace(token, "")
    return text


def precompute_hf_outputs(hf_preset, cases):
    """Runs the HF model on every case and frees the model.

    Args:
        hf_preset: str. The HF model id.
        cases: dict. Maps a label to a `(prompt, images, crop_to_patches)`
            tuple. `images` is a list of arrays, or `None` for a text case.
            `crop_to_patches=True` is the gundam mode. `False` is the base
            mode.

    Returns:
        A dict with the HF token ids, the logits and the greedy text of
        every case, and the parameter count.
    """
    processor = AutoProcessor.from_pretrained(hf_preset)
    model = UnlimitedOcrForConditionalGeneration.from_pretrained(
        hf_preset, dtype=torch.float32
    ).eval()
    results = {"num_parameters": model.num_parameters(), "cases": {}}
    stripped_tokens = get_hf_stripped_tokens(processor.tokenizer)
    for label, (prompt, images, crop_to_patches) in cases.items():
        kwargs = {
            "text": [prompt],
            "return_tensors": "pt",
            "crop_to_patches": crop_to_patches,
        }
        if images is not None:
            kwargs["images"] = [Image.fromarray(image) for image in images]
        inputs = processor(**kwargs)
        with torch.no_grad():
            logits = model(**inputs).logits.float().numpy()
        result = {
            "token_ids": inputs["input_ids"].numpy(),
            "logits": logits,
            "stripped_tokens": stripped_tokens,
        }
        if not FLAGS.skip_generate:
            with torch.no_grad():
                generated = model.generate(
                    **inputs, max_new_tokens=NUM_NEW_TOKENS, do_sample=False
                )
            prompt_length = inputs["input_ids"].shape[1]
            result["text"] = processor.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )[0]
        results["cases"][label] = result
    del model, processor
    gc.collect()
    return results


def test_parameter_count(keras_model, hf_results):
    keras_params = keras_model.backbone.count_params()
    hf_params = hf_results["num_parameters"]
    print(f"KerasHub params: {keras_params:,}")
    print(f"HF params:       {hf_params:,}")
    np.testing.assert_equal(
        keras_params, hf_params, err_msg="parameter counts differ"
    )
    print("✓ Parameter count matches.")


def make_keras_prompt(case):
    prompt, images, _ = case
    x = {"prompts": [prompt]}
    if images is not None:
        x["images"] = images
    return x


@contextlib.contextmanager
def crop_mode(keras_model, case):
    """Sets `crop_to_patches` of the image converter for one case.

    The context restores the value on exit.
    """
    converter = keras_model.preprocessor.image_converter
    original = converter.crop_to_patches
    converter.crop_to_patches = case[2]
    try:
        yield
    finally:
        converter.crop_to_patches = original


def preprocess_keras(keras_model, case, result):
    """Preprocesses the raw input with the KerasHub preprocessor.

    The first call uses a long sequence. The count of real tokens must equal
    the HF count. The second call returns inputs of exactly that length.
    """
    x = make_keras_prompt(case)
    preprocessor = keras_model.preprocessor
    hf_length = result["token_ids"].shape[1]
    with crop_mode(keras_model, case):
        padded = preprocessor.generate_preprocess(
            x, sequence_length=hf_length + 16
        )
        np.testing.assert_equal(
            int(ops.convert_to_numpy(padded["padding_mask"]).sum()),
            hf_length,
            err_msg="token counts differ",
        )
        return preprocessor.generate_preprocess(x, sequence_length=hf_length)


def test_token_ids(label, keras_inputs, hf_token_ids):
    np.testing.assert_array_equal(
        ops.convert_to_numpy(keras_inputs["token_ids"]),
        hf_token_ids,
        err_msg=f"[{label}] token ids differ",
    )
    print(f"✓ [{label}] Token IDs match ({hf_token_ids.shape[1]} tokens).")


def report_numerics(
    label, keras_logits, hf_logits, token_ids=None, image_token_id=None
):
    keras_logits = ops.convert_to_numpy(keras_logits).astype("float32")
    diff = np.abs(keras_logits - hf_logits)
    outliers = int(np.sum(diff > ATOL + RTOL * np.abs(hf_logits)))
    print(f"[{label}] KerasHub logits: {keras_logits[0, 0, :5]}")
    print(f"[{label}] HF logits:       {hf_logits[0, 0, :5]}")
    print(
        f"[{label}] max {diff.max():.2e} · mean {diff.mean():.2e} · "
        f"outliers {outliers}/{diff.size}"
    )
    if outliers:
        # Outliers in a few token rows point at one routing or vision
        # event. Outliers in many rows point at spread-out noise.
        bad = diff > ATOL + RTOL * np.abs(hf_logits)
        rows, counts = np.unique(np.nonzero(bad)[1], return_counts=True)
        print(f"[{label}] outlier token rows: {dict(zip(rows, counts))}")
        row_max = diff.max(axis=(0, 2))
        worst = np.argsort(row_max)[-5:][::-1]
        print(
            f"[{label}] worst rows: "
            f"{[(int(r), float(f'{row_max[r]:.2e}')) for r in worst]}"
        )
        if token_ids is not None:
            ids = np.asarray(token_ids)[0]
            first = int(np.argmax(ids == image_token_id))
            for r in worst[:2]:
                print(
                    f"[{label}] row {int(r)}: token id {int(ids[r])}, "
                    f"placeholder ordinal {int(r) - first}, "
                    f"neighbors {ids[r - 2 : r + 3].tolist()}"
                )


def compare_rows(keras_logits, hf_logits, rows, atol):
    """Compares the logits of the rows in `rows` with HF.

    Args:
        keras_logits: float32 array of shape `(batch, length, vocabulary)`.
        hf_logits: float32 array of the same shape.
        rows: tuple of two int arrays. The batch and position indices of
            the rows to compare.
        atol: float. The absolute tolerance. `RTOL` is the relative one.

    Returns:
        A `(max difference, number of outliers)` tuple.
    """
    max_difference = 0.0
    outliers = 0
    for start in range(0, len(rows[0]), CHECK_ROWS):
        batch = rows[0][start : start + CHECK_ROWS]
        position = rows[1][start : start + CHECK_ROWS]
        expected = hf_logits[batch, position]
        difference = np.abs(keras_logits[batch, position] - expected)
        max_difference = max(max_difference, float(difference.max()))
        outliers += int(np.sum(difference > atol + RTOL * np.abs(expected)))
    return max_difference, outliers


def test_numerics(label, keras_logits, hf_logits, token_ids, image_token_id):
    """Checks the logits of the text rows and of the image rows.

    The text rows must be within `ATOL`. The image rows must be within
    `IMAGE_ATOL`. The function checks both groups, then raises if one fails.
    """
    report_numerics(label, keras_logits, hf_logits, token_ids, image_token_id)
    keras_logits = ops.convert_to_numpy(keras_logits).astype("float32")
    is_image = np.asarray(token_ids) == image_token_id
    failed = []
    for name, mask, atol in (
        ("text", ~is_image, ATOL),
        ("image", is_image, IMAGE_ATOL),
    ):
        rows = np.nonzero(mask)
        if len(rows[0]) == 0:
            continue
        max_difference, outliers = compare_rows(
            keras_logits, hf_logits, rows, atol
        )
        print(
            f"[{label}] {name} rows: {len(rows[0])} · max "
            f"{max_difference:.2e} · outliers at atol {atol:g}: {outliers}"
        )
        if outliers:
            failed.append(name)
    if failed:
        raise AssertionError(
            f"[{label}] logits differ from HF at the {', '.join(failed)} rows"
        )
    print(
        f"✓ [{label}] Logits within {ATOL:g} at the text rows and "
        f"{IMAGE_ATOL:g} at the image rows."
    )


def test_generation(label, keras_model, keras_inputs, case, result):
    x = make_keras_prompt(case)
    prompt_length = result["token_ids"].shape[1]
    prompt_text = keras_model.preprocessor.generate_postprocess(
        {
            "token_ids": keras_inputs["token_ids"],
            "padding_mask": keras_inputs["padding_mask"],
        }
    )[0]
    with crop_mode(keras_model, case):
        keras_text = keras_model.generate(
            x, max_length=prompt_length + NUM_NEW_TOKENS
        )
    if not isinstance(keras_text, str):
        keras_text = keras_text[0]
    assert keras_text.startswith(prompt_text), (
        f"[{label}] the output does not start with the prompt"
    )
    new_text = keras_text[len(prompt_text) :]
    assert len(new_text) > 0, f"[{label}] KerasHub generated no new text"
    print(f"\n[{label}] HF generated: {result['text']!r}")
    # HF drops its special tokens when it decodes. Drop the same tokens from
    # the KerasHub text, so both texts print in the same form.
    new_text = strip_tokens(new_text, result["stripped_tokens"])
    print(f"[{label}] KH generated: {new_text!r}")
    np.testing.assert_equal(
        new_text,
        result["text"],
        err_msg=f"[{label}] the generated text differs from HF",
    )
    print(f"[{label}] ✓ generated text matches HF")


def run_check(failures, name, check, *args):
    """Runs one check and records a failure instead of raising."""
    try:
        return check(*args)
    except Exception as error:
        print(f"✗ {name} failed: {error}")
        failures.append(name)
        return None


def validate_output(keras_model, cases, hf_results, skip_generate=False):
    """Run every check of the KerasHub model against the HF results.

    Every check runs, even after a failure. At the end, the function raises
    if any check failed. The caller saves the preset only when this function
    returns.

    Args:
        keras_model: The KerasHub task model.
        cases: dict. Maps a label to a `(prompt, images, crop_to_patches)`
            tuple. The KerasHub model reads its raw inputs from this dict.
        hf_results: dict. The output of `precompute_hf_outputs`.
        skip_generate: bool. If true, skip the generation check.
    """
    hf_cases = hf_results["cases"]
    failures = []

    print("\n--- Parameter Count ---")
    run_check(
        failures,
        "parameter count",
        test_parameter_count,
        keras_model,
        hf_results,
    )

    print("\n--- Token IDs ---")
    keras_inputs = {}
    for label, result in hf_cases.items():
        keras_inputs[label] = run_check(
            failures,
            f"[{label}] preprocessing",
            preprocess_keras,
            keras_model,
            cases[label],
            result,
        )
        if keras_inputs[label] is None:
            continue
        run_check(
            failures,
            f"[{label}] token ids",
            test_token_ids,
            label,
            keras_inputs[label],
            result["token_ids"],
        )

    print("\n--- Numerics Comparison ---")
    for label, result in hf_cases.items():
        if keras_inputs[label] is None:
            continue
        run_check(
            failures,
            f"[{label}] numerics",
            lambda label=label, result=result: test_numerics(
                label,
                keras_model(keras_inputs[label]),
                result["logits"],
                token_ids=result["token_ids"],
                image_token_id=keras_model.preprocessor.tokenizer.image_token_id,
            ),
        )
        print()

    if not skip_generate:
        # `compile` defaults to the `top_k` sampler, which samples at random.
        keras_model.compile(sampler="greedy")
        print("--- Generation ---")
        for label, result in hf_cases.items():
            if keras_inputs[label] is None:
                continue
            run_check(
                failures,
                f"[{label}] generation",
                test_generation,
                label,
                keras_model,
                keras_inputs[label],
                cases[label],
                result,
            )

    if failures:
        raise AssertionError(
            f"{len(failures)} check(s) failed: {', '.join(failures)}"
        )
    print("\n✓ All checks passed.")


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )
    if preset not in backbone_presets:
        raise ValueError(f"'{preset}' is missing from unlimited_ocr_presets.")
    hf_preset = PRESET_MAP[preset]

    page, receipt = load_images()
    # Multi-page input uses base mode, as in the HF README.
    cases = {
        "gundam page with tiles": (IMAGE_PROMPT, [page], True),
        "gundam receipt": (IMAGE_PROMPT, [receipt], True),
        "base page": (IMAGE_PROMPT, [page], False),
        "base multi-page": (MULTI_PAGE_PROMPT, [page, receipt], False),
    }
    print("-> Running the HF model...")
    hf_results = precompute_hf_outputs(hf_preset, cases)

    print("-> Loading the KerasHub model...")
    keras_model = keras_hub.models.UnlimitedOCRCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    validate_output(
        keras_model, cases, hf_results, skip_generate=FLAGS.skip_generate
    )

    # Save with the dtype of the released checkpoint.
    # Reset the Keras name counter. Otherwise the saved layers get a `_1`
    # suffix.
    del keras_model
    keras.backend.clear_session()
    gc.collect()
    keras_model = keras_hub.models.UnlimitedOCRCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="bfloat16"
    )
    keras_model.save_to_preset(f"./{preset}")
    print(f"-> Saved the preset to ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
