import os
import traceback

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from absl import app
from absl import flags

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

from keras import ops  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers import Qwen2VLForConditionalGeneration  # noqa: E402

import keras_hub  # noqa: E402
from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (  # noqa: E402
    Qwen2VLImageConverter,
)

PRESET_MAP = {
    "qwen2_vl_2b_instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2_vl_7b_instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2_vl_72b_instruct": "Qwen/Qwen2-VL-72B-Instruct",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
)


# ── Tokenizer verification ──────────────────────────────────────────


def test_tokenizer(keras_tokenizer, hf_tokenizer):
    """Compare token ids for several prompts."""
    test_strings = [
        "What is Keras?",
        "Describe the weather today.",
        "Hello, world! 🌍",
    ]
    print("\n── Tokenizer verification ──")
    for s in test_strings:
        hf_ids = hf_tokenizer(s, add_special_tokens=False)["input_ids"]
        keras_ids = keras_tokenizer(s)
        if hasattr(keras_ids, "numpy"):
            keras_ids = keras_ids.numpy()
        keras_ids = np.asarray(keras_ids).flatten().tolist()
        np.testing.assert_equal(
            keras_ids,
            hf_ids,
            err_msg=f"Tokenizer mismatch on: {s!r}",
        )
        print(f" '{s}' → {len(hf_ids)} tokens match")
    print(" All tokenizer checks passed")


# ── Preprocessor verification ────────────────────────────────────────


def test_preprocessor(keras_tokenizer, hf_processor):
    """Compare vision token insertion and padding."""
    print("\n── Preprocessor verification ──")

    # 1. Text-only path
    text = "Describe the weather"
    hf_text_ids = hf_processor.tokenizer(text, add_special_tokens=False)[
        "input_ids"
    ]

    keras_preprocessor = keras_hub.models.Qwen2VLCausalLMPreprocessor(
        tokenizer=keras_tokenizer,
        sequence_length=32,
    )
    result = keras_preprocessor.generate_preprocess(text)
    keras_text_ids = result["token_ids"]
    # The non-padded portion should match HF
    keras_trimmed = keras_text_ids[: len(hf_text_ids)]
    np.testing.assert_equal(
        np.asarray(keras_trimmed),
        np.asarray(hf_text_ids),
        err_msg="Text-only preprocessor mismatch",
    )
    print("Text-only preprocessing matches")

    # 2. With image — verify vision token block structure
    image_converter = Qwen2VLImageConverter()
    keras_preprocessor_img = keras_hub.models.Qwen2VLCausalLMPreprocessor(
        tokenizer=keras_tokenizer,
        image_converter=image_converter,
        sequence_length=512,
        spatial_merge_size=2,
    )
    dummy_image = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
    result = keras_preprocessor_img.generate_preprocess(
        {"text": "Describe this image", "images": dummy_image}
    )
    assert result["patch_values"] is not None, "patch_values should not be None"
    assert result["image_grid_thw"] is not None, "grid_thw should not be None"

    # Check vision token block is present in token_ids
    token_ids = result["token_ids"]
    vision_start_id = keras_tokenizer.vision_start_token_id
    vision_end_id = keras_tokenizer.vision_end_token_id
    image_pad_id = keras_tokenizer.image_pad_token_id
    assert vision_start_id in token_ids, "Missing <|vision_start|> token"
    assert vision_end_id in token_ids, "Missing <|vision_end|> token"
    assert image_pad_id in token_ids, "Missing <|image_pad|> tokens"

    # Count image_pad tokens and verify against grid_thw
    grid_thw = result["image_grid_thw"]
    expected_vision_tokens = int(
        np.prod(grid_thw[0]) // (2**2)  # spatial_merge_size²
    )
    actual_vision_tokens = int(np.sum(np.asarray(token_ids) == image_pad_id))
    assert actual_vision_tokens == expected_vision_tokens, (
        f"Vision token count mismatch: expected {expected_vision_tokens}, "
        f"got {actual_vision_tokens}"
    )
    print(
        f"Image preprocessing: {expected_vision_tokens} vision tokens "
        f"from grid {grid_thw[0].tolist()}"
    )
    print("All preprocessor checks passed")


# ── Backbone verification ────────────────────────────────────────────


def test_backbone(
    keras_backbone,
    keras_tokenizer,
    hf_model,
    hf_tokenizer,
):
    """Compare backbone hidden-state outputs (text-only path)."""
    print("\n── Backbone verification ──")

    # Parameter count
    keras_params = keras_backbone.count_params()
    # HF model includes the VL adapter + lm_head; compare just the
    # backbone portion (text model + visual).
    hf_params = hf_model.num_parameters()
    print(f"  KerasHub params: {keras_params:,}")
    print(f"  HF total params: {hf_params:,}")

    # Text-only logits comparison
    test_text = "What is Keras?"
    hf_inputs = hf_tokenizer(
        test_text, return_tensors="pt", add_special_tokens=False
    ).to(device)
    with torch.no_grad():
        hf_outputs = hf_model.model(**hf_inputs)
    hf_hidden = hf_outputs.last_hidden_state.detach().cpu().float().numpy()

    keras_preprocessor = keras_hub.models.Qwen2VLCausalLMPreprocessor(
        tokenizer=keras_tokenizer,
        sequence_length=hf_hidden.shape[1],
    )
    keras_inputs = keras_preprocessor(
        [test_text], sequence_length=hf_hidden.shape[1]
    )[0]
    keras_inputs = {k: v.to(device) for k, v in keras_inputs.items()}
    keras_hidden = keras_backbone(keras_inputs)
    keras_hidden = ops.convert_to_numpy(keras_hidden)

    try:
        np.testing.assert_allclose(
            keras_hidden, hf_hidden, atol=1e-4, rtol=1e-4
        )
        print(" Backbone hidden states match (atol=1e-4)")
    except AssertionError as err:
        max_diff = np.max(np.abs(keras_hidden - hf_hidden))
        print(f" Max abs diff: {max_diff:.6e}")
        print(traceback.format_exc())
        print(err.args[0])

    # Also compare logits through the LM head
    keras_logits = keras_backbone.token_embedding(
        ops.convert_to_tensor(keras_hidden), reverse=True
    )
    keras_logits = ops.convert_to_numpy(keras_logits)

    with torch.no_grad():
        hf_logits = hf_model.lm_head(hf_outputs.last_hidden_state)
    hf_logits = hf_logits.detach().cpu().float().numpy()

    try:
        np.testing.assert_allclose(
            keras_logits, hf_logits, atol=1e-4, rtol=1e-4
        )
        print("LM head logits match (atol=1e-4)")
    except AssertionError as err:
        max_diff = np.max(np.abs(keras_logits - hf_logits))
        print(f"Max logits diff: {max_diff:.6e}")
        print(traceback.format_exc())
        print(err.args[0])

    print("Backbone verification complete")


# ── Main ─────────────────────────────────────────────────────────────


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {preset}. "
            f"Must be one of {','.join(PRESET_MAP.keys())}"
        )
    hf_preset = PRESET_MAP[preset]

    print(f"🏃 Loading HuggingFace model: {hf_preset}")
    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
        hf_preset,
        device_map=device,
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_processor = AutoProcessor.from_pretrained(hf_preset)
    print("HF model loaded")

    print("🏃 Loading KerasHub model")
    keras_backbone = keras_hub.models.Qwen2VLBackbone.from_preset(
        f"hf://{hf_preset}"
    )
    keras_tokenizer = keras_hub.models.Qwen2VLTokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    print("KerasHub model loaded")

    # Run all verification steps
    test_tokenizer(keras_tokenizer, hf_tokenizer)
    test_preprocessor(keras_tokenizer, hf_processor)
    test_backbone(keras_backbone, keras_tokenizer, hf_model, hf_tokenizer)

    print(f"\n🏁 All verification passed for {preset}!")

    # Save preset
    keras_backbone.save_to_preset(f"./{preset}")
    keras_tokenizer.save_to_preset(f"./{preset}")
    print(f"Preset saved to ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
