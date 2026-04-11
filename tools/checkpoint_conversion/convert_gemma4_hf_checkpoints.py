"""Convert Gemma4 HuggingFace checkpoints to the KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_gemma4_hf_checkpoints.py \
        --preset gemma4_instruct_2b \
        --save_dtype bfloat16
"""

import gc
import os
import random
from io import BytesIO

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import requests
import torch
from absl import app
from absl import flags
from keras import ops
from PIL import Image
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers import AutoTokenizer

import keras_hub

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

device = torch.device("cpu")
torch.set_default_device(device)

PRESET_MAP = {
    "gemma4_2b": "google/gemma-4-E2B",
    "gemma4_instruct_2b": "google/gemma-4-E2B-it",
    "gemma4_4b": "google/gemma-4-E4B",
    "gemma4_instruct_4b": "google/gemma-4-E4B-it",
    "gemma4_26b_a4b": "google/gemma-4-26B-A4B",
    "gemma4_instruct_26b_a4b": "google/gemma-4-26B-A4B-it",
    "gemma4_31b": "google/gemma-4-31B",
    "gemma4_instruct_31b": "google/gemma-4-31B-it",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
PROMPT = (
    "<start_of_turn>user\n\n<|image|>\nWhat is in this image?"
    "<end_of_turn>\n<start_of_turn>model\n"
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)

flags.DEFINE_string(
    "save_dtype",
    "bfloat16",
    "Dtype to save the model in. Defaults to bfloat16.",
)


def _evict_hf_cache(repo_id):
    """Delete all cached revisions for `repo_id` from the HF hub cache.

    This ensures that the subsequent `from_preset("hf://...")` call fetches
    the same weights that AutoModel loaded above (force_download=True), rather
    than reading a potentially stale local copy.
    """
    import huggingface_hub

    try:
        cache_info = huggingface_hub.scan_cache_dir()
    except Exception:
        return  # Non-fatal; skip if cache scan fails.

    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            commit_hashes = [rev.commit_hash for rev in repo.revisions]
            if commit_hashes:
                strategy = cache_info.delete_revisions(*commit_hashes)
                strategy.execute()
                print(
                    f"-> Evicted {len(commit_hashes)} cached revision(s) "
                    f"for {repo_id} from HF hub cache."
                )
            break


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _count_hf_params(hf_model):
    param_names = {name for name, _ in hf_model.named_parameters()}
    num_params = sum(param.numel() for param in hf_model.parameters())
    num_buffers = sum(
        value.numel()
        for name, value in hf_model.state_dict().items()
        if name not in param_names
        and (
            name.endswith(".layer_scalar")
            or (
                "vision_tower.encoder.layers" in name
                and name.endswith(
                    (".input_min", ".input_max", ".output_min", ".output_max")
                )
            )
            # std_bias / std_scale are registered buffers on the vision tower
            # for 26B-A4B and 31B models (standardize=True).
            or name
            in (
                "model.vision_tower.std_bias",
                "model.vision_tower.std_scale",
            )
        )
    )
    return num_params + num_buffers


def _count_keras_hub_params(backbone):
    unique_weights = {
        id(weight): weight for weight in backbone.weights
    }.values()
    return sum(weight.numpy().size for weight in unique_weights)


def _precompute_hf_outputs(
    hf_model, hf_tokenizer, hf_preset, prompt, raw_image
):
    processor = AutoProcessor.from_pretrained(hf_preset, force_download=True)
    hf_inputs = processor(
        text=prompt,
        images=raw_image,
        return_mm_token_type_ids=True,
        return_tensors="pt",
    )
    hf_inputs = {key: value.to(device) for key, value in hf_inputs.items()}

    bos_id = hf_tokenizer.bos_token_id
    if bos_id is not None and hf_inputs["input_ids"][0, 0].item() != bos_id:
        bos = torch.full(
            (hf_inputs["input_ids"].shape[0], 1),
            bos_id,
            dtype=hf_inputs["input_ids"].dtype,
            device=hf_inputs["input_ids"].device,
        )
        hf_inputs["input_ids"] = torch.cat([bos, hf_inputs["input_ids"]], dim=1)
        if "attention_mask" in hf_inputs:
            hf_inputs["attention_mask"] = torch.ones_like(
                hf_inputs["input_ids"]
            )
        if "mm_token_type_ids" in hf_inputs:
            mm_pad = torch.zeros(
                (hf_inputs["mm_token_type_ids"].shape[0], 1),
                dtype=hf_inputs["mm_token_type_ids"].dtype,
                device=hf_inputs["mm_token_type_ids"].device,
            )
            hf_inputs["mm_token_type_ids"] = torch.cat(
                [mm_pad, hf_inputs["mm_token_type_ids"]], dim=1
            )

    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)

    hf_logits = hf_outputs.logits.detach().cpu().float().numpy()
    hf_input_ids = hf_inputs["input_ids"].detach().cpu().numpy()
    hf_attention_mask = hf_inputs["attention_mask"].detach().cpu().numpy()
    hf_pixel_values = hf_inputs["pixel_values"].detach().cpu().float().numpy()
    hf_image_position_ids = (
        hf_inputs["image_position_ids"].detach().cpu().numpy()
    )

    with torch.no_grad():
        generated_ids = hf_model.generate(
            **hf_inputs,
            max_new_tokens=64,
            do_sample=False,
        )
    prompt_length = hf_inputs["input_ids"].shape[1]
    hf_generated_text = hf_tokenizer.decode(
        generated_ids[0, prompt_length:], skip_special_tokens=True
    )

    return {
        "logits": hf_logits,
        "input_ids": hf_input_ids,
        "attention_mask": hf_attention_mask,
        "pixel_values": hf_pixel_values,
        "image_position_ids": hf_image_position_ids,
        "generated_text": hf_generated_text,
        "param_count": _count_hf_params(hf_model),
    }


def _build_preprocessor_free_inputs(backbone, hf_data, image_placeholder_id):
    token_ids = hf_data["input_ids"].astype(np.int32)
    padding_mask = hf_data["attention_mask"].astype(np.int32)
    pixel_values = hf_data["pixel_values"].astype(np.float32)[
        :, np.newaxis, :, :
    ]
    pixel_position_ids = hf_data["image_position_ids"].astype(np.int32)[
        :, np.newaxis, :, :
    ]
    vision_mask = (token_ids == image_placeholder_id).astype(np.int32)

    batch_size = token_ids.shape[0]
    vision_rows = [
        np.where(vision_mask[index])[0].astype(np.int32)
        for index in range(batch_size)
    ]
    max_vision_tokens = max((len(row) for row in vision_rows), default=0)
    vision_indices = np.zeros((batch_size, max_vision_tokens), dtype=np.int32)
    for index, row in enumerate(vision_rows):
        vision_indices[index, : len(row)] = row

    keras_hub_inputs = {
        "token_ids": ops.convert_to_tensor(token_ids),
        "padding_mask": ops.convert_to_tensor(padding_mask),
        "pixel_values": ops.convert_to_tensor(pixel_values),
        "pixel_position_ids": ops.convert_to_tensor(pixel_position_ids),
        "vision_indices": ops.convert_to_tensor(vision_indices),
        "vision_mask": ops.convert_to_tensor(vision_mask),
    }

    if getattr(backbone, "audio_encoder", None) is not None:
        feat_size = getattr(backbone.audio_encoder, "input_feat_size", 128)
        keras_hub_inputs["audio_mel"] = ops.convert_to_tensor(
            np.zeros((batch_size, 0, 1, feat_size), dtype=np.float32)
        )
        keras_hub_inputs["audio_mel_mask"] = ops.convert_to_tensor(
            np.zeros((batch_size, 0, 1), dtype=np.int32)
        )
        keras_hub_inputs["audio_indices"] = ops.convert_to_tensor(
            np.zeros((batch_size, 0), dtype=np.int32)
        )

    return keras_hub_inputs


def _test_token_ids(preprocessor, prompt, raw_image, hf_token_ids):
    keras_hub_inputs = preprocessor.generate_preprocess(
        {"prompts": [prompt], "images": [raw_image]},
        sequence_length=hf_token_ids.shape[1],
    )
    keras_hub_token_ids = ops.convert_to_numpy(keras_hub_inputs["token_ids"])

    print("HF token ids (first 15):")
    print(hf_token_ids[0, :15].tolist())
    print("KerasHub token ids (first 15):")
    print(keras_hub_token_ids[0, :15].tolist())

    np.testing.assert_array_equal(keras_hub_token_ids, hf_token_ids)
    print("✓ Token IDs match.")


def _test_numerics(backbone, keras_hub_inputs, hf_logits):
    keras_hub_output = backbone(keras_hub_inputs)
    keras_hub_logits = backbone.token_embedding(keras_hub_output, reverse=True)
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits).astype(np.float32)

    abs_diff = np.abs(keras_hub_logits - hf_logits)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    print("\nPreprocessor-free logit comparison:")
    print(f"   Max absolute difference: {max_abs_diff:.6f}")
    print(f"   Mean absolute difference: {mean_abs_diff:.6f}")
    print("   Tolerance - atol: 0.001, rtol: 0.001")

    np.testing.assert_allclose(
        keras_hub_logits, hf_logits, atol=1e-3, rtol=1e-3
    )
    print("✓ Preprocessor-free logits within 1e-3 tolerance.")


def validate_output(
    keras_hub_model, prompt, raw_image, hf_generated_text, num_tokens
):
    keras_hub_model.preprocessor.num_vision_tokens_per_image = num_tokens
    keras_hub_output = keras_hub_model.generate(
        {"prompts": [prompt], "images": [raw_image]},
        max_length=2048 + 64,
    )
    keras_hub_generated_text = (
        keras_hub_output[0]
        if isinstance(keras_hub_output, list)
        else keras_hub_output
    )
    if isinstance(
        keras_hub_generated_text, str
    ) and keras_hub_generated_text.startswith(prompt):
        keras_hub_generated_text = keras_hub_generated_text[len(prompt) :]

    print("\nHF generate output:")
    print(hf_generated_text)
    print("\nKerasHub generate output:")
    print(keras_hub_generated_text)


def main(_):
    if FLAGS.preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one of "
            f"{','.join(PRESET_MAP.keys())}"
        )

    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]
    keras_hub_preset = f"hf://{hf_preset}"
    raw_image = _load_test_image()

    # Evict stale cache BEFORE any download so that both AutoModel and
    # from_preset share the same single fresh download.
    _evict_hf_cache(hf_preset)

    hf_model = AutoModelForImageTextToText.from_pretrained(
        hf_preset,
        device_map="cpu",
        torch_dtype=torch.float32,
        force_download=True,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        hf_preset,
        return_tensors="pt",
        force_download=True,
    )
    hf_model.eval()

    print("-> HuggingFace model loaded")
    hf_data = _precompute_hf_outputs(
        hf_model,
        hf_tokenizer,
        hf_preset,
        PROMPT,
        raw_image,
    )

    del hf_model
    gc.collect()

    keras_dtype = "float32"
    keras_hub_backbone = keras_hub.models.Gemma4Backbone.from_preset(
        keras_hub_preset,
        dtype=keras_dtype,
    )
    keras_hub_tokenizer = keras_hub.models.Gemma4Tokenizer.from_preset(
        keras_hub_preset
    )
    keras_hub_preprocessor = (
        keras_hub.models.Gemma4CausalLMPreprocessor.from_preset(
            keras_hub_preset,
        )
    )

    # Count the actual soft tokens HF produced for this specific image
    # (depends on image resolution; differs from the preset's max default).
    # Patch only for the token ID test, then restore so the saved preset
    # keeps the correct maximum value from the model config.
    actual_num_tokens = int(
        np.sum(
            hf_data["input_ids"][0] == keras_hub_tokenizer.image_placeholder_id
        )
    )
    original_num_tokens = keras_hub_preprocessor.num_vision_tokens_per_image
    keras_hub_preprocessor.num_vision_tokens_per_image = actual_num_tokens

    keras_hub_param_count = _count_keras_hub_params(keras_hub_backbone)
    hf_param_count = hf_data["param_count"]
    np.testing.assert_equal(keras_hub_param_count, hf_param_count)
    print(f"\n✓ Parameter count match: {keras_hub_param_count:,} params")

    _test_token_ids(
        keras_hub_preprocessor, PROMPT, raw_image, hf_data["input_ids"]
    )
    keras_hub_preprocessor.num_vision_tokens_per_image = original_num_tokens

    keras_hub_inputs = _build_preprocessor_free_inputs(
        keras_hub_backbone,
        hf_data,
        keras_hub_tokenizer.image_placeholder_id,
    )
    _test_numerics(keras_hub_backbone, keras_hub_inputs, hf_data["logits"])

    gemma4_lm = keras_hub.models.Gemma4CausalLM(
        backbone=keras_hub_backbone,
        preprocessor=keras_hub_preprocessor,
        sampler="greedy",
    )

    save_dtype = FLAGS.save_dtype
    preset_save_path = f"./{preset}"

    if save_dtype == "float32":
        print(f"\n-> Saving model in {save_dtype}...")
        gemma4_lm.save_to_preset(preset_save_path)
    else:
        del gemma4_lm
        del keras_hub_backbone
        gc.collect()

        print(f"\n-> Reloading model in {save_dtype} for saving...")
        keras_hub_backbone_save = keras_hub.models.Gemma4Backbone.from_preset(
            keras_hub_preset,
            dtype=save_dtype,
        )
        gemma4_lm_save = keras_hub.models.Gemma4CausalLM(
            backbone=keras_hub_backbone_save,
            preprocessor=keras_hub_preprocessor,
        )
        gemma4_lm_save.save_to_preset(preset_save_path)

    print(f"\n-> Saved converted model ({save_dtype}) to {preset_save_path}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
