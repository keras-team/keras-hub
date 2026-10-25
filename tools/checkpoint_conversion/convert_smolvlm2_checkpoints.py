"""Convert SmolVLM2 HuggingFace checkpoints to KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_smolvlm2_checkpoints.py \
        --preset smolvlm2_256m_video_instruct
"""

import gc
import os
import random
import tempfile

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from io import BytesIO

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
from transformers.video_utils import VideoMetadata

import keras_hub

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

device = torch.device("cpu")
torch.set_default_device(device)

PRESET_MAP = {
    "smolvlm2_256m_video_instruct": (
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    ),
    "smolvlm2_500m_video_instruct": (
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    ),
    "smolvlm2_2.2b_instruct": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)
VIDEO_FPS = 1  # SmolVLM2 default sampling fps
# New tokens generated in the generation smoke test, on both sides.
GENERATE_NEW_TOKENS = 8
# KerasHub pads its own prompts to this length; the valid positions come
# from KerasHub's own padding mask. The preset default (1024) is shorter
# than a 13-crop prompt on the 2.2B preset (13 x 81 image tokens). Nothing
# about this value comes from the HF side.
KERASHUB_SEQUENCE_LENGTH = 2048
# Weight parity: HF's own pixel_values through the KerasHub backbone must
# reproduce HF's logits. This isolates the port from the preprocessing.
#
# The 2.2B value is the model's own float32 noise floor, measured on the
# COCO prompt by running HF in float64 and float32 on the same input: the
# two differ by up to 3.2e-3, all at `<image>` positions, the largest at
# sequence position 745. KerasHub's float32 logits differ from HF's by at
# most 1.5e-3, at that same position and nowhere else above 1e-3 (126 of
# 54M elements). The checkpoint is natively float32, so no dtype upcast is
# involved. 256M and 500M stay within 1e-3 (max 1.2e-4 measured).
WEIGHT_PARITY_ATOL = {
    "smolvlm2_256m_video_instruct": 1e-3,
    "smolvlm2_500m_video_instruct": 1e-3,
    "smolvlm2_2.2b_instruct": 4e-3,
}

# PROVISIONAL: pending reviewer decision on keras-team/keras-hub#3096 T37.
#
# End-to-end logit parity: raw image -> SmolVLM2CausalLMPreprocessor ->
# backbone, against HF processor -> HF model, with nothing fed across. The
# gate applies to the positions whose logits are consumed, i.e. every
# position that is not an `<image>` slot (the next token after an image
# slot is always another image token). The `<image>`-slot maximum is
# printed alongside but not asserted: those positions carry no output,
# and on the 2.2B preset they are where float32 itself is least stable
# (HF in float64 vs float32 already differs by 3.2e-3 there on HF's own
# pixels).
#
# The residual comes from `ops.image.scale_and_translate` (float Lanczos)
# vs PIL's fixed-point Lanczos: pixels agree to 1-2 u8, which the model
# amplifies most on the 64x64 image that gets upsampled ~30x. Each value
# is the measured text-position max over the seven inputs, rounded up to
# one significant figure.
#
#   preset   max@text (input)   max@<image> (input)   COCO max@text
#   256M     0.24 (64x64)       1.04 (64x64)          0.05
#   500M     0.23 (64x64)       1.91 (64x64)          0.02
#   2.2B     0.19 (batch)       9.84 (COCO)           0.04
E2E_LOGIT_ATOL = {
    "smolvlm2_256m_video_instruct": 0.3,
    "smolvlm2_500m_video_instruct": 0.3,
    "smolvlm2_2.2b_instruct": 0.2,
}

TEXT_PROMPT = "What is Keras?"
MULTIMODAL_TEXT = "Describe this image in detail."
VIDEO_TEXT = "Describe what is happening in this video."
# KerasHub prompt with <image> placeholder and chat formatting.
KERASHUB_MULTIMODAL_PROMPT = (
    "<|im_start|>User:<image>"
    + MULTIMODAL_TEXT
    + "<end_of_utterance>\nAssistant:"
)
# HF's chat template puts a space before a video, but not before an image.
KERASHUB_VIDEO_PROMPT = (
    "<|im_start|>User: <video>" + VIDEO_TEXT + "<end_of_utterance>\nAssistant:"
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _count_keras_params(backbone):
    """Count unique parameters (handles tied weights)."""
    unique = {id(w): w for w in backbone.weights}.values()
    return sum(w.numpy().size for w in unique)


def _decode_video(path):
    """Decode an MP4 into `(THWC uint8 frames, fps)`, failing loudly.

    Uses TorchCodec, which torchvision now points to for video decoding
    (`torchvision.io.read_video` was removed). TorchCodec needs FFmpeg
    installed on the system.
    """
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as e:
        raise RuntimeError(
            "Video validation needs `torchcodec`, which also needs FFmpeg. "
            "Install both; the video checks cannot be skipped."
        ) from e
    decoder = VideoDecoder(path, dimension_order="NHWC")
    return decoder[:], decoder.metadata.average_fps


def _hf_pixels_to_channels_last(pixel_values):
    """HF `(batch, images, C, H, W)` to KerasHub `(batch*images, H, W, C)`."""
    if pixel_values.ndim == 5:
        b, n, c, h, w = pixel_values.shape
        pixel_values = pixel_values.reshape(b * n, c, h, w)
    return np.transpose(pixel_values, (0, 2, 3, 1))


# ---------------------------------------------------------------
# 1. Precompute HF outputs (before freeing HF model)
# ---------------------------------------------------------------
def precompute_hf_outputs(hf_model, hf_tokenizer, hf_preset):
    """Precompute all HF outputs needed for validation."""
    results = {}

    processor = AutoProcessor.from_pretrained(hf_preset)

    # 1. text
    print("\n   Precomputing: text")
    hf_ids = hf_tokenizer(TEXT_PROMPT, return_tensors="pt")["input_ids"].to(
        device
    )
    with torch.no_grad():
        hf_out = hf_model(input_ids=hf_ids)
        hf_gen = hf_model.generate(
            input_ids=hf_ids,
            max_new_tokens=GENERATE_NEW_TOKENS,
            do_sample=False,
        )
    results["text"] = {
        "raw_prompts": [TEXT_PROMPT],
        "input_ids": hf_ids.cpu().numpy().astype(np.int32),
        "logits": hf_out.logits.detach().cpu().float().numpy(),
        "generated": hf_tokenizer.decode(hf_gen[0], skip_special_tokens=True),
    }

    # Helper for multimodal
    def _run_hf_image(
        name, images, do_image_splitting=True, prompt_text=MULTIMODAL_TEXT
    ):
        print(f"   Precomputing: {name}")
        if isinstance(images, list):
            messages = []
            for _ in range(len(images)):
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]
                )
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

        mm_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        if isinstance(mm_prompt, str):
            mm_prompt = [mm_prompt]

        # One image list per prompt when batching, a flat list otherwise.
        hf_images = [[img] for img in images] if len(images) > 1 else images
        kwargs = {
            "text": mm_prompt,
            "images": hf_images,
            "return_tensors": "pt",
        }
        # `SmolVLMImageProcessor` accepts `do_image_splitting` per call.
        if not do_image_splitting:
            kwargs["do_image_splitting"] = False

        hf_inputs = processor(**kwargs).to(device)

        with torch.no_grad():
            hf_out = hf_model(**hf_inputs)

        results[name] = {
            "input_ids": hf_inputs["input_ids"].cpu().numpy().astype(np.int32),
            "logits": hf_out.logits.detach().cpu().float().numpy(),
        }

        if name == "image_coco":
            # For weight parity validation, keep the raw pixel values.
            results[name]["pixel_values"] = (
                hf_inputs["pixel_values"].cpu().float().numpy()
            )
            with torch.no_grad():
                hf_gen = hf_model.generate(
                    **hf_inputs,
                    max_new_tokens=GENERATE_NEW_TOKENS,
                    do_sample=False,
                )
            results[name]["generated"] = processor.batch_decode(
                hf_gen, skip_special_tokens=True
            )[0]

    # 2. image_coco
    raw_coco = _load_test_image()
    _run_hf_image("image_coco", [raw_coco])
    results["image_coco"]["raw_images"] = [raw_coco]
    results["image_coco"]["raw_prompts"] = [KERASHUB_MULTIMODAL_PROMPT]

    # 3. image_small
    raw_small = raw_coco.resize((64, 64), Image.LANCZOS)
    _run_hf_image("image_small", [raw_small])
    results["image_small"]["raw_images"] = [raw_small]
    results["image_small"]["raw_prompts"] = [KERASHUB_MULTIMODAL_PROMPT]

    # 4. image_wide
    raw_wide = raw_coco.resize(
        (2048, int(2048 * raw_coco.size[1] / raw_coco.size[0])), Image.LANCZOS
    )
    center = raw_wide.size[1] // 2
    raw_wide = raw_wide.crop((0, center - 64, 2048, center + 64))
    _run_hf_image("image_wide", [raw_wide])
    results["image_wide"]["raw_images"] = [raw_wide]
    results["image_wide"]["raw_prompts"] = [KERASHUB_MULTIMODAL_PROMPT]

    # 5. image_batch
    raw_coco_flipped = raw_coco.transpose(Image.FLIP_LEFT_RIGHT)
    _run_hf_image("image_batch", [raw_coco, raw_coco_flipped])
    results["image_batch"]["raw_images"] = [raw_coco, raw_coco_flipped]
    results["image_batch"]["raw_prompts"] = [
        KERASHUB_MULTIMODAL_PROMPT,
        KERASHUB_MULTIMODAL_PROMPT,
    ]

    # 6. image_nosplit
    _run_hf_image("image_nosplit", [raw_coco], do_image_splitting=False)
    results["image_nosplit"]["raw_images"] = [raw_coco]
    results["image_nosplit"]["raw_prompts"] = [KERASHUB_MULTIMODAL_PROMPT]

    # 7. video
    print("\n   Precomputing: video")
    vid_response = requests.get(VIDEO_URL, timeout=60)
    vid_path = os.path.join(tempfile.gettempdir(), "smolvlm2_test_video.mp4")
    with open(vid_path, "wb") as f:
        f.write(vid_response.content)

    try:
        video_tensor, video_fps = _decode_video(vid_path)
    finally:
        if os.path.exists(vid_path):
            os.remove(vid_path)

    total_frames = video_tensor.shape[0]
    num_sample = max(4, min(int(total_frames / video_fps * VIDEO_FPS), 8))
    indices = np.linspace(0, total_frames - 1, num_sample).round().astype(int)
    sampled_frames = video_tensor[indices]

    video_frames = [
        Image.fromarray(sampled_frames[i].numpy())
        for i in range(sampled_frames.shape[0])
    ]
    video_metadata = {
        "fps": float(video_fps),
        "frames_indices": indices.tolist(),
    }

    video_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": VIDEO_TEXT},
            ],
        }
    ]
    video_prompt = processor.apply_chat_template(
        video_messages, add_generation_prompt=True
    )
    video_inputs = processor(
        text=video_prompt,
        videos=[[video_frames]],
        video_metadata=[
            VideoMetadata(total_num_frames=len(video_frames), **video_metadata)
        ],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        hf_video_out = hf_model(**video_inputs)
        hf_video_gen = hf_model.generate(
            **video_inputs,
            max_new_tokens=GENERATE_NEW_TOKENS,
            do_sample=False,
        )

    results["video"] = {
        "raw_videos": np.stack([np.array(f) for f in video_frames], axis=0)[
            None
        ],
        "raw_prompts": [KERASHUB_VIDEO_PROMPT],
        "video_metadata": video_metadata,
        "input_ids": video_inputs["input_ids"].cpu().numpy().astype(np.int32),
        "logits": hf_video_out.logits.detach().cpu().float().numpy(),
        "generated": processor.batch_decode(
            hf_video_gen, skip_special_tokens=True
        )[0],
    }

    return results


# ---------------------------------------------------------------
# 2. Parameter count comparison
# ---------------------------------------------------------------
def test_parameter_count(keras_backbone, hf_param_count):
    """Compare parameter counts between KerasHub and HF models."""
    print("\n" + "=" * 50)
    print("PARAMETER COUNT COMPARISON")
    print("=" * 50)

    keras_params = _count_keras_params(keras_backbone)
    print(f"\n  KerasHub params: {keras_params:,}")
    print(f"  HF params:       {hf_param_count:,}")

    if keras_params == hf_param_count:
        print("  ✓ Parameter counts match!")
    else:
        diff = hf_param_count - keras_params
        print(f"  ⚠ Parameter count difference: {diff:,}")


# ---------------------------------------------------------------
# 3. Validate text-only output (Weight Parity for Text)
# ---------------------------------------------------------------
def validate_text_output(keras_model, hf_results, preset):
    """Validate text-only tokenization and logits (Weight Parity)."""
    print("\n" + "=" * 50)
    print("TEXT-ONLY VALIDATION (WEIGHT PARITY)")
    print("=" * 50)

    hf_ids = hf_results["text"]["input_ids"]

    # --- Token ID parity ---
    keras_preprocessed = keras_model.preprocessor.generate_preprocess(
        [TEXT_PROMPT], sequence_length=KERASHUB_SEQUENCE_LENGTH
    )
    keras_ids = ops.convert_to_numpy(keras_preprocessed["token_ids"])
    keras_mask = ops.convert_to_numpy(keras_preprocessed["padding_mask"])

    hf_valid_len = hf_ids.shape[1]
    keras_valid_len = (
        int(np.sum(keras_mask[0]))
        if keras_mask.ndim == 2
        else int(np.sum(keras_mask))
    )

    print(f"\n  HF token ids:      {hf_ids[0][:10].tolist()}")
    print(f"  KerasHub token ids: {keras_ids[0][:10].tolist()}")

    common_len = min(hf_valid_len, keras_valid_len)
    np.testing.assert_array_equal(
        keras_ids[0, :common_len], hf_ids[0, :common_len]
    )
    print("  ✓ Token IDs match on valid positions.")

    # --- Logit comparison (preprocessor-free forward pass) ---
    token_ids = ops.convert_to_tensor(hf_ids.astype(np.int32))
    padding_mask = ops.ones_like(token_ids)

    keras_hidden = keras_model.backbone(
        {"token_ids": token_ids, "padding_mask": padding_mask}
    )
    keras_logits = keras_model.backbone.token_embedding(
        keras_hidden, reverse=True
    )
    keras_logits = ops.convert_to_numpy(keras_logits).astype(np.float32)

    hf_logits = hf_results["text"]["logits"]
    abs_diff = np.abs(keras_logits - hf_logits)
    print(f"\n  Logit mean absolute diff: {abs_diff.mean():.6f}")
    print(f"  Logit max absolute diff:  {abs_diff.max():.6f}")
    atol = WEIGHT_PARITY_ATOL[preset]
    np.testing.assert_allclose(keras_logits, hf_logits, atol=atol)
    print(f"  ✓ Logits match within atol={atol}.")


# ---------------------------------------------------------------
# 4. Validate Multimodal Weight Parity
# ---------------------------------------------------------------
def validate_weight_parity(keras_model, hf_results, preset):
    """Validate weight parity on COCO image using HF pixel_values."""
    print("\n" + "=" * 50)
    print("MULTIMODAL WEIGHT PARITY")
    print("=" * 50)

    hf_ids = hf_results["image_coco"]["input_ids"]
    hf_pixels = _hf_pixels_to_channels_last(
        hf_results["image_coco"]["pixel_values"]
    )
    hf_logits = hf_results["image_coco"]["logits"]

    preprocessed = keras_model.preprocessor.generate_preprocess(
        {
            "prompts": [KERASHUB_MULTIMODAL_PROMPT],
            "images": [np.array(hf_results["image_coco"]["raw_images"][0])],
        },
        sequence_length=KERASHUB_SEQUENCE_LENGTH,
    )

    backbone_inputs = {
        key: ops.convert_to_tensor(value) for key, value in preprocessed.items()
    }
    backbone_inputs["pixel_values"] = ops.convert_to_tensor(hf_pixels)
    backbone_inputs["padding_mask"] = ops.cast(
        backbone_inputs["padding_mask"], "int32"
    )

    hidden = keras_model.backbone(backbone_inputs)
    keras_logits = ops.convert_to_numpy(
        keras_model.backbone.token_embedding(hidden, reverse=True)
    ).astype(np.float32)

    # We compare on the common length
    keras_mask = ops.convert_to_numpy(preprocessed["padding_mask"])
    hf_valid_len = hf_ids.shape[1]
    keras_valid_len = (
        int(np.sum(keras_mask[0]))
        if keras_mask.ndim == 2
        else int(np.sum(keras_mask))
    )
    common_len = min(hf_valid_len, keras_valid_len)

    port_diff = np.abs(keras_logits[0, :common_len] - hf_logits[0, :common_len])
    print(f"\n  [HF pixels] logit mean absolute diff: {port_diff.mean():.6f}")
    print(f"  [HF pixels] logit max absolute diff:  {port_diff.max():.6f}")
    atol = WEIGHT_PARITY_ATOL[preset]
    np.testing.assert_allclose(port_diff, 0.0, atol=atol)
    print(f"  ✓ Logits on HF pixels match within atol={atol}.")


# ---------------------------------------------------------------
# 5. Validate E2E Parity
# ---------------------------------------------------------------
def validate_e2e_parity(keras_model, hf_results, preset):
    print("\n" + "=" * 50)
    print("END-TO-END PARITY")
    print("=" * 50)

    from keras_hub.src.models.smolvlm2.smolvlm2_causal_lm_preprocessor import (
        SmolVLM2CausalLMPreprocessor,
    )
    from keras_hub.src.models.smolvlm2.smolvlm2_image_converter import (
        SmolVLM2ImageConverter,
    )

    cfg = keras_model.preprocessor.image_converter.get_config()
    nosplit_converter = SmolVLM2ImageConverter(
        max_image_size=cfg["max_image_size"],
        size=cfg["size"],
        scale=cfg["scale"],
        offset=cfg["offset"],
        interpolation=cfg["interpolation"],
        antialias=cfg["antialias"],
        do_image_splitting=False,
    )
    nosplit_preprocessor = SmolVLM2CausalLMPreprocessor(
        tokenizer=keras_model.preprocessor.tokenizer,
        image_converter=nosplit_converter,
        video_converter=keras_model.preprocessor.video_converter,
        sequence_length=keras_model.preprocessor.sequence_length,
        image_seq_len=keras_model.preprocessor.image_seq_len,
        add_start_token=keras_model.preprocessor.add_start_token,
        add_end_token=keras_model.preprocessor.add_end_token,
    )

    tests = [
        ("text", keras_model.preprocessor),
        ("image_small", keras_model.preprocessor),
        ("image_coco", keras_model.preprocessor),
        ("image_wide", keras_model.preprocessor),
        ("image_batch", keras_model.preprocessor),
        ("image_nosplit", nosplit_preprocessor),
        ("video", keras_model.preprocessor),
    ]

    table_data = []

    for name, prep in tests:
        res = hf_results[name]
        hf_logits = res["logits"]
        hf_ids = res["input_ids"]

        kwargs = {"prompts": res["raw_prompts"]}
        if "raw_images" in res:
            kwargs["images"] = [np.array(img) for img in res["raw_images"]]
        if "raw_videos" in res:
            kwargs["videos"] = res["raw_videos"]
        if "video_metadata" in res:
            kwargs["video_metadata"] = res["video_metadata"]

        keras_preprocessed = prep.generate_preprocess(
            kwargs, sequence_length=KERASHUB_SEQUENCE_LENGTH
        )
        keras_ids = ops.convert_to_numpy(keras_preprocessed["token_ids"])
        keras_mask = ops.convert_to_numpy(keras_preprocessed["padding_mask"])

        # Every prompt in a batch expands to the same number of tokens, so
        # one valid length describes the batch.
        hf_valid_len = hf_ids.shape[1]
        keras_valid_len = (
            int(np.max(np.sum(keras_mask, axis=1)))
            if keras_mask.ndim == 2
            else int(np.sum(keras_mask))
        )

        common_len = min(hf_valid_len, keras_valid_len)

        ids_match = "Yes"
        if hf_valid_len != keras_valid_len:
            ids_match = f"No (HF:{hf_valid_len} KH:{keras_valid_len})"
        else:
            if not np.array_equal(
                hf_ids[:, :common_len], keras_ids[:, :common_len]
            ):
                ids_match = "No (Values)"

        keras_hidden = keras_model.backbone(keras_preprocessed)
        keras_logits = ops.convert_to_numpy(
            keras_model.backbone.token_embedding(keras_hidden, reverse=True)
        ).astype(np.float32)

        e2e_diff = np.abs(
            keras_logits[:, :common_len] - hf_logits[:, :common_len]
        )
        max_diff = e2e_diff.max()
        mean_diff = e2e_diff.mean()
        # Logits at `<image>` slots are never consumed (the next token is
        # another image token), and on the 2.2B preset they are also where
        # float32 itself is least stable, so report them separately from
        # the text positions that produce output.
        image_token_id = keras_model.backbone.image_token_id
        is_image = hf_ids[:, :common_len] == image_token_id
        max_text = e2e_diff[~is_image].max() if (~is_image).any() else 0.0
        max_image = e2e_diff[is_image].max() if is_image.any() else 0.0

        table_data.append(
            (
                name,
                mean_diff,
                max_diff,
                max_text,
                max_image,
                ids_match,
                e2e_diff[~is_image],
            )
        )

    print("\nE2E LOGIT DIFFERENCE (KERASHUB PIXELS vs HF PIXELS)")
    print(
        f"{'Input':<14} | {'Mean':<9} | {'Max':<9} | {'Max@text':<9} | "
        f"{'Max@<image>':<11} | IDs"
    )
    print("-" * 72)
    for (
        name,
        mean_diff,
        max_diff,
        max_text,
        max_image,
        ids_match,
        _,
    ) in table_data:
        print(
            f"{name:<14} | {mean_diff:<9.6f} | {max_diff:<9.4f} | "
            f"{max_text:<9.4f} | {max_image:<11.4f} | {ids_match}"
        )

    atol = E2E_LOGIT_ATOL.get(preset)
    if atol is None:
        raise AssertionError(
            f"E2E_LOGIT_ATOL['{preset}'] is unset: fill it from the table "
            "above (Max@text rounded up to 1 significant figure) and re-run."
        )

    for name, _, _, _, _, _, text_diff in table_data:
        np.testing.assert_allclose(
            text_diff,
            0.0,
            atol=atol,
            err_msg=f"E2E logit parity at text positions failed for {name}",
        )

    print(
        f"\n  ✓ All inputs match E2E at text positions within atol={atol} "
        "(`<image>`-slot maxima reported above, not asserted)."
    )


# ---------------------------------------------------------------
# 6. Generation Smoke Test
# ---------------------------------------------------------------
def generation_smoke(keras_model, hf_results):
    print("\n" + "=" * 50)
    print("GENERATION SMOKE TEST")
    print("=" * 50)

    for name in ["text", "image_coco", "video"]:
        print(f"\n--- {name} ---")
        res = hf_results[name]

        kwargs = {"prompts": res["raw_prompts"]}
        if "raw_images" in res:
            kwargs["images"] = [np.array(img) for img in res["raw_images"]]
        if "raw_videos" in res:
            kwargs["videos"] = res["raw_videos"]
        if "video_metadata" in res:
            kwargs["video_metadata"] = res["video_metadata"]

        preprocessed = keras_model.preprocessor.generate_preprocess(
            kwargs, sequence_length=KERASHUB_SEQUENCE_LENGTH
        )
        prompt_len = int(
            ops.convert_to_numpy(preprocessed["padding_mask"])
            .sum(axis=-1)
            .max()
        )
        keras_output = keras_model.generate(
            kwargs, max_length=prompt_len + GENERATE_NEW_TOKENS
        )

        keras_text = (
            keras_output[0] if isinstance(keras_output, list) else keras_output
        )
        print(f"HF:       {res.get('generated', 'N/A')}")
        print(f"KerasHub: {keras_text}")


# ---------------------------------------------------------------
# 6. Save preset
# ---------------------------------------------------------------
def save_preset(keras_model, preset_name):
    """Save the converted model as a KerasHub preset."""
    print(f"\n-> Saving KerasHub preset to ./{preset_name}...")
    keras_model.save_to_preset(f"./{preset_name}")
    print(f"  ✓ Preset saved to ./{preset_name}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )

    hf_preset = PRESET_MAP[preset]

    # --- Phase 1: Load HF model and precompute all outputs ---
    print("-> Loading HF model...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        hf_preset,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"   HF model loaded: {hf_params:,} params")

    print("\n-> Precomputing all HF outputs...")
    hf_results = precompute_hf_outputs(hf_model, hf_tokenizer, hf_preset)
    hf_results["hf_param_count"] = hf_params
    print("   HF outputs precomputed!")

    # --- Phase 2: Free HF model to reclaim memory ---
    print("\n-> Releasing HF model to free memory...")
    del hf_model
    del hf_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("   HF model released.")

    # --- Phase 3: Load KerasHub model ---
    print("\n-> Loading KerasHub model from HF preset...")
    keras_model = keras_hub.models.SmolVLM2CausalLM.from_preset(
        f"hf://{hf_preset}", dtype="float32"
    )
    print("   KerasHub model loaded!")

    # --- Phase 4: Validate against precomputed HF outputs ---
    test_parameter_count(keras_model.backbone, hf_results["hf_param_count"])
    validate_text_output(keras_model, hf_results, preset)
    validate_weight_parity(keras_model, hf_results, preset)
    validate_e2e_parity(keras_model, hf_results, preset)
    generation_smoke(keras_model, hf_results)

    # --- Phase 5: Save preset ---
    save_preset(keras_model, preset)

    print("\n=== Done! ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
