import gc
import os
import random

import numpy as np
import requests
import torch
import transformers
from absl import app
from absl import flags
from keras import ops
from PIL import Image

import keras_hub
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.t5gemma2.t5gemma2_seq_2_seq_lm import (
    T5Gemma2Seq2SeqLM,
)
from keras_hub.src.models.t5gemma2.t5gemma2_seq_2_seq_lm_preprocessor import (
    T5Gemma2Seq2SeqLMPreprocessor,
)

random.seed(123)
torch.manual_seed(123)
device = torch.device("cpu")
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)

PRESET_MAP = {
    "t5gemma2_270m_270m": "google/t5gemma-2-270m-270m",
    "t5gemma2_1b_1b": "google/t5gemma-2-1b-1b",
    "t5gemma2_4b_4b": "google/t5gemma-2-4b-4b",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
)


def check_text_output(
    keras_hub_model,
    hf_tokenizer,
    hf_model,
    preprocessor,
):
    """Check outputs of KerasHub and HuggingFace models match."""
    # Note: KerasHub counts encoder + decoder embeddings as separate
    # weight matrices. HF shares a single nn.Embedding across
    # encoder/decoder/lm_head, so counts it once.
    print("\n--- Model Verification starts ---")
    print("\n")
    print("\n-> Verify parameter counts.")
    keras_hub_params = keras_hub_model.count_params()
    hf_params = hf_model.num_parameters()
    print(f"KerasHub params: {keras_hub_params:,}")
    print(f"HF params:       {hf_params:,}")
    if keras_hub_params == hf_params:
        print("-> Parameter counts match!")
    else:
        diff = keras_hub_params - hf_params
        print(
            f"-> Parameter count difference: {diff:,} "
            f"(expected — KerasHub has separate encoder/decoder "
            f"embeddings; HF shares a single nn.Embedding)"
        )

    # Output comparison.
    print("\n-> ---- Text-only verification. ----\n")
    enc_sample_text = [
        "cricket is awesome, easily the best sport in the world!"
    ]
    dec_sample_text = [
        "football is good too, but nowhere near as good as cricket."
    ]

    # KerasHub — use preprocessor natively
    # This automatically prepends start_token_id, appends EOS tokens, pads,
    # and generates dummy vision variables automatically.
    x, y, sample_weight = preprocessor(
        {
            "encoder_text": enc_sample_text,
            "decoder_text": dec_sample_text,
        }
    )
    keras_hub_inputs = x
    keras_hub_output = keras_hub_model.predict(keras_hub_inputs)

    # HF — align sequence lengths EXACTLY with KerasHub preprocessor output
    def to_torch(x):
        return torch.tensor(ops.convert_to_numpy(x), dtype=torch.long)

    hf_enc_inputs = {
        "input_ids": to_torch(keras_hub_inputs["encoder_token_ids"]),
        "attention_mask": to_torch(keras_hub_inputs["encoder_padding_mask"]),
    }
    hf_decoder_input_ids = to_torch(keras_hub_inputs["decoder_token_ids"])
    hf_decoder_attention_mask = to_torch(
        keras_hub_inputs["decoder_padding_mask"]
    )

    hf_output = hf_model(
        **hf_enc_inputs,
        decoder_input_ids=hf_decoder_input_ids,
        decoder_attention_mask=hf_decoder_attention_mask,
        output_hidden_states=True,
    )

    # Encoder output comparison.
    keras_enc_out = keras_hub_output["encoder_sequence_output"]
    hf_enc_out = hf_output.encoder_last_hidden_state.detach().float().numpy()

    # Slice to unpadded length to avoid padding token noise
    enc_valid_len = int(
        ops.convert_to_numpy(keras_hub_inputs["encoder_padding_mask"][0]).sum()
    )
    keras_enc_out = keras_enc_out[:, :enc_valid_len, :]
    hf_enc_out = hf_enc_out[:, :enc_valid_len, :]

    enc_abs_diff = np.abs(keras_enc_out - hf_enc_out)
    print()
    print("Encoder Outputs:")
    print("KerasHub output:", keras_enc_out[0, 0, :10])
    print("HF output:", hf_enc_out[0, 0, :10])
    print(f"Mean absolute diff: {enc_abs_diff.mean():.6f}")
    try:
        np.testing.assert_allclose(
            keras_enc_out, hf_enc_out, rtol=1e-4, atol=1e-4
        )
        print("-> Encoder outputs match! (rtol=1e-4, atol=1e-4)")
    except AssertionError:
        mismatch = np.sum(
            ~np.isclose(keras_enc_out, hf_enc_out, rtol=1e-4, atol=1e-4)
        )
        total = keras_enc_out.size
        print(
            f"-> Encoder outputs differ slightly beyond rtol=1e-4 "
            f"(mismatched: {mismatch}/{total}, "
            f"{mismatch / total * 100:.2f}%)"
        )

    # Decoder output comparison.
    keras_dec_out = keras_hub_output["decoder_sequence_output"]
    hf_dec_out = hf_output.decoder_hidden_states[-1].detach().float().numpy()

    # Slice to unpadded length to avoid padding token noise
    dec_valid_len = int(
        ops.convert_to_numpy(keras_hub_inputs["decoder_padding_mask"][0]).sum()
    )
    keras_dec_out = keras_dec_out[:, :dec_valid_len, :]
    hf_dec_out = hf_dec_out[:, :dec_valid_len, :]

    dec_abs_diff = np.abs(keras_dec_out - hf_dec_out)
    print()
    print("Decoder Outputs:")
    print("KerasHub output:", keras_dec_out[0, 0, :10])
    print("HF output:", hf_dec_out[0, 0, :10])
    print(f"Mean absolute diff: {dec_abs_diff.mean():.6f}")
    try:
        np.testing.assert_allclose(
            keras_dec_out, hf_dec_out, rtol=1e-4, atol=1e-4
        )
        print("-> Decoder outputs match! (rtol=1e-4, atol=1e-4)")
    except AssertionError:
        mismatch = np.sum(
            ~np.isclose(keras_dec_out, hf_dec_out, rtol=1e-4, atol=1e-4)
        )
        total = keras_dec_out.size
        print(
            f"-> Decoder outputs differ slightly beyond rtol=1e-4 "
            f"(mismatched: {mismatch}/{total}, "
            f"{mismatch / total * 100:.2f}%)"
        )


def check_multimodal_output(
    keras_hub_model,
    hf_model,
    hf_model_name,
    hf_tokenizer,
):
    """Check multimodal (text+image) outputs match between KerasHub and HF."""
    if keras_hub_model.vision_encoder is None:
        print("\n-> Skipping multimodal check (text-only model).")
        return

    print("\n-> ---- Multimodal (text+image) verification. ----\n")

    # Download a test image.
    image_url = (
        "https://huggingface.co/datasets/huggingface/"
        "documentation-images/resolve/main/bee.jpg"
    )
    print(f"  Downloading test image: {image_url}")
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    # HF side: use AutoProcessor for proper multimodal preprocessing.
    hf_processor = transformers.AutoProcessor.from_pretrained(hf_model_name)
    enc_prompt = "<start_of_image> Describe this image"
    dec_prompt = "This image shows"

    # HF encoder inputs (with image).
    hf_enc_inputs = hf_processor(
        text=enc_prompt, images=image, return_tensors="pt"
    )
    # HF decoder inputs.
    hf_dec_inputs = hf_tokenizer(dec_prompt, return_tensors="pt")
    hf_decoder_input_ids = hf_dec_inputs["input_ids"]
    hf_decoder_attention_mask = hf_dec_inputs["attention_mask"]

    with torch.no_grad():
        hf_output = hf_model(
            input_ids=hf_enc_inputs["input_ids"],
            attention_mask=hf_enc_inputs["attention_mask"],
            pixel_values=hf_enc_inputs["pixel_values"],
            decoder_input_ids=hf_decoder_input_ids,
            decoder_attention_mask=hf_decoder_attention_mask,
            output_hidden_states=True,
        )

    # Build KerasHub inputs from HF token_ids (same tokenizer).
    keras_enc_token_ids = hf_enc_inputs["input_ids"].numpy()
    keras_enc_padding_mask = hf_enc_inputs["attention_mask"].numpy()
    keras_dec_token_ids = hf_decoder_input_ids.numpy()
    keras_dec_padding_mask = hf_decoder_attention_mask.numpy()

    # Transpose HF pixel_values (B,C,H,W) to KerasHub (B,1,H,W,C).
    pixel_values = hf_enc_inputs["pixel_values"].numpy()
    if pixel_values.ndim == 5:
        pixel_values = np.transpose(pixel_values, (0, 1, 3, 4, 2))
    elif pixel_values.ndim == 4:
        pixel_values = np.transpose(pixel_values, (0, 2, 3, 1))
        pixel_values = np.expand_dims(pixel_values, axis=1)

    # Find positions of image placeholder tokens for vision_indices.
    image_token_id = hf_processor.tokenizer.convert_tokens_to_ids(
        "<image_soft_token>"
    )
    num_vision_tokens = (
        keras_hub_model.vision_encoder.num_vision_tokens_per_image
    )
    # Find indices of image placeholder tokens.
    token_ids_flat = keras_enc_token_ids[0]
    vision_idx_list = np.where(token_ids_flat == image_token_id)[0].tolist()

    # Pad or truncate to num_vision_tokens.
    if len(vision_idx_list) < num_vision_tokens:
        vision_idx_list = vision_idx_list + [0] * (
            num_vision_tokens - len(vision_idx_list)
        )
    vision_indices = np.array(
        [vision_idx_list[:num_vision_tokens]], dtype="int32"
    )

    keras_hub_inputs = {
        "encoder_token_ids": keras_enc_token_ids,
        "encoder_padding_mask": keras_enc_padding_mask,
        "decoder_token_ids": keras_dec_token_ids,
        "decoder_padding_mask": keras_dec_padding_mask,
        "images": pixel_values.astype("float32"),
        "vision_indices": vision_indices,
    }

    print("\n--- Multimodal Verification ---")
    keras_hub_output = keras_hub_model.predict(keras_hub_inputs)

    # Encoder output comparison.
    keras_enc_out = keras_hub_output["encoder_sequence_output"]
    hf_enc_out = hf_output.encoder_last_hidden_state.detach().float().numpy()
    enc_abs_diff = np.abs(keras_enc_out - hf_enc_out)
    print()
    print("Encoder Outputs (multimodal):")
    print("KerasHub output:", keras_enc_out[0, 0, :10])
    print("HF output:", hf_enc_out[0, 0, :10])
    print(f"Mean absolute diff: {enc_abs_diff.mean():.6f}")
    try:
        np.testing.assert_allclose(
            keras_enc_out, hf_enc_out, rtol=1e-4, atol=1e-4
        )
        print("-> Encoder outputs match! (rtol=1e-4, atol=1e-4)")
    except AssertionError:
        mismatch = np.sum(
            ~np.isclose(keras_enc_out, hf_enc_out, rtol=1e-4, atol=1e-4)
        )
        total = keras_enc_out.size
        print(
            f"-> Encoder outputs differ slightly beyond rtol=1e-4 "
            f"(mismatched: {mismatch}/{total}, "
            f"{mismatch / total * 100:.2f}%)"
        )

    # Decoder output comparison.
    keras_dec_out = keras_hub_output["decoder_sequence_output"]
    hf_dec_out = hf_output.decoder_hidden_states[-1].detach().float().numpy()
    dec_abs_diff = np.abs(keras_dec_out - hf_dec_out)
    print()
    print("Decoder Outputs (multimodal):")
    print("KerasHub output:", keras_dec_out[0, 0, :10])
    print("HF output:", hf_dec_out[0, 0, :10])
    print(f"Mean absolute diff: {dec_abs_diff.mean():.6f}")
    try:
        np.testing.assert_allclose(
            keras_dec_out, hf_dec_out, rtol=1e-4, atol=1e-4
        )
        print("-> Decoder outputs match! (rtol=1e-4, atol=1e-4)")
    except AssertionError:
        mismatch = np.sum(
            ~np.isclose(keras_dec_out, hf_dec_out, rtol=1e-4, atol=1e-4)
        )
        total = keras_dec_out.size
        print(
            f"-> Decoder outputs differ slightly beyond rtol=1e-4 "
            f"(mismatched: {mismatch}/{total}, "
            f"{mismatch / total * 100:.2f}%)"
        )


def main(_):
    preset = FLAGS.preset
    hf_model_name = PRESET_MAP[preset]
    os.makedirs(preset, exist_ok=True)

    # Load HF model for output verification.
    print("\n-> Load HF model and tokenizer.")
    hf_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
    hf_model.float()  # Convert all params/buffers to float32.
    # Re-create embed_scale with true f32 precision (bf16-init artifact).
    enc_hdim = hf_model.config.encoder.text_config.hidden_size
    dec_hdim = hf_model.config.decoder.hidden_size
    hf_model.model.encoder.text_model.embed_tokens.embed_scale = torch.tensor(
        enc_hdim**0.5, dtype=torch.float32
    )
    hf_model.model.decoder.embed_tokens.embed_scale = torch.tensor(
        dec_hdim**0.5, dtype=torch.float32
    )
    hf_model.eval()
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    # Load KerasHub model via auto-loader (uses convert_t5gemma2.py).
    print("\n-> Loading KerasHub model from HuggingFace preset.")
    keras_hub_model = keras_hub.models.T5Gemma2Backbone.from_preset(
        f"hf://{hf_model_name}", dtype="float32"
    )

    # Create preprocessor for verification.
    preprocessor_kwargs = {}
    if keras_hub_model.vision_encoder is not None:
        preprocessor_kwargs.update(
            {
                "image_size": keras_hub_model.vision_encoder.image_size,
                "num_vision_tokens_per_image": (
                    keras_hub_model.vision_encoder.num_vision_tokens_per_image
                ),
            }
        )
    preprocessor = T5Gemma2Seq2SeqLMPreprocessor.from_preset(
        f"hf://{hf_model_name}",
        **preprocessor_kwargs,
    )

    check_text_output(
        keras_hub_model,
        hf_tokenizer,
        hf_model,
        preprocessor,
    )

    check_multimodal_output(
        keras_hub_model,
        hf_model,
        hf_model_name,
        hf_tokenizer,
    )

    print("\n-> Releasing HF backbone from memory.")
    del hf_model
    gc.collect()

    keras_lm = T5Gemma2Seq2SeqLM(
        backbone=keras_hub_model,
        preprocessor=preprocessor,
        dtype=keras_hub_model.dtype,
    )

    print(f"\n-> Saving T5Gemma2Seq2SeqLM preset to `{preset}`.")
    keras_lm.save_to_preset(preset)
    print("-> Preset saved successfully.")

    print("\n-> Testing preset loading.")
    keras_lm = Seq2SeqLM.from_preset(preset)
    print("-> Preset loading verified successfully.")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
