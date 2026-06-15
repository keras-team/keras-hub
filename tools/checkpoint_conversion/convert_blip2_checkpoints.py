"""Convert BLIP-2 and InstructBLIP checkpoints to KerasHub format.

Covers the OPT, Flan-T5, and Vicuna (InstructBLIP) language-model variants.

Weight mapping lives in ``keras_hub/src/utils/transformers/convert_blip2.py``,
so this script simply loads the model through ``from_preset("hf://...")``,
validates the outputs against HuggingFace, and saves a KerasHub preset.

Usage:
```shell
python convert_blip2_checkpoints.py --preset blip2_opt_2_7b
python convert_blip2_checkpoints.py --preset blip2_flan_t5_xl
```
"""

import gc
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import Blip2ForConditionalGeneration  # noqa: E402
from transformers import Blip2Processor  # noqa: E402
from transformers import InstructBlipForConditionalGeneration  # noqa: E402
from transformers import InstructBlipProcessor  # noqa: E402

import keras_hub  # noqa: E402
from keras_hub.src.models.causal_lm import CausalLM  # noqa: E402

PRESET_MAP = {
    "blip2_opt_2_7b": "Salesforce/blip2-opt-2.7b",
    "blip2_opt_6_7b": "Salesforce/blip2-opt-6.7b",
    "blip2_flan_t5_xl": "Salesforce/blip2-flan-t5-xl",
    "blip2_flan_t5_xxl": "Salesforce/blip2-flan-t5-xxl",
    "instructblip_vicuna_7b": "Salesforce/instructblip-vicuna-7b",
}

_PROMPT = "Question: what is in the picture? Answer:"
_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/"
    "documentation-images/resolve/main/bee.jpg"
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
)


def _to_np(tensor):
    """Detach a torch tensor (or array-like) to a float32 numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def _report_diff(label, keras_out, hf_out, atol=1e-4, rtol=1e-4):
    """Print a per-component KerasHub vs HuggingFace comparison.

    Mirrors the validation style used by other conversion scripts
    (e.g. ``convert_t5gemma2_checkpoints.py``): the first few elements of
    each tensor, the mean / max absolute difference, and an
    ``atol``/``rtol`` tolerance check that degrades gracefully to a
    mismatch percentage instead of crashing.
    """
    keras_out = _to_np(keras_out)
    hf_out = _to_np(hf_out)
    abs_diff = np.abs(keras_out - hf_out)
    print(f"\n-> {label}")
    print(f"   KerasHub shape : {keras_out.shape}")
    print(f"   HF       shape : {hf_out.shape}")
    print(f"   KerasHub[:5]   : {keras_out.reshape(-1)[:5]}")
    print(f"   HF      [:5]   : {hf_out.reshape(-1)[:5]}")
    print(f"   mean abs diff  : {abs_diff.mean():.8f}")
    print(f"   max  abs diff  : {abs_diff.max():.8f}")
    if np.allclose(keras_out, hf_out, atol=atol, rtol=rtol):
        print(f"   -> match (atol={atol}, rtol={rtol})")
    else:
        mismatch = int(
            np.sum(~np.isclose(keras_out, hf_out, atol=atol, rtol=rtol))
        )
        total = keras_out.size
        print(
            f"   -> differs beyond atol={atol}, rtol={rtol} "
            f"(mismatched {mismatch}/{total}, {mismatch / total * 100:.2f}%)"
        )


def validate_output(keras_lm, hf_model, hf_processor, image):
    """Compare parameter counts, per-component outputs, and generation.

    The per-component checks (vision encoder, Q-Former, language-model
    hidden states, and lm_head logits) feed *identical* normalized pixels
    and token ids to both frameworks, so the reported difference reflects
    only weight-conversion error and not preprocessing or tokenization.
    """
    backbone = keras_lm.backbone
    lm = backbone.language_model
    is_encoder_decoder = hasattr(lm, "encoder_transformer_layers")
    instruction_aware = getattr(backbone.qformer, "instruction_aware", False)
    if is_encoder_decoder:
        lm_name = "Flan-T5"
    elif instruction_aware:
        lm_name = "Vicuna"
    else:
        lm_name = "OPT"

    print("\n-> Comparing parameter counts.")
    keras_params = keras_lm.backbone.count_params()
    hf_params = hf_model.num_parameters()
    print(f"   KerasHub backbone params : {keras_params:,}")
    print(f"   HuggingFace total params : {hf_params:,}")
    print(
        "   (counts may differ: HF counts a separate lm_head / tied "
        "embeddings, and KerasHub pads the vocabulary.)"
    )

    print("\n-> Comparing per-component outputs vs HuggingFace.")
    # Feed both frameworks identical inputs: HF-normalized pixels (NCHW->NHWC)
    # and HF token ids. This isolates weight-conversion error from any
    # preprocessing / tokenization differences.
    hf_inputs = hf_processor(
        images=image, text=_PROMPT, return_tensors="pt", padding=False
    )
    # Newer `Blip2Processor` versions expand `input_ids` with image-placeholder
    # tokens (one per query token); HuggingFace swaps the visual embeddings in
    # place. KerasHub's backbone prepends the visual prompt itself, so feed it
    # the text tokens only. (Older processors add no placeholders, in which case
    # this filter is a no-op.)
    input_ids = hf_inputs["input_ids"]
    attention_mask = hf_inputs["attention_mask"]
    image_token_id = getattr(
        hf_model.config,
        "image_token_index",
        getattr(hf_model.config, "image_token_id", None),
    )
    if image_token_id is not None:
        keep = input_ids[0] != image_token_id
        input_ids = input_ids[:, keep]
        attention_mask = attention_mask[:, keep]
    token_ids = input_ids.numpy().astype("int32")
    padding_mask = attention_mask.numpy().astype("int32")
    pixel_values = hf_inputs["pixel_values"]
    keras_images = np.transpose(_to_np(pixel_values), (0, 2, 3, 1))

    # InstructBLIP feeds the instruction to the Q-Former through a second
    # (BERT) tokenization. Reuse HF's `qformer_input_ids` so the per-component
    # comparison isolates weight-conversion error.
    qformer_token_ids = qformer_padding_mask = None
    if instruction_aware:
        qformer_token_ids = (
            hf_inputs["qformer_input_ids"].numpy().astype("int32")
        )
        qformer_padding_mask = (
            hf_inputs["qformer_attention_mask"].numpy().astype("int32")
        )

    # Single HF forward pass; intermediate outputs are read off the result.
    with torch.no_grad():
        if is_encoder_decoder:
            # Teacher-force a single decoder-start token (pad id 0 for T5).
            decoder_input_ids = torch.zeros(
                (token_ids.shape[0], 1), dtype=torch.long
            )
            hf_out = hf_model(
                **hf_inputs,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
            )
        else:
            hf_out = hf_model(**hf_inputs, output_hidden_states=True)

    # Vision encoder (ViT).
    keras_vision = backbone.vision_encoder(keras_images)
    _report_diff(
        "Vision encoder (ViT) hidden states",
        keras_vision,
        hf_out.vision_outputs.last_hidden_state,
    )

    # Q-Former.
    if instruction_aware:
        # Feed this direct sub-model call uniform numpy: `keras_vision` is a
        # backend tensor while the instruction ids are numpy, and Keras rejects
        # a dict that mixes tensors and non-tensors in a nested call.
        keras_qformer = backbone.qformer(
            {
                "vision_features": _to_np(keras_vision),
                "qformer_token_ids": qformer_token_ids,
                "qformer_padding_mask": qformer_padding_mask,
            }
        )
        # HF returns query + instruction tokens; keep only the query outputs.
        num_query = backbone.num_query_tokens
        hf_qformer = hf_out.qformer_outputs.last_hidden_state[:, :num_query, :]
    else:
        keras_qformer = backbone.qformer(keras_vision)
        hf_qformer = hf_out.qformer_outputs.last_hidden_state
    _report_diff(
        "Q-Former query outputs",
        keras_qformer,
        hf_qformer,
    )

    # Language model hidden states + lm_head logits.
    if is_encoder_decoder:
        enc_hidden, enc_mask = lm.call_encoder(
            token_ids, padding_mask, keras_qformer
        )
        decoder_input = np.zeros((token_ids.shape[0], 1), dtype="int32")
        decoder_mask = np.ones((token_ids.shape[0], 1), dtype="int32")
        keras_lm_hidden = lm.call_decoder(
            decoder_input, decoder_mask, enc_hidden, enc_mask
        )
        keras_logits = lm.lm_head(keras_lm_hidden)
        hf_lm_hidden = hf_out.language_model_outputs.decoder_hidden_states[-1]
        hf_logits = hf_out.logits
    else:
        keras_inputs = {
            "images": keras_images,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        if instruction_aware:
            keras_inputs["qformer_token_ids"] = qformer_token_ids
            keras_inputs["qformer_padding_mask"] = qformer_padding_mask
        # Backbone returns LM hidden states; the lm head is applied by the
        # CausalLM functional model (which strips the query-token prefix). The
        # text tokens are contiguous at the end of both sequences, so compare
        # the trailing `text_len` positions (robust to however many visual /
        # placeholder tokens each framework prepends).
        text_len = token_ids.shape[1]
        keras_lm_hidden = _to_np(backbone(keras_inputs))[:, -text_len:, :]
        keras_logits = _to_np(keras_lm(keras_inputs))[:, -text_len:, :]
        hf_lm_hidden = _to_np(hf_out.language_model_outputs.hidden_states[-1])[
            :, -text_len:, :
        ]
        hf_logits = _to_np(hf_out.logits)[:, -text_len:, :]

    _report_diff(
        f"{lm_name} language-model hidden states (logits input)",
        keras_lm_hidden,
        hf_lm_hidden,
    )
    _report_diff(
        f"{lm_name} lm_head logits",
        keras_logits,
        hf_logits,
    )

    print("\n-> Comparing greedy generation.")
    max_new_tokens = 20

    # `hf_inputs` (images + prompt) was built above for the component checks.
    with torch.no_grad():
        hf_generated = hf_model.generate(
            **hf_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    hf_text = hf_processor.batch_decode(hf_generated, skip_special_tokens=True)[
        0
    ].strip()

    keras_lm.compile(sampler="greedy")
    keras_output = keras_lm.generate(
        {"images": np.array(image), "text": [_PROMPT]},
        max_length=max_new_tokens + 64,
        strip_prompt=True,
    )
    keras_text = (
        keras_output[0]
        if isinstance(keras_output, (list, tuple))
        else keras_output
    ).strip()

    print(f"   HuggingFace : {hf_text!r}")
    print(f"   KerasHub    : {keras_text!r}")
    if keras_text == hf_text:
        print("-> Generation matches!")
    else:
        print("-> Generation differs (review tolerances / sampling).")


def main(_):
    preset = FLAGS.preset
    hf_model_name = PRESET_MAP[preset]
    os.makedirs(preset, exist_ok=True)

    is_instructblip = preset.startswith("instructblip")

    print(f"\n-> Loading HF model: {hf_model_name}")
    if is_instructblip:
        hf_model = InstructBlipForConditionalGeneration.from_pretrained(
            hf_model_name, torch_dtype=torch.float32
        )
        hf_processor = InstructBlipProcessor.from_pretrained(hf_model_name)
    else:
        hf_model = Blip2ForConditionalGeneration.from_pretrained(
            hf_model_name, torch_dtype=torch.float32
        )
        hf_processor = Blip2Processor.from_pretrained(hf_model_name)
    hf_model.eval()

    print("\n-> Loading KerasHub model from the HF preset.")
    # BLIP-2 / Flan-T5 must run in float32 (or bf16) — fp16 overflows to NaN.
    keras_lm = keras_hub.models.BLIP2CausalLM.from_preset(
        f"hf://{hf_model_name}", dtype="float32"
    )

    image = Image.open(requests.get(_IMAGE_URL, stream=True).raw).convert("RGB")

    validate_output(keras_lm, hf_model, hf_processor, image)

    print("\n-> Releasing HF model from memory.")
    del hf_model
    gc.collect()

    print(f"\n-> Saving KerasHub preset to `{preset}`.")
    keras_lm.save_to_preset(preset)
    print("-> Preset saved.")

    # Free the converted model before reloading so we don't hold two full
    # copies in memory at once (peaks at ~2x the model size otherwise).
    del keras_lm
    gc.collect()

    print("\n-> Verifying preset reload.")
    CausalLM.from_preset(preset)
    print("-> Preset reload verified.")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
