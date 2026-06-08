"""Convert BLIP-2 checkpoints (OPT and Flan-T5) to KerasHub format.

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

import keras_hub  # noqa: E402
from keras_hub.src.models.causal_lm import CausalLM  # noqa: E402

PRESET_MAP = {
    "blip2_opt_2_7b": "Salesforce/blip2-opt-2.7b",
    "blip2_opt_6_7b": "Salesforce/blip2-opt-6.7b",
    "blip2_flan_t5_xl": "Salesforce/blip2-flan-t5-xl",
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


def validate_output(keras_lm, hf_model, hf_processor, image):
    """Compare parameter counts and greedy generation vs HuggingFace."""
    print("\n-> Comparing parameter counts.")
    keras_params = keras_lm.backbone.count_params()
    hf_params = hf_model.num_parameters()
    print(f"   KerasHub backbone params : {keras_params:,}")
    print(f"   HuggingFace total params : {hf_params:,}")
    print(
        "   (counts may differ: HF counts a separate lm_head / tied "
        "embeddings, and KerasHub pads the vocabulary.)"
    )

    print("\n-> Comparing greedy generation.")
    max_new_tokens = 20

    hf_inputs = hf_processor(
        images=image, text=_PROMPT, return_tensors="pt", padding=False
    )
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

    print(f"\n-> Loading HF model: {hf_model_name}")
    hf_model = Blip2ForConditionalGeneration.from_pretrained(
        hf_model_name, torch_dtype=torch.float32
    )
    hf_model.eval()
    hf_processor = Blip2Processor.from_pretrained(hf_model_name)

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
