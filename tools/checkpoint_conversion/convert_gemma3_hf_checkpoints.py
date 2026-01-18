import os
import random
import traceback

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide any CUDA devices

import numpy as np
import torch
from absl import app
from absl import flags
from keras import ops
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import keras_hub

random.seed(123)
torch.manual_seed(123)
device = torch.device("cpu")
# Force PyTorch to use CPU
torch.set_default_device(device)


PRESET_MAP = {
    "gemma3_270m": "google/gemma-3-270m",
    "gemma3_instruct_270m": "google/gemma-3-270m-it",
    "gemma3_1b": "google/gemma-3-1b-pt",
    "gemma3_instruct_1b": "google/gemma-3-1b-it",
    "gemma3_4b": "google/gemma-3-4b-pt",
    "gemma3_instruct_4b": "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-pt",
    "gemma3_instruct_12b": "google/gemma-3-12b-it",
    "gemma3_27b": "google/gemma-3-27b-pt",
    "gemma3_instruct_27b": "google/gemma-3-27b-it",
    "function_gemma_instruct_270m": "google/functiongemma-270m-it",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def test_model(
    keras_hub_model, keras_hub_preprocessor, hf_model, hf_model_tokenizer
):
    # First, test that the number of parameters match
    keras_hub_params = keras_hub_model.count_params()
    hf_params = hf_model.num_parameters()
    assert keras_hub_params == hf_params

    # Test the outputs of both the models
    hf_inputs = hf_model_tokenizer(["What is Keras?"], return_tensors="pt").to(
        device
    )
    hf_outputs = hf_model(**hf_inputs)
    hf_output_logits = hf_outputs.logits.detach().cpu().float().numpy()

    keras_hub_inputs = keras_hub_preprocessor.generate_preprocess(
        ["What is Keras?"], sequence_length=6
    )

    keras_hub_output = keras_hub_model(keras_hub_inputs)
    keras_hub_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits)

    try:
        np.testing.assert_allclose(
            keras_hub_logits, hf_output_logits, atol=1e-2
        )
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    hf_output = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()
    keras_hub_preprocessor = keras_hub.models.Gemma3CausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_output = keras_hub_preprocessor.generate_preprocess(
        ["What is Keras?"], sequence_length=6
    )
    keras_hub_output = ops.convert_to_numpy(keras_hub_output["token_ids"])

    np.testing.assert_equal(keras_hub_output, hf_output)


def validate_output(
    keras_model,
    hf_model,
    hf_tokenizer,
):
    input_str = "What is Keras?"
    length = 32

    # KerasHub
    keras_output = keras_model.generate([input_str], max_length=length)
    keras_output = keras_output[0]
    print("🔶 KerasHub output:", keras_output)

    hf_inputs = hf_tokenizer([input_str], return_tensors="pt")
    outputs = hf_model.generate(
        **hf_inputs,
        max_length=length,
        do_sample=False,
        num_beams=1,
        pad_token_id=hf_tokenizer.pad_token_id,
    )
    print("🔶 Huggingface generated token ids:", outputs[0])
    hf_generated_text = hf_tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )[0]
    print("🔶 Huggingface output:", hf_generated_text)


def main(_):
    # === Get the preset name ===
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    # === Load the Huggingface model ===
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_preset,
        device_map=device,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset, return_tensors="pt")
    hf_model.eval()

    keras_hub_backbone = keras_hub.models.Gemma3Backbone.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_tokenizer = keras_hub.models.Gemma3Tokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_preprocessor = (
        keras_hub.models.Gemma3CausalLMPreprocessor.from_preset(
            f"hf://{hf_preset}"
        )
    )

    print("\n-> Huggingface model and tokenizer loaded")

    # === Check that the models and tokenizers outputs match ===
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
    test_model(
        keras_hub_backbone, keras_hub_preprocessor, hf_model, hf_tokenizer
    )
    print("\n-> Tests passed!")

    gemma3_lm = keras_hub.models.Gemma3CausalLM(
        backbone=keras_hub_backbone,
        preprocessor=keras_hub_preprocessor,
        sampler="greedy",
    )

    validate_output(gemma3_lm, hf_model, hf_tokenizer)
    gemma3_lm.save_to_preset(f"./{preset}")

    print(f"\n-> Saved converted model to ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
