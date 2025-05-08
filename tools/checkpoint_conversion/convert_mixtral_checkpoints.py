import os
import traceback

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide any CUDA devices

import numpy as np
import torch
from absl import app
from absl import flags

device = torch.device("cpu")
# Force PyTorch to use CPU
torch.set_default_device(device)

from keras import ops  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import keras_hub  # noqa: E402

PRESET_MAP = {
    "mixtral_8_7b_en": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral_8_instruct_7b_en": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def compute_hf_output(hf_model, hf_model_tokenizer):
    hf_inputs = hf_model_tokenizer(["What is Keras?"], return_tensors="pt").to(
        device
    )
    hf_outputs = hf_model(**hf_inputs)
    hf_output_logits = hf_outputs.logits.detach().cpu().float().numpy()

    return hf_output_logits


def compute_keras_output(keras_hub_model, keras_hub_tokenizer):
    keras_hub_preprocessor = keras_hub.models.MixtralCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_inputs = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=6
    )[0]
    keras_hub_inputs = {k: v.to(device) for k, v in keras_hub_inputs.items()}

    keras_hub_output = keras_hub_model(keras_hub_inputs)
    keras_hub_output_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_output_logits = ops.convert_to_numpy(keras_hub_output_logits)
    return keras_hub_output_logits


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    hf_output = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()
    keras_hub_preprocessor = keras_hub.models.MixtralCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_output = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=6
    )
    keras_hub_output = ops.convert_to_numpy(keras_hub_output[0]["token_ids"])

    np.testing.assert_equal(keras_hub_output, hf_output)


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
    print("\n-> Huggingface model and tokenizer loaded")

    keras_hub_tokenizer = keras_hub.models.MixtralTokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    print("\n-> Keras tokenizer loaded")
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)

    print("\n -> Keras tokenizer test successful")

    hf_params = hf_model.num_parameters()
    hf_output_logits = compute_hf_output(hf_model, hf_tokenizer)
    print("\n -> Computed HF outputs successfully")

    del hf_model, hf_tokenizer
    keras_hub_backbone = keras_hub.models.MixtralBackbone.from_preset(
        f"hf://{hf_preset}"
    )
    print("\n-> Keras model loaded")

    keras_hub_params = keras_hub_backbone.count_params()
    assert keras_hub_params == hf_params

    keras_hub_output_logits = compute_keras_output(
        keras_hub_backbone, keras_hub_tokenizer
    )

    try:
        np.testing.assert_allclose(
            keras_hub_output_logits, hf_output_logits, atol=1e-4
        )
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")

    print("\n-> Tests passed!")

    preprocessor = keras_hub.models.MixtralCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_model = keras_hub.models.MixtralCausalLM(
        keras_hub_backbone, preprocessor
    )

    keras_hub_model.save_to_preset(f"./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
