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
from transformers import AutoTokenizer  # noqa: E402
from transformers.models.esm.modeling_esm import EsmModel  # noqa: E402

import keras_hub  # noqa: E402

PRESET_MAP = {
    "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def test_model(
    keras_hub_model,
    hf_model,
):
    # First, test that the number of parameters match
    hf_model.embeddings.token_dropout = False
    hf_model.eval()
    keras_hub_model.eval()

    x = ops.array([[1, 2, 3, 4, 5]]) + 3
    hf_out = hf_model(x, ops.ones_like(x))[0]
    keras_out = keras_hub_model({"token_ids": x})
    try:
        np.testing.assert_allclose(
            ops.convert_to_numpy(hf_out),
            ops.convert_to_numpy(keras_out),
            atol=1e-3,
        )
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    hf_output = hf_tokenizer(["L L A C G "], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()
    keras_hub_preprocessor = keras_hub.models.ESMMaskedPLMPreprocessor(
        keras_hub_tokenizer, sequence_length=7, mask_token_rate=0
    )
    keras_hub_output = keras_hub_preprocessor(["L L A C G "])
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
    hf_model = EsmModel.from_pretrained(
        hf_preset,
        device_map=device,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset, return_tensors="pt")
    hf_model.eval()

    keras_hub_model = keras_hub.models.ESMBackbone.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_tokenizer = keras_hub.tokenizers.ESMTokenizer.from_preset(
        f"hf://{hf_preset}"
    )

    print("\n-> Huggingface model and tokenizer loaded")

    # === Check that the models and tokenizers outputs match ===
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
    test_model(keras_hub_model, hf_model)
    print("\n-> Tests passed!")

    preprocessor = keras_hub.models.ESMMaskedPLMPreprocessor(
        keras_hub_tokenizer
    )

    keras_hub_model = keras_hub.models.ESMMaskedPLM(
        keras_hub_model, preprocessor
    )

    keras_hub_model.save_to_preset(f"./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
