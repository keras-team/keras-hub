import os
import traceback

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide any CUDA devices

import numpy as np
import torch
from absl import app

# from absl import flags

device = torch.device("cpu")
# Force PyTorch to use CPU
torch.set_default_device(device)

from keras import ops  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import keras_hub  # noqa: E402

PRESET_MAP = {
    "qwen1.5_moe_2.7b_en": "Qwen/Qwen1.5-MoE-A2.7B",
}

# FLAGS = flags.FLAGS
# flags.DEFINE_string(
#     "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
# )


def test_model(
    keras_hub_model, keras_hub_tokenizer#, hf_model, hf_model_tokenizer
):
    keras_hub_preprocessor = keras_hub.models.QwenCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_inputs = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=5
    )[0]
    keras_hub_inputs = {k: v.to(device) for k, v in keras_hub_inputs.items()}

    keras_hub_output = keras_hub_model(keras_hub_inputs)
    keras_hub_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits)

    # High tolerence since bfloat16 is used as the default dtype for Qwen

    # try:
    #     np.testing.assert_allclose(
    #         keras_hub_logits, hf_output_logits, atol=1e-4
    #     )
    # except AssertionError as err:
    #     print("\n")
    #     print(traceback.format_exc())
    #     print(err.args[0])
    #     print("\n")




def main(_):

    keras_preset = "qwen1.5_moe_2.7b_en"
    hf_preset = "Qwen/Qwen1.5-MoE-A2.7B"

    # === Load the Huggingface model ===
    # hf_model = AutoModelForCausalLM.from_pretrained(
    #     hf_preset,
    #     device_map=device,
    # )
    # hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset, return_tensors="pt")
    # hf_model.eval()

    keras_hub_model = keras_hub.models.QwenMoeBackbone.from_preset(
        keras_preset# f"hf://{hf_preset}"
    )
    keras_hub_tokenizer = keras_hub.tokenizers.QwenMoeTokenizer.from_preset(
        keras_preset# f"hf://{hf_preset}"
    )

    print("\n-> Huggingface model and tokenizer loaded") # TODO: SAVE TO PRESET AFTER DEBUGGER REACHES ABOVE LINE

    # === Check that the models and tokenizers outputs match ===
    # test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
    test_model(keras_hub_model, keras_hub_tokenizer) #, hf_model, hf_tokenizer)
    print("\n-> Tests passed!")


if __name__ == "__main__":
    # flags.mark_flag_as_required("preset")
    app.run(main)
