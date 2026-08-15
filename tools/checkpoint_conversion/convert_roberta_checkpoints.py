"""Convert RoBERTa checkpoints from Hugging Face to KerasHub format.

RoBERTa's Hugging Face converter (`keras_hub/src/utils/transformers/
convert_roberta.py`) is registered with the general `from_preset()` loading
path, so this script does not port weights manually. It loads the backbone
and tokenizer straight from the Hugging Face preset, checks numerical parity
against the reference Hugging Face model, and saves the result as a
KerasHub preset.

Usage:
```shell
cd tools/checkpoint_conversion
python convert_roberta_checkpoints.py --preset roberta_base_en
```
"""

import numpy as np
import tensorflow as tf
import transformers
from absl import app
from absl import flags

import keras_hub

PRESET_MAP = {
    "roberta_base_en": "FacebookAI/roberta-base",
    "roberta_large_en": "FacebookAI/roberta-large",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def check_output(
    keras_hub_model,
    keras_hub_preprocessor,
    hf_model,
    hf_tokenizer,
):
    print("\n-> Check the outputs.")
    input_str = ["the quick brown fox ran, galloped and jumped."]

    # KerasHub
    keras_hub_inputs = keras_hub_preprocessor(tf.constant(input_str))
    keras_hub_output = keras_hub_model.predict(keras_hub_inputs)

    # HF
    hf_inputs = hf_tokenizer(
        input_str, padding="max_length", return_tensors="pt"
    )
    hf_output = hf_model(**hf_inputs).last_hidden_state

    print("KerasHub output:", keras_hub_output[0, 0, :10])
    print("HF output:", hf_output[0, 0, :10])
    print("Difference:", np.mean(keras_hub_output - hf_output.detach().numpy()))


def main(_):
    assert FLAGS.preset in PRESET_MAP.keys(), (
        f"Invalid preset {FLAGS.preset}. "
        f"Must be one of {','.join(PRESET_MAP.keys())}"
    )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    print(f"\n-> Convert {hf_preset} to KerasHub format.")
    # `RobertaBackbone`/`RobertaTokenizer.from_preset` dispatch through
    # `convert_roberta.py`, which handles config translation and weight
    # porting directly from the Hugging Face safetensors checkpoint.
    keras_hub_model = keras_hub.models.RobertaBackbone.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_tokenizer = keras_hub.models.RobertaTokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_preprocessor = keras_hub.models.RobertaTextClassifierPreprocessor(
        keras_hub_tokenizer
    )

    print("\n-> Load HF model.")
    hf_model = transformers.AutoModel.from_pretrained(hf_preset)
    hf_model.eval()
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_preset)

    check_output(
        keras_hub_model,
        keras_hub_preprocessor,
        hf_model,
        hf_tokenizer,
    )

    print(f"\n-> Save KerasHub preset to `./{preset}`.")
    keras_hub_model.save_to_preset(preset)
    keras_hub_tokenizer.save_to_preset(preset)


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
