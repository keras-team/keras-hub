"""
Quantize preset checkpoints with dynamic int8 and optionally upload the quantized preset.

Usage:
export KERAS_BACKEND=jax CUDA_VISIBLE_DEVICES=
python tools/quantize_checkpoints.py --preset llama3_8b_en
python tools/quantize_checkpoints.py --preset llama3_8b_en --upload_uri kaggle://keras/llama3/keras/llama3_8b_en_int8
"""

import keras
from absl import app
from absl import flags

import keras_hub

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "preset",
    None,
    "Must be a valid `CausalLM` preset from KerasHub",
    required=True,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/{variant}/keras/{preset}_int8"',
    required=False,
)


def validate_output(causal_lm):
    input_str = "What is Keras?"
    length = 32

    keras_output = causal_lm.generate([input_str], max_length=length)
    keras_output = keras_output[0]
    print("🔶 KerasHub output:", keras_output)


def main(_):
    preset = FLAGS.preset
    upload_uri = FLAGS.upload_uri
    print(f"🏃 Quantizing {preset}")

    keras.config.set_floatx("bfloat16")

    causal_lm = keras_hub.models.CausalLM.from_preset(preset, dtype="bfloat16")
    backbone = causal_lm.backbone
    tokenizer = causal_lm.preprocessor.tokenizer

    backbone.quantize("int8")
    print("✅ Weights quantized")

    causal_lm.backbone = backbone
    validate_output(causal_lm)
    print("✅ Output validated")

    quantized_preset = f"{preset}_int8"
    backbone.save_to_preset(quantized_preset)
    tokenizer.save_to_preset(quantized_preset)
    print(f"🏁 Preset saved to ./{quantized_preset}")

    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=quantized_preset)
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
