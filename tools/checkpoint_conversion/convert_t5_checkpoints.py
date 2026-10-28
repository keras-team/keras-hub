"""Convert T5 checkpoints.

Weight porting lives in `keras_hub/src/utils/transformers/convert_t5.py` and is
shared with `from_preset("hf://...")`. This script only loads the Hugging Face
model, converts it through that converter, validates the numerics against
Hugging Face and saves (optionally uploads) the KerasHub preset.

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_t5_checkpoints.py \
    --preset t5_small_multi --upload_uri kaggle://keras/t5/keras/t5_small_multi
python tools/checkpoint_conversion/convert_t5_checkpoints.py \
    --preset t5_1.1_small --upload_uri kaggle://keras/t5/keras/t5_1.1_small
python tools/checkpoint_conversion/convert_t5_checkpoints.py \
    --preset flan_small_multi --upload_uri kaggle://keras/t5/keras/flan_small_multi
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import shutil  # noqa: E402
import tempfile  # noqa: E402

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402

import keras_hub  # noqa: E402
from keras_hub.src.models.t5.t5_backbone import T5Backbone  # noqa: E402
from keras_hub.src.models.t5.t5_tokenizer import T5Tokenizer  # noqa: E402

PRESET_MAP = {
    "t5_small_multi": "google-t5/t5-small",
    "t5_base_multi": "google-t5/t5-base",
    "t5_large_multi": "google-t5/t5-large",
    "t5_1.1_small": "google/t5-v1_1-small",
    "t5_1.1_base": "google/t5-v1_1-base",
    "t5_1.1_large": "google/t5-v1_1-large",
    "t5_1.1_xl": "google/t5-v1_1-xl",
    "t5_1.1_xxl": "google/t5-v1_1-xxl",
    "flan_small_multi": "google/flan-t5-small",
    "flan_base_multi": "google/flan-t5-base",
    "flan_large_multi": "google/flan-t5-large",
}

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
    required=True,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/{variant}/keras/{preset}"',
    required=False,
)


def load_keras_hub_components(hf_model, hf_preset, hf_dir):
    """Load the KerasHub T5 backbone and tokenizer via the HF converter.

    Some upstream checkpoints (e.g. `google/t5-v1_1-*`) only ship
    `pytorch_model.bin`, which the safetensors-based converter cannot read.
    To support all presets, the HF model is re-saved as a local HF-format
    directory with safetensors and the backbone is loaded from there. The
    tokenizer only needs `spiece.model`, which every upstream repo provides,
    so it is loaded from the Hub directly.
    """
    hf_model.save_pretrained(hf_dir, safe_serialization=True)
    keras_model = T5Backbone.from_preset(hf_dir)
    keras_tokenizer = T5Tokenizer.from_preset(f"hf://{hf_preset}")
    return keras_model, keras_tokenizer


def validate_output(keras_model, keras_tokenizer, hf_model, hf_tokenizer):
    encoder_input = ["the quick brown fox jumped."]
    decoder_input = ["the quick brown fox fell."]
    sequence_length = 12

    # KerasHub tokenization.
    packer = keras_hub.layers.StartEndPacker(
        sequence_length=sequence_length,
        pad_value=keras_tokenizer.pad_token_id,
        end_value=keras_tokenizer.end_token_id,
    )
    encoder_token_ids = keras.ops.convert_to_numpy(
        packer(keras_tokenizer(encoder_input))
    )
    decoder_token_ids = keras.ops.convert_to_numpy(
        packer(keras_tokenizer(decoder_input))
    )
    encoder_padding_mask = encoder_token_ids != keras_tokenizer.pad_token_id
    decoder_padding_mask = decoder_token_ids != keras_tokenizer.pad_token_id

    # HF tokenization.
    hf_encoder_inputs = hf_tokenizer(
        encoder_input,
        padding="max_length",
        max_length=sequence_length,
        return_tensors="pt",
    )
    hf_decoder_inputs = hf_tokenizer(
        decoder_input,
        padding="max_length",
        max_length=sequence_length,
        return_tensors="pt",
    )
    np.testing.assert_array_equal(
        encoder_token_ids, hf_encoder_inputs["input_ids"].numpy()
    )
    np.testing.assert_array_equal(
        decoder_token_ids, hf_decoder_inputs["input_ids"].numpy()
    )
    print("🔶 Tokenization matches.")

    # Forward pass.
    keras_outputs = keras_model.predict(
        {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        },
        verbose=0,
    )
    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=hf_encoder_inputs["input_ids"],
            attention_mask=hf_encoder_inputs["attention_mask"],
            decoder_input_ids=hf_decoder_inputs["input_ids"],
            decoder_attention_mask=hf_decoder_inputs["attention_mask"],
            output_hidden_states=True,
        )

    # Only compare non-padded positions.
    encoder_mask = encoder_padding_mask[0]
    decoder_mask = decoder_padding_mask[0]
    keras_encoder = keras_outputs["encoder_sequence_output"][0][encoder_mask]
    hf_encoder = hf_outputs.encoder_last_hidden_state[0].numpy()[encoder_mask]
    keras_decoder = keras_outputs["decoder_sequence_output"][0][decoder_mask]
    hf_decoder = hf_outputs.decoder_hidden_states[-1][0].numpy()[decoder_mask]

    # Logits. Original T5 ties the LM head to the token embedding and scales
    # the decoder output; T5 v1.1 / FLAN-T5 use a separate unscaled head.
    decoder_output = keras_outputs["decoder_sequence_output"]
    if keras_model.tie_embedding_weights:
        decoder_output = decoder_output * keras_model.hidden_dim**-0.5
    keras_logits = keras.ops.convert_to_numpy(
        keras_model.token_embedding(decoder_output, reverse=True)
    )[0][decoder_mask]
    hf_logits = hf_outputs.logits[0].numpy()[decoder_mask]

    for name, keras_value, hf_value in (
        ("Encoder output", keras_encoder, hf_encoder),
        ("Decoder output", keras_decoder, hf_decoder),
        ("Logits", keras_logits, hf_logits),
    ):
        print(f"🔶 {name} Keras:", keras_value[0, :5])
        print(f"🔶 {name} HF:   ", hf_value[0, :5])
        print(
            f"🔶 {name} mean abs diff:",
            np.mean(np.abs(keras_value - hf_value)),
        )
    # HF's gated-gelu uses the tanh approximation (`gelu_new`) while KerasHub
    # uses the exact gelu, hence the loose tolerances.
    np.testing.assert_allclose(keras_decoder, hf_decoder, atol=1e-2)
    np.testing.assert_allclose(keras_logits, hf_logits, atol=1e-1)


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    hf_preset = PRESET_MAP[preset]
    if os.path.exists(preset):
        shutil.rmtree(preset)
    os.makedirs(preset)

    print(f"🏃 Converting {preset}")

    # Load the Hugging Face model. `T5ForConditionalGeneration` (rather than
    # `T5Model`) also carries `lm_head.weight` for untied variants.
    hf_model = transformers.T5ForConditionalGeneration.from_pretrained(
        hf_preset
    )
    hf_model.eval()
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_preset)

    # Load KerasHub components through the transformers converter
    # (`keras_hub/src/utils/transformers/convert_t5.py`).
    with tempfile.TemporaryDirectory() as hf_dir:
        keras_model, keras_tokenizer = load_keras_hub_components(
            hf_model, hf_preset, hf_dir
        )
    print("✅ KerasHub model loaded and weights converted.")
    print(f"🔶 Parameter count: {keras_model.count_params()}")

    validate_output(keras_model, keras_tokenizer, hf_model, hf_tokenizer)
    print("✅ Output validated.")

    keras_model.save_to_preset(f"./{preset}")
    keras_tokenizer.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
