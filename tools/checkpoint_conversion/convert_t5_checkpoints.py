# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil

import numpy as np
import tensorflow as tf
import transformers
from absl import app
from absl import flags
from keras.utils.layer_utils import count_params

import keras_nlp
from tools.checkpoint_conversion.checkpoint_conversion_utils import (
    get_md5_checksum,
)
from tools.checkpoint_conversion.checkpoint_conversion_utils import (
    port_weights_by_creation_order,
)

PRESET_MAP = {
    "t5_small_en": "google/t5-v1_1-small",
    "t5_base_en": "google/t5-v1_1-base",
    "t5_large_en": "google/t5-v1_1-large",
    "t5_extra_large_en": "google/t5-v1_1-xl",
    "t5_extra_extra_large_en": "google/t5-v1_1-xxl",
}

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "preset", "t5_base", f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def extract_vocab(hf_tokenizer):
    proto_path = f"./{FLAGS.preset}/vocab.spm"
    print(f"\n-> Save KerasNLP vocab to `{proto_path}`.")

    # Huggingface has a save_vocabulary function but it's not byte-for-byte
    # with the source. Instead copy the original downloaded file directly.
    shutil.copyfile(
        transformers.utils.hub.get_file_from_repo(
            hf_tokenizer.name_or_path, "spiece.model"
        ),
        proto_path,
    )

    keras_tokenizer = keras_nlp.models.T5Tokenizer(
        proto=proto_path,
    )

    print("-> Print MD5 checksum of the vocab files.")
    print(f"`{proto_path}` md5sum: ", get_md5_checksum(proto_path))

    return keras_tokenizer


def check_output(
    keras_model,
    keras_tokenizer,
    hf_model,
    hf_tokenizer,
):
    print("\n-> Compare the outputs.")
    encoder_input = ["the quick brown fox jumped."]
    decoder_input = ["the quick brown fox fell."]

    sequence_length = 12

    # KerasNLP Tokenization
    packer = keras_nlp.layers.StartEndPacker(
        sequence_length=sequence_length,
        pad_value=keras_tokenizer.pad_token_id,
        end_value=keras_tokenizer.end_token_id,
    )
    encoder_token_ids = packer(keras_tokenizer(encoder_input))
    encoder_padding_mask = encoder_token_ids != keras_tokenizer.pad_token_id
    decoder_token_ids = packer(keras_tokenizer(decoder_input))
    decoder_padding_mask = decoder_token_ids != keras_tokenizer.pad_token_id
    keras_inputs = {
        "encoder_token_ids": encoder_token_ids,
        "encoder_padding_mask": encoder_padding_mask,
        "decoder_token_ids": decoder_token_ids,
        "decoder_padding_mask": decoder_padding_mask,
    }

    # HF Tokenization.
    hf_encoder_inputs = hf_tokenizer(
        encoder_input,
        padding="max_length",
        max_length=sequence_length,
        return_tensors="tf",
    )
    hf_decoder_inputs = hf_tokenizer(
        decoder_input,
        padding="max_length",
        max_length=sequence_length,
        return_tensors="tf",
    )
    hf_inputs = {
        "input_ids": hf_encoder_inputs["input_ids"],
        "attention_mask": hf_encoder_inputs["attention_mask"],
        "decoder_input_ids": hf_decoder_inputs["input_ids"],
        "decoder_attention_mask": hf_decoder_inputs["attention_mask"],
    }

    # Compare tokenized inputs. This should be a compete match.
    print("-> KerasNLP inputs:")
    for k, v in keras_inputs.items():
        print(k, v)
    print("-> HF inputs:")
    for k, v in hf_inputs.items():
        print(k, v)

    # Forward pass
    keras_outputs = keras_model(keras_inputs)
    hf_outputs = hf_model(**hf_inputs)

    # Only compare non-padded token ids.
    keras_outputs = keras_outputs["decoder_sequence_output"]
    keras_outputs = tf.gather_nd(keras_outputs, tf.where(decoder_padding_mask))
    hf_outputs = hf_outputs.last_hidden_state
    hf_outputs = tf.gather_nd(hf_outputs, tf.where(decoder_padding_mask))

    print("-> KerasNLP output:", keras_outputs[0, :5])
    print("-> HF output:", hf_outputs[0, :5])
    np.testing.assert_allclose(
        keras_outputs.numpy(), hf_outputs.numpy(), atol=1e-5
    )


def main(_):
    hf_id = PRESET_MAP[FLAGS.preset]
    shutil.rmtree(f"./{FLAGS.preset}", ignore_errors=True)
    os.mkdir(f"./{FLAGS.preset}")

    print("\n-> Convert weights.")
    hf_model, keras_model = port_weights_by_creation_order(
        lambda: transformers.TFAutoModel.from_pretrained(hf_id),
        lambda: keras_nlp.models.T5Backbone.from_preset(
            FLAGS.preset, load_weights=False
        ),
    )

    # Save the model.
    model_path = f"./{FLAGS.preset}/model.h5"
    print(f"\n-> Save KerasNLP model weights to `{model_path}`.")
    keras_model.save_weights(model_path)
    print("-> Print MD5 checksum of the model weights files.")
    print(f"`{model_path}` md5sum: ", get_md5_checksum(model_path))
    print(f"-> Param count {count_params(keras_model.weights)}")

    print("\n-> Convert vocab.")
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_id)
    keras_tokenizer = extract_vocab(hf_tokenizer)

    check_output(
        keras_model,
        keras_tokenizer,
        hf_model,
        hf_tokenizer,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
