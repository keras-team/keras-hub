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
import math
import os
import shutil

import numpy as np
import transformers
from absl import app
from absl import flags
from checkpoint_conversion_utils import get_md5_checksum
from keras import ops

import keras_nlp

PRESET_MAP = {
    "t5_small_multi": "t5-small",
    "t5_base_multi": "t5-base",
    "t5_large_multi": "t5-large",
    "flan_small_multi": "google/flan-t5-small",
    "flan_base_multi": "google/flan-t5-base",
    "flan_large_multi": "google/flan-t5-large",
}

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "preset", "t5_base_multi", f'Must be one of {",".join(PRESET_MAP.keys())}'
)
os.environ["KERAS_BACKEND"] = "torch"


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


def convert_checkpoints(hf_model):
    keras_nlp_model = keras_nlp.models.T5Backbone.from_preset(
        FLAGS.preset, load_weights=False
    )

    hf_wts = hf_model.state_dict()
    print("Original weights:")
    print(list(hf_wts.keys()))

    for i in range(keras_nlp_model.num_layers):
        for section in ["encoder", "decoder"]:
            n = 0

            # Token embedding layer
            keras_nlp_model.get_layer("token_embedding").embeddings.assign(
                hf_wts[f"{section}.embed_tokens.weight"]
            )
            if not keras_nlp_model.tie_embedding_weights:
                keras_nlp_model.get_layer(
                    "token_embedding"
                ).reverse_embeddings.assign(
                    hf_wts["lm_head.weight"].transpose(1, 0).numpy()
                )

            # Query, key, value, and output projectors in self-attention
            keras_nlp_model.get_layer(
                f"transformer_{section}_layer_{i}"
            ).self_attention.query_projector.kernel.assign(
                hf_wts[f"{section}.block.{i}.layer.{n}.SelfAttention.q.weight"]
                .transpose(1, 0)
                .numpy()
            )
            keras_nlp_model.get_layer(
                f"transformer_{section}_layer_{i}"
            ).self_attention.key_projector.kernel.assign(
                hf_wts[f"{section}.block.{i}.layer.{n}.SelfAttention.k.weight"]
                .transpose(1, 0)
                .numpy()
            )
            keras_nlp_model.get_layer(
                f"transformer_{section}_layer_{i}"
            ).self_attention.value_projector.kernel.assign(
                hf_wts[f"{section}.block.{i}.layer.{n}.SelfAttention.v.weight"]
                .transpose(1, 0)
                .numpy()
            )
            keras_nlp_model.get_layer(
                f"transformer_{section}_layer_{i}"
            ).self_attention.output_projector.kernel.assign(
                hf_wts[f"{section}.block.{i}.layer.{n}.SelfAttention.o.weight"]
                .transpose(1, 0)
                .numpy()
            )

            # Add relative attention bias
            if keras_nlp_model.get_layer(
                f"transformer_{section}_layer_{i}"
            ).self_attention.use_relative_attention_bias:
                keras_nlp_model.get_layer(
                    f"transformer_{section}_layer_{i}"
                ).self_attention.relative_attention_bias.assign(
                    hf_wts[
                        f"{section}.block.{i}.layer.{n}.SelfAttention.relative_attention_bias.weight"
                    ].numpy()
                )

            # Self-attention norm
            keras_nlp_model.get_layer(
                f"transformer_{section}_layer_{i}"
            ).self_attention_layer_norm.weight.assign(
                hf_wts[
                    f"{section}.block.{i}.layer.{n}.layer_norm.weight"
                ].numpy()
            )

            # Increment for next layer
            n += 1

            if section == "decoder":
                # Cross-attention QKV and output proj (one between encoder and decoder)
                keras_nlp_model.get_layer(
                    f"transformer_{section}_layer_{i}"
                ).cross_attention.query_projector.kernel.assign(
                    hf_wts[
                        f"{section}.block.{i}.layer.{n}.EncDecAttention.q.weight"
                    ]
                    .transpose(1, 0)
                    .numpy()
                )
                keras_nlp_model.get_layer(
                    f"transformer_{section}_layer_{i}"
                ).cross_attention.key_projector.kernel.assign(
                    hf_wts[
                        f"{section}.block.{i}.layer.{n}.EncDecAttention.k.weight"
                    ]
                    .transpose(1, 0)
                    .numpy()
                )
                keras_nlp_model.get_layer(
                    f"transformer_{section}_layer_{i}"
                ).cross_attention.value_projector.kernel.assign(
                    hf_wts[
                        f"{section}.block.{i}.layer.{n}.EncDecAttention.v.weight"
                    ]
                    .transpose(1, 0)
                    .numpy()
                )
                keras_nlp_model.get_layer(
                    f"transformer_{section}_layer_{i}"
                ).cross_attention.output_projector.kernel.assign(
                    hf_wts[
                        f"{section}.block.{i}.layer.{n}.EncDecAttention.o.weight"
                    ]
                    .transpose(1, 0)
                    .numpy()
                )

                # Cross-attention layer norm
                keras_nlp_model.get_layer(
                    f"transformer_{section}_layer_{i}"
                ).cross_attention_layer_norm.weight.assign(
                    hf_wts[
                        f"{section}.block.{i}.layer.{n}.layer_norm.weight"
                    ].numpy()
                )
                # Increment for next layer
                n += 1

            if keras_nlp_model.get_layer(
                f"transformer_{section}_layer_{i}"
            ).use_gated_activation:
                # Input projection layer
                keras_nlp_model.get_layer(
                    f"transformer_{section}_layer_{i}"
                ).input_projector.weights[0].assign(
                    hf_wts[
                        f"{section}.block.{i}.layer.{n}.DenseReluDense.wi_0.weight"
                    ]
                    .transpose(1, 0)
                    .numpy()
                )

                # Gated activation layer
                keras_nlp_model.get_layer(
                    f"transformer_{section}_layer_{i}"
                ).gate_projector.weights[0].assign(
                    hf_wts[
                        f"{section}.block.{i}.layer.{n}.DenseReluDense.wi_1.weight"
                    ]
                    .transpose(1, 0)
                    .numpy()
                )
            else:
                # Input projection layer
                keras_nlp_model.get_layer(
                    f"transformer_{section}_layer_{i}"
                ).input_projector.weights[0].assign(
                    hf_wts[
                        f"{section}.block.{i}.layer.{n}.DenseReluDense.wi.weight"
                    ]
                    .transpose(1, 0)
                    .numpy()
                )

            # Output projection layer
            keras_nlp_model.get_layer(
                f"transformer_{section}_layer_{i}"
            ).output_projector.weights[0].assign(
                hf_wts[
                    f"{section}.block.{i}.layer.{n}.DenseReluDense.wo.weight"
                ]
                .transpose(1, 0)
                .numpy()
            )

            # Layer norm
            keras_nlp_model.get_layer(
                f"transformer_{section}_layer_{i}"
            ).layer_norm.weight.assign(
                hf_wts[
                    f"{section}.block.{i}.layer.{n}.layer_norm.weight"
                ].numpy()
            )

            # Final normalization
            keras_nlp_model.get_layer(f"{section}_output_layer_norm").weights[
                -1
            ].assign(hf_wts[f"{section}.final_layer_norm.weight"].numpy())

    return keras_nlp_model


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
        return_tensors="pt",
    )
    hf_decoder_inputs = hf_tokenizer(
        decoder_input,
        padding="max_length",
        max_length=sequence_length,
        return_tensors="pt",
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
    keras_out = keras_model(keras_inputs)
    hf_out = hf_model(**hf_inputs, output_hidden_states=True)

    # Only compare non-padded token ids.
    keras_hidden_states = keras_out["decoder_sequence_output"]
    hf_hidden_states = hf_out.decoder_hidden_states[-1]

    keras_outputs = ops.take_along_axis(
        keras_hidden_states, ops.where(decoder_padding_mask)
    )
    hf_outputs = ops.take_along_axis(
        hf_hidden_states, ops.where(decoder_padding_mask)
    )

    print("-> KerasNLP output:", keras_outputs[0:5])
    print("-> HF output:", hf_outputs[0:5])
    np.testing.assert_allclose(
        keras_outputs.detach().numpy(), hf_outputs.detach().numpy(), atol=1e-5
    )

    if keras_model.tie_embedding_weights:
        keras_hidden_states = keras_hidden_states * (
            keras_model.hidden_dim**-0.5
        )

    keras_logits = keras_model.token_embedding(
        keras_hidden_states, reverse=True
    )
    hf_logits = hf_out.logits
    print("-> KerasNLP logits:", keras_logits[0:5])
    print("-> HF logits:", hf_logits[0:5])
    np.testing.assert_allclose(
        keras_logits.detach().numpy(), hf_logits.detach().numpy(), atol=1e-3
    )


def count_params(weights):
    shapes = [v.shape for v in weights]
    return int(sum(math.prod(p) for p in shapes))


def main(_):
    hf_id = PRESET_MAP[FLAGS.preset]
    shutil.rmtree(f"./{FLAGS.preset}", ignore_errors=True)
    os.mkdir(f"./{FLAGS.preset}")

    print("\n-> Convert weights.")
    hf_model = transformers.T5ForConditionalGeneration.from_pretrained(hf_id)
    keras_model = convert_checkpoints(hf_model)

    # Save the model.
    model_path = f"./{FLAGS.preset}/model.weights.h5"
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
