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
import tempfile

import keras
import numpy as np
import tensorflow as tf
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from keras_nlp.models.falcon.falcon_backbone import FalconBackbone

keras.config.disable_traceback_filtering()


def convert_checkpoints(hf_model):
    hf_config = hf_model.config.to_dict()
    cfg = {}
    cfg["vocabulary_size"] = hf_config["vocab_size"]
    cfg["num_layers"] = hf_config["num_hidden_layers"]
    cfg["num_attention_heads"] = hf_config["num_attention_heads"]
    cfg["hidden_dim"] = hf_config["hidden_size"]
    cfg["intermediate_dim"] = 4 * cfg["hidden_dim"]
    cfg["feedforward_dropout_rate"] = hf_config["hidden_dropout"]
    cfg["attention_dropout_rate"] = hf_config["attention_dropout"]

    keras_model = FalconBackbone(**cfg)

    hf_wts = hf_model.state_dict()

    # transformer.word_embeddings.weight
    keras_model.get_layer("token_embedding").embeddings.assign(
        hf_wts["transformer.word_embeddings.weight"]
    )

    for i in range(keras_model.num_layers):
        # split key query value
        fused_qkv = (
            hf_wts[f"transformer.h.{i}.self_attention.query_key_value.weight"]
            .numpy()
            .T
        )
        seq_length, _ = fused_qkv.shape
        head_dim = cfg["hidden_dim"] // cfg["num_attention_heads"]
        fused_qkv = fused_qkv.reshape(
            seq_length, cfg["num_attention_heads"], 3, head_dim
        )
        query, key, value = (
            fused_qkv[..., 0, :],
            fused_qkv[..., 1, :],
            fused_qkv[..., 2, :],
        )

        fused_bias = hf_wts[
            f"transformer.h.{i}.self_attention.query_key_value.bias"
        ].numpy()
        fused_bias = fused_bias.reshape(cfg["num_attention_heads"], 3, head_dim)
        query_bias, key_bias, value_bias = (
            fused_bias[..., 0, :],
            fused_bias[..., 1, :],
            fused_bias[..., 2, :],
        )

        # TODO: check if bias is true before assigning bias.
        # transformer.h.0.self_attention.query_key_value.weight
        # transformer.h.0.self_attention.query_key_value.bias
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).attention_layer.query_dense.kernel.assign(query)
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).attention_layer.query_dense.bias.assign(query_bias)

        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).attention_layer.key_dense.kernel.assign(key)
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).attention_layer.key_dense.bias.assign(key_bias)

        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).attention_layer.value_dense.kernel.assign(value)
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).attention_layer.value_dense.bias.assign(value_bias)

        # transformer.h.0.self_attention.dense.weight
        # transformer.h.0.self_attention.dense.bias
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).attention_layer.output_dense.kernel.assign(
            hf_wts[f"transformer.h.{i}.self_attention.dense.weight"].T.numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).attention_layer.output_dense.bias.assign(
            hf_wts[f"transformer.h.{i}.self_attention.dense.bias"].numpy()
        )

        # transformer.h.0.mlp.dense_h_to_4h.weight
        # transformer.h.0.mlp.dense_h_to_4h.bias
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).dense_h_to_4h.kernel.assign(
            hf_wts[f"transformer.h.{i}.mlp.dense_h_to_4h.weight"].T.numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).dense_h_to_4h.bias.assign(
            hf_wts[f"transformer.h.{i}.mlp.dense_h_to_4h.bias"].numpy()
        )

        # transformer.h.0.mlp.dense_4h_to_h.weight
        # transformer.h.0.mlp.dense_4h_to_h.bias
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).dense_4h_to_h.kernel.assign(
            hf_wts[f"transformer.h.{i}.mlp.dense_4h_to_h.weight"].T.numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).dense_4h_to_h.bias.assign(
            hf_wts[f"transformer.h.{i}.mlp.dense_4h_to_h.bias"].numpy()
        )

        # transformer.h.0.input_layernorm.weight
        # transformer.h.0.input_layernorm.bias
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).input_layernorm.gamma.assign(
            hf_wts[f"transformer.h.{i}.input_layernorm.weight"]
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).input_layernorm.beta.assign(
            hf_wts[f"transformer.h.{i}.input_layernorm.bias"]
        )

        # transformer.h.0.post_attention_layernorm.weight
        # transformer.h.0.post_attention_layernorm.bias
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).post_attention_layernorm.gamma.assign(
            hf_wts[f"transformer.h.{i}.post_attention_layernorm.weight"].numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        ).post_attention_layernorm.beta.assign(
            hf_wts[f"transformer.h.{i}.post_attention_layernorm.bias"].numpy()
        )

    # transformer.ln_f.weight
    # transformer.ln_f.bias
    keras_model.get_layer("final_layernorm").gamma.assign(
        hf_wts["transformer.ln_f.weight"].numpy()
    )
    keras_model.get_layer("final_layernorm").beta.assign(
        hf_wts["transformer.ln_f.bias"].numpy()
    )

    # TODO: Assign lm_head weights for CausalLM.
    # # lm_head.weight
    # keras_model.get_layer("lm_head").kernel.assign(
    #     hf_wts["lm_head.weight"].T.numpy()
    # )

    # Save the model.
    print("Save KerasNLP model weights.")
    temp_dir = tempfile.mkdtemp()
    keras_model.save_weights(os.path.join(temp_dir, "model.weights.h5"))

    return keras_model


def check_output(keras_model, hf_model, hf_model_name):
    sample_text = ["I am so happy today!"]
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    hf_sample_input = hf_tokenizer(
        sample_text, padding="max_length", return_tensors="pt"
    )
    sample_input = {
        "token_ids": tf.constant(hf_sample_input["input_ids"].numpy()),
        "padding_mask": tf.constant(hf_sample_input["attention_mask"].numpy()),
    }
    print("token_ids: ", sample_input["token_ids"][0, :7])
    print("padding_mask", sample_input["padding_mask"][0, :7])

    keras_output = keras_model.predict(sample_input)

    activation = {}

    def get_activation(name):
        def hook(hf_model, input, output):
            activation[name] = output[0].detach()

        return hook

    hf_model.transformer.register_forward_hook(
        get_activation("transformer.ln_f")
    )
    hf_model(**hf_sample_input)
    hf_output = activation["transformer.ln_f"]
    print("Keras shape: ", keras_output.shape)
    print("HF shape: ", hf_output.shape)

    print("KerasNLP output:", keras_output[0, 1, :5])
    print("HF output:", hf_output[0, 1, :5])
    print(
        "Difference:",
        np.mean(
            abs(keras_output[:, :6, :] - hf_output.detach().numpy()[:, :6, :])
        ),
    )


def main():
    hf_model_name = "tiiuae/falcon-rw-1b"
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    keras_model = convert_checkpoints(hf_model)
    check_output(keras_model, hf_model, hf_model_name)


if __name__ == "__main__":
    main()
