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
import json
import traceback

import numpy as np
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from keras import ops
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM

from keras_nlp import upload_preset
from keras_nlp.models import Llama3Backbone
from keras_nlp.models import Llama3CausalLMPreprocessor
from keras_nlp.models import Llama3Tokenizer

PRESET_MAP = {
    "llama3_8b_en": "meta-llama/Meta-Llama-3-8B",
    "llama3_instruct_8b_en": "meta-llama/Meta-Llama-3-8B-Instruct",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def convert_checkpoints(keras_nlp_model, hf_model):
    config = hf_model.config

    keras_nlp_model.token_embedding.embeddings.assign(
        hf_model.model.embed_tokens.weight.detach().cpu().float().numpy()
    )

    for i in range(keras_nlp_model.num_layers):
        keras_nlp_model.transformer_layers[
            i
        ]._self_attention_layer._key_dense.set_weights(
            [
                hf_model.model.layers[i]
                .self_attn.k_proj.weight.T.reshape(
                    config.hidden_size,
                    config.num_key_value_heads,
                    config.hidden_size // config.num_attention_heads,
                )
                .detach()
                .cpu()
                .float()
                .numpy()
            ]
        )
        keras_nlp_model.transformer_layers[
            i
        ]._self_attention_layer._query_dense.set_weights(
            [
                hf_model.model.layers[i]
                .self_attn.q_proj.weight.T.reshape(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.hidden_size // config.num_attention_heads,
                )
                .detach()
                .cpu()
                .float()
                .numpy()
            ]
        )
        keras_nlp_model.transformer_layers[
            i
        ]._self_attention_layer._value_dense.set_weights(
            [
                hf_model.model.layers[i]
                .self_attn.v_proj.weight.T.reshape(
                    config.hidden_size,
                    config.num_key_value_heads,
                    config.hidden_size // config.num_attention_heads,
                )
                .detach()
                .cpu()
                .float()
                .numpy()
            ]
        )
        keras_nlp_model.transformer_layers[
            i
        ]._self_attention_layer._output_dense.set_weights(
            [
                hf_model.model.layers[i]
                .self_attn.o_proj.weight.T.reshape(
                    config.num_attention_heads,
                    config.hidden_size // config.num_attention_heads,
                    config.hidden_size,
                )
                .detach()
                .cpu()
                .float()
                .numpy()
            ]
        )
        keras_nlp_model.transformer_layers[
            i
        ]._self_attention_layernorm.set_weights(
            [
                hf_model.model.layers[i]
                .input_layernorm.weight.detach()
                .cpu()
                .float()
                .numpy()
            ]
        )
        keras_nlp_model.transformer_layers[
            i
        ]._feedforward_intermediate_dense.set_weights(
            [
                hf_model.model.layers[i]
                .mlp.up_proj.weight.T.detach()
                .cpu()
                .float()
                .numpy()
            ]
        )
        keras_nlp_model.transformer_layers[
            i
        ]._feedforward_output_dense.set_weights(
            [
                hf_model.model.layers[i]
                .mlp.down_proj.weight.T.detach()
                .cpu()
                .float()
                .numpy()
            ]
        )
        keras_nlp_model.transformer_layers[
            i
        ]._feedforward_gate_dense.set_weights(
            [
                hf_model.model.layers[i]
                .mlp.gate_proj.weight.T.detach()
                .cpu()
                .float()
                .numpy()
            ]
        )
        keras_nlp_model.transformer_layers[
            i
        ]._feedforward_layernorm.set_weights(
            [
                hf_model.model.layers[i]
                .post_attention_layernorm.weight.detach()
                .cpu()
                .float()
                .numpy()
            ]
        )

    keras_nlp_model.layer_norm.set_weights(
        [hf_model.model.norm.weight.detach().cpu().float().numpy()]
    )
    keras_nlp_model.token_embedding.reverse_embeddings.assign(
        hf_model.lm_head.weight.T.detach().cpu().float().numpy()
    )


def test_model(
    keras_nlp_model, keras_nlp_tokenizer, hf_model, hf_model_tokenizer
):
    # First, test that the number of parameters match
    keras_nlp_params = keras_nlp_model.count_params()
    hf_params = hf_model.num_parameters()
    assert keras_nlp_params == hf_params

    # Test the outputs of both the models
    hf_outputs = hf_model(
        **hf_model_tokenizer(["What is Keras?"], return_tensors="pt")
    )
    hf_output_logits = hf_outputs.logits.detach().cpu().numpy()

    keras_nlp_preprocessor = Llama3CausalLMPreprocessor(keras_nlp_tokenizer)
    keras_nlp_output = keras_nlp_model(
        keras_nlp_preprocessor(["What is Keras?"], sequence_length=5)[0]
    )
    keras_nlp_logits = keras_nlp_model.token_embedding(
        keras_nlp_output, reverse=True
    )
    keras_nlp_logits = ops.convert_to_numpy(keras_nlp_logits)

    # High tolerence since bfloat16 is used as the default dtype for Llama
    try:
        np.testing.assert_allclose(
            keras_nlp_logits, hf_output_logits, atol=1e-4
        )
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")


def test_tokenizer(keras_nlp_tokenizer, hf_tokenizer):
    hf_output = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()
    keras_nlp_preprocessor = Llama3CausalLMPreprocessor(keras_nlp_tokenizer)
    keras_nlp_output = keras_nlp_preprocessor(
        ["What is Keras?"], sequence_length=5
    )
    keras_nlp_output = ops.convert_to_numpy(keras_nlp_output[0]["token_ids"])

    np.testing.assert_equal(keras_nlp_output, hf_output)


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
    hf_model = LlamaForCausalLM.from_pretrained(
        hf_preset, torch_dtype=torch.bfloat16
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_model.eval()
    print("\n-> Huggingface model and tokenizer loaded")

    # === Load the KerasNLP model ===
    backbone_kwargs = dict(
        vocabulary_size=hf_model.config.vocab_size,
        hidden_dim=hf_model.config.hidden_size,
        num_layers=hf_model.config.num_hidden_layers,
        num_query_heads=hf_model.config.num_attention_heads,
        num_key_value_heads=hf_model.config.num_key_value_heads,
        intermediate_dim=hf_model.config.intermediate_size,
        layer_norm_epsilon=hf_model.config.rms_norm_eps,
        rope_max_wavelength=hf_model.config.rope_theta,
        dtype="bfloat16",
    )
    keras_nlp_model = Llama3Backbone(**backbone_kwargs)

    # === Get the tokenizer from the Huggingface model ===
    tokenizer_path = hf_hub_download(
        "meta-llama/Meta-Llama-3-8B", "tokenizer.json", token=True
    )
    with open(tokenizer_path, "r") as tokenizer_file:
        tokenizer_content = json.load(tokenizer_file)
    vocabulary = hf_tokenizer.vocab
    merges = tokenizer_content["model"]["merges"]
    keras_nlp_tokenizer = Llama3Tokenizer(vocabulary, merges)
    print("\n-> Keras 3 model and tokenizer loaded.")

    # === Port the weights ===
    convert_checkpoints(keras_nlp_model, hf_model)
    print("\n-> Weight transfer done.")

    # === Check that the models and tokenizers outputs match ===
    test_tokenizer(keras_nlp_tokenizer, hf_tokenizer)
    test_model(keras_nlp_model, keras_nlp_tokenizer, hf_model, hf_tokenizer)
    print("\n-> Tests passed!")

    keras_nlp_model.save_to_preset(preset)
    print("\n-> Saved the model preset in float16")

    # === Save the tokenizer ===
    keras_nlp_tokenizer.save_to_preset(preset)
    print("\n-> Saved the tokenizer")

    # === Upload the preset ===
    try:
        uri = f"kaggle://keras/llama3/keras/{preset}"
        upload_preset(uri, preset)
        print("-> Uploaded the preset!")
    except Exception:
        print(
            "-> Failed to upload the preset. Make sure you have the "
            "correct premissions to upload and/or the page "
            "you are pushing to exists."
        )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
