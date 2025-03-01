import json
import os
import traceback

import numpy as np
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download

os.environ["KERAS_BACKEND"] = "torch"


from keras import ops
from transformers import AutoTokenizer
from transformers import Qwen2ForCausalLM

from keras_hub.models import Qwen2Backbone
from keras_hub.models import Qwen2CausalLMPreprocessor
from keras_hub.models import Qwen2Tokenizer

PRESET_MAP = {
    "qwen2.5_0.5b_en": "Qwen/Qwen2.5-0.5B",
    "qwen2.5_instruct_0.5b_en": "Qwen/Qwen2.5-0.5B-Instruct",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def convert_checkpoints(keras_hub_model, hf_model):
    config = hf_model.config

    keras_hub_model.token_embedding.embeddings.assign(
        hf_model.model.embed_tokens.weight.detach().cpu().float().numpy()
    )

    for i in range(keras_hub_model.num_layers):
        keras_hub_model.transformer_layers[
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
                .numpy(),
                hf_model.model.layers[i]
                .self_attn.k_proj.bias.T.reshape(
                    config.num_key_value_heads,
                    -1,
                )
                .detach()
                .cpu()
                .float()
                .numpy(),
            ]
        )
        keras_hub_model.transformer_layers[
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
                .numpy(),
                hf_model.model.layers[i]
                .self_attn.q_proj.bias.T.reshape(
                    config.num_attention_heads,
                    -1,
                )
                .detach()
                .cpu()
                .float()
                .numpy(),
            ]
        )

        keras_hub_model.transformer_layers[
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
                .numpy(),
                hf_model.model.layers[i]
                .self_attn.v_proj.bias.T.reshape(
                    config.num_key_value_heads,
                    -1,
                )
                .detach()
                .cpu()
                .float()
                .numpy(),
            ]
        )
        keras_hub_model.transformer_layers[
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
                .numpy(),
            ]
        )
        keras_hub_model.transformer_layers[
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
        keras_hub_model.transformer_layers[
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
        keras_hub_model.transformer_layers[
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
        keras_hub_model.transformer_layers[
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
        keras_hub_model.transformer_layers[
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

    keras_hub_model.layer_norm.set_weights(
        [hf_model.model.norm.weight.detach().cpu().float().numpy()]
    )


def test_model(
    keras_hub_model, keras_hub_tokenizer, hf_model, hf_model_tokenizer
):
    # First, test that the number of parameters match
    keras_hub_params = keras_hub_model.count_params()
    hf_params = hf_model.num_parameters()
    assert keras_hub_params == hf_params

    # Test the outputs of both the models
    hf_outputs = hf_model(
        **hf_model_tokenizer(["What is Keras?"], return_tensors="pt")
    )
    hf_output_logits = hf_outputs.logits.detach().cpu().float().numpy()

    keras_hub_preprocessor = Qwen2CausalLMPreprocessor(keras_hub_tokenizer)
    keras_hub_output = keras_hub_model(
        keras_hub_preprocessor(["What is Keras?"], sequence_length=5)[0]
    )
    keras_hub_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits)

    # High tolerence since bfloat16 is used as the default dtype for Qwen

    try:
        np.testing.assert_allclose(
            keras_hub_logits, hf_output_logits, atol=1e-3
        )
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    hf_output = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()
    keras_hub_preprocessor = Qwen2CausalLMPreprocessor(keras_hub_tokenizer)
    keras_hub_output = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=5
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
    hf_model = Qwen2ForCausalLM.from_pretrained(
        hf_preset, torch_dtype=torch.bfloat16
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_model.eval()

    print("\n-> Huggingface model and tokenizer loaded")

    # === Load the KerasHub model ===
    backbone_kwargs = dict(
        vocabulary_size=hf_model.config.vocab_size,
        hidden_dim=hf_model.config.hidden_size,
        num_layers=hf_model.config.num_hidden_layers,
        num_query_heads=hf_model.config.num_attention_heads,
        num_key_value_heads=hf_model.config.num_key_value_heads,
        intermediate_dim=hf_model.config.intermediate_size,
        layer_norm_epsilon=hf_model.config.rms_norm_eps,
        rope_max_wavelength=hf_model.config.rope_theta,
        use_sliding_window=hf_model.config.use_sliding_window,
        sliding_window_size=hf_model.config.sliding_window,
        dtype="bfloat16",
    )

    keras_hub_model = Qwen2Backbone(**backbone_kwargs)

    # === Port the weights ===
    convert_checkpoints(keras_hub_model, hf_model)
    print("\n-> Weight transfer done.")

    # === Get the tokenizer from the Huggingface model ===
    tokenizer_path = hf_hub_download(hf_preset, "tokenizer.json", token=True)
    with open(tokenizer_path, "r") as tokenizer_file:
        tokenizer_content = json.load(tokenizer_file)
    vocabulary = hf_tokenizer.vocab
    merges = tokenizer_content["model"]["merges"]
    keras_hub_tokenizer = Qwen2Tokenizer(vocabulary, merges)
    print("\n-> Keras 3 model and tokenizer loaded.")

    # === Check that the models and tokenizers outputs match ===
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
    test_model(keras_hub_model, keras_hub_tokenizer, hf_model, hf_tokenizer)
    print("\n-> Tests passed!")

    # keras_hub_model.save_to_preset(preset)
    print("\n-> Saved the model preset in float16")

    # === Save the tokenizer ===
    # keras_hub_tokenizer.save_to_preset(preset)
    print("\n-> Saved the tokenizer")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
