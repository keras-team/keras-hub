import gc
import os
import shutil
import tempfile
import traceback

import numpy as np
from huggingface_hub import hf_hub_download
from absl import app
from absl import flags
from keras import ops
from transformers import AutoTokenizer
from transformers import MistralForCausalLM

from keras_hub.models import MistralBackbone
from keras_hub.models import MistralCausalLMPreprocessor
from keras_hub.models import MistralTokenizer

PRESET_MAP = {
    "mistral_7b_en": "mistralai/Mistral-7B-v0.1",
    "mistral_0.3_7b_en": "mistralai/Mistral-7B-v0.3",
    "mistral_instruct_7b_en": "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral_0.2_instruct_7b_en": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral_0.3_instruct_7b_en": "mistralai/Mistral-7B-Instruct-v0.3",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def convert_checkpoints(keras_hub_model, hf_model):
    config = hf_model.config

    keras_hub_model.token_embedding.embeddings.assign(
        hf_model.model.embed_tokens.weight.detach().cpu().numpy()
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
                .numpy()
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
                .numpy()
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
                .numpy()
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
                .numpy()
            ]
        )
        keras_hub_model.transformer_layers[
            i
        ]._self_attention_layernorm.set_weights(
            [
                hf_model.model.layers[i]
                .input_layernorm.weight.detach()
                .cpu()
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
                .numpy()
            ]
        )

    keras_hub_model.layer_norm.set_weights(
        [hf_model.model.norm.weight.detach().cpu().numpy()]
    )
    keras_hub_model.token_embedding.reverse_embeddings.assign(
        hf_model.lm_head.weight.T.detach().cpu().numpy()
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
    hf_output_logits = hf_outputs.logits.detach().cpu().numpy()

    keras_hub_preprocessor = MistralCausalLMPreprocessor(keras_hub_tokenizer)
    keras_hub_output = keras_hub_model(
        keras_hub_preprocessor(["What is Keras?"], sequence_length=6)[0]
    )
    keras_hub_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits)

    # High tolerence since bfloat16 is used as the default dtype for Mistral
    try:
        np.testing.assert_allclose(
            keras_hub_logits, hf_output_logits, atol=1e-4
        )
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    hf_output = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()
    if isinstance(keras_hub_tokenizer, MistralTokenizer):
        keras_hub_preprocessor = MistralCausalLMPreprocessor(keras_hub_tokenizer)
        keras_hub_output = keras_hub_preprocessor(
            ["What is Keras?"], sequence_length=6
        )
        keras_hub_output = ops.convert_to_numpy(keras_hub_output[0]["token_ids"])
    else:
        keras_hub_output = keras_hub_tokenizer(
            ["What is Keras?"], return_tensors="pt"
        )
        keras_hub_output = (
            keras_hub_output["input_ids"].detach().cpu().numpy()
        )

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

    # === Create the temporary save directories ===
    temp_dir = tempfile.mkdtemp()

    try:
        # === Load the Huggingface model ===
        hf_model = MistralForCausalLM.from_pretrained(hf_preset)
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
            sliding_window=hf_model.config.sliding_window,
            layer_norm_epsilon=hf_model.config.rms_norm_eps,
            rope_max_wavelength=hf_model.config.rope_theta,
            dropout=0,
            dtype="float32",
        )
        keras_hub_model = MistralBackbone(**backbone_kwargs)

        # === Download the tokenizer from Huggingface model card ===
        try:
            tokenizer_path = hf_hub_download(
                repo_id=hf_preset,
                filename="tokenizer.model",
            )
            keras_hub_tokenizer = MistralTokenizer(tokenizer_path)
        except:
            try:
                hf_hub_download(
                    repo_id=hf_preset,
                    filename="tokenizer.json",
                )
                keras_hub_tokenizer = hf_tokenizer
            except:
                raise ValueError(f"Couldn't fetch {preset}'s tokenizer.")
        print("\n-> Keras 3 model and tokenizer loaded.")

        # === Port the weights ===
        convert_checkpoints(keras_hub_model, hf_model)
        print("\n-> Weight transfer done.")

        # === Check that the models and tokenizers outputs match ===
        test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
        test_model(keras_hub_model, keras_hub_tokenizer, hf_model, hf_tokenizer)
        print("\n-> Tests passed!")

        # === Save the model weights in float32 format ===
        keras_hub_model.save_weights(os.path.join(temp_dir, "model.weights.h5"))
        print("\n-> Saved the model weights in float32")

        del keras_hub_model, hf_model
        gc.collect()

        # === Save the weights again in float16 ===
        backbone_kwargs["dtype"] = "float16"
        keras_hub_model = MistralBackbone(**backbone_kwargs)
        keras_hub_model.load_weights(os.path.join(temp_dir, "model.weights.h5"))
        keras_hub_model.save_to_preset(preset)
        print("\n-> Saved the model preset in float16")

        # === Save the tokenizer ===
        if isinstance(keras_hub_tokenizer, MistralTokenizer):
            keras_hub_tokenizer.save_to_preset(preset)
        else:
            hf_tokenizer.save_pretrained(preset)
        print("\n-> Saved the tokenizer")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
