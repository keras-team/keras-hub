import gc
import os
import shutil
import tempfile
import traceback

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from keras import ops  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers import LlamaForCausalLM  # noqa: E402

from keras_hub import upload_preset  # noqa: E402
from keras_hub.models import LlamaBackbone  # noqa: E402
from keras_hub.models import LlamaCausalLMPreprocessor  # noqa: E402
from keras_hub.models import LlamaTokenizer  # noqa: E402

PRESET_MAP = {
    "llama2_7b_en": "meta-llama/Llama-2-7b-hf",
    "llama2_instruct_7b_en": "meta-llama/Llama-2-7b-chat-hf",
    "vicuna_1.5_7b_en": "lmsys/vicuna-7b-v1.5",
}

torch_dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)

flags.DEFINE_string(
    name="validate_dtype",
    default="bfloat16",
    help=(
        "The dtype of the two models while validating numerics."
        "can be 'float32', 'float16', or 'bfloat16'"
    ),
)

flags.DEFINE_string(
    name="save_dtype",
    default="bfloat16",
    help=(
        "The dtype of the two models while validating numerics."
        "can be 'float32', 'float16', or 'bfloat16'"
    ),
)


flags.DEFINE_string(
    name="upload_link",
    default=None,
    help=(
        "The link to upload the model. can be in these formats: "
        "`kaggle://<KAGGLE_USERNAME>/<MODEL>/<FRAMEWORK>/<VARIATION>`, "
        "`hf://[<HF_USERNAME>/]<MODEL>`"
    ),
)


def convert_checkpoints(keras_hub_model, hf_model):
    config = hf_model.config

    keras_hub_model.token_embedding.embeddings.assign(
        hf_model.model.embed_tokens.weight
    )

    for i in range(keras_hub_model.num_layers):
        keras_hub_model.transformer_layers[
            i
        ]._self_attention_layer._key_dense.set_weights(
            [
                hf_model.model.layers[i].self_attn.k_proj.weight.T.reshape(
                    config.hidden_size,
                    config.num_key_value_heads,
                    config.hidden_size // config.num_attention_heads,
                )
            ]
        )
        keras_hub_model.transformer_layers[
            i
        ]._self_attention_layer._query_dense.set_weights(
            [
                hf_model.model.layers[i].self_attn.q_proj.weight.T.reshape(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.hidden_size // config.num_attention_heads,
                )
            ]
        )
        keras_hub_model.transformer_layers[
            i
        ]._self_attention_layer._value_dense.set_weights(
            [
                hf_model.model.layers[i].self_attn.v_proj.weight.T.reshape(
                    config.hidden_size,
                    config.num_key_value_heads,
                    config.hidden_size // config.num_attention_heads,
                )
            ]
        )
        keras_hub_model.transformer_layers[
            i
        ]._self_attention_layer._output_dense.set_weights(
            [
                hf_model.model.layers[i].self_attn.o_proj.weight.T.reshape(
                    config.num_attention_heads,
                    config.hidden_size // config.num_attention_heads,
                    config.hidden_size,
                )
            ]
        )
        keras_hub_model.transformer_layers[
            i
        ]._self_attention_layernorm.set_weights(
            [hf_model.model.layers[i].input_layernorm.weight]
        )
        keras_hub_model.transformer_layers[
            i
        ]._feedforward_intermediate_dense.set_weights(
            [hf_model.model.layers[i].mlp.up_proj.weight.T]
        )
        keras_hub_model.transformer_layers[
            i
        ]._feedforward_output_dense.set_weights(
            [hf_model.model.layers[i].mlp.down_proj.weight.T]
        )
        keras_hub_model.transformer_layers[
            i
        ]._feedforward_gate_dense.set_weights(
            [hf_model.model.layers[i].mlp.gate_proj.weight.T]
        )
        keras_hub_model.transformer_layers[
            i
        ]._feedforward_layernorm.set_weights(
            [hf_model.model.layers[i].post_attention_layernorm.weight.detach()]
        )

    keras_hub_model.layer_norm.set_weights([hf_model.model.norm.weight])
    keras_hub_model.token_embedding.reverse_embeddings.assign(
        hf_model.lm_head.weight.T
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
    hf_output_logits = ops.convert_to_numpy(hf_outputs.logits)

    keras_hub_preprocessor = LlamaCausalLMPreprocessor(keras_hub_tokenizer)
    keras_hub_output = keras_hub_model(
        keras_hub_preprocessor(["What is Keras?"], sequence_length=6)[0]
    )
    keras_hub_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits)

    # High tolerence when bfloat16 is used as the default dtype for Llama
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
    hf_output = ops.convert_to_numpy(hf_output["input_ids"])
    keras_hub_preprocessor = LlamaCausalLMPreprocessor(keras_hub_tokenizer)
    keras_hub_output = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=6
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
    upload_link = FLAGS.upload_link
    hf_preset = PRESET_MAP[preset]
    torch_dtype = torch_dtype_map.get(FLAGS.validate_dtype)

    # === Create the temporary save directories ===
    temp_dir = tempfile.mkdtemp()

    try:
        # === Load the Huggingface model ===
        hf_model = LlamaForCausalLM.from_pretrained(
            hf_preset, torch_dtype=torch_dtype
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
        hf_model.eval()
        print(
            "\n-> Huggingface model and tokenizer loaded with dtype: "
            f"{FLAGS.validate_dtype}"
        )

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
            dtype=FLAGS.validate_dtype,
        )
        keras_hub_model = LlamaBackbone(**backbone_kwargs)

        # === Get the tokenizer from the Huggingface model ===
        tokenizer_path = hf_tokenizer.vocab_file
        keras_hub_tokenizer = LlamaTokenizer(tokenizer_path)
        print("\n-> Keras 3 model and tokenizer loaded.")

        # === Port the weights ===
        convert_checkpoints(keras_hub_model, hf_model)
        print("\n-> Weight transfer done.")

        # === Check that the models and tokenizers outputs match ===
        test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
        test_model(keras_hub_model, keras_hub_tokenizer, hf_model, hf_tokenizer)
        print("\n-> Tests passed!")

        keras_hub_model.save_weights(os.path.join(temp_dir, "model.weights.h5"))
        print(f"\n-> Saved the model weights in {FLAGS.validate_dtype}")

        del keras_hub_model, hf_model
        gc.collect()

        # === Save the weights again in user defined dtype ===
        backbone_kwargs["dtype"] = FLAGS.save_dtype
        keras_hub_model = LlamaBackbone(**backbone_kwargs)
        keras_hub_model.load_weights(os.path.join(temp_dir, "model.weights.h5"))

        # === Save the model ===
        keras_hub_model.save_to_preset(preset)
        print(f"\n-> Saved the model preset in {FLAGS.save_dtype}")

        # === Save the tokenizer ===
        keras_hub_tokenizer.save_to_preset(preset)
        print("\n-> Saved the tokenizer")

        # == Upload preset ==
        if upload_link is not None:
            upload_preset(upload_link, preset)
            print("-> Uploaded the preset!")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
