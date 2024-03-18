# Copyright 2024 The KerasNLP Authors
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

import contextlib
import os

import torch
import transformers
from absl import app
from absl import flags

import keras_nlp

os.environ["KERAS_BACKEND"] = "torch"

"""
Sample usage:

For converting a keras model to HuggingFace format using a custom or fine-tuned
checkpoint from Keras, make sure to pass the path for the Keras weights file
(ending in `.weights.h5`), the model size (`2b` or `7b`), and the tokenizer
vocabulary file (`.spm`, `.model`, or equivalent) to
`--weights_file`, `--size`, and `--vocab_path`, respectively.

Optionally, you can specify the output directory
for the converted model at `--output_dir`. (defaults to `gg_hf`)
```
python tools/gemma/export_gemma_to_hf.py \
  --weights_file fine_tuned_imdb.weights.h5 \
  --size 2b \
  --vocab_path gemma_lm_tokenizer/vocabulary.spm \
  --output_dir fine_tuned_gg_hf
```

For converting a Keras model to HuggingFace format from a preset,
simply pass the Keras preset name to `--preset` and its model size
(`2b` or `7b`) to `--size`.
```
python tools/gemma/export_gemma_to_hf.py \
    --preset gemma_2b_en \
    --size 2b \
    --output_dir keras_hf_model/
```
"""


PRESET_MAP = {
    "gemma_2b_en": "gg-hf/gemma-2b",
    "gemma_instruct_2b_en": "gg-hf/gemma-2b",
    "gemma_7b_en": "gg-hf/gemma-7b",
    "gemma_instruct_7b_en": "gg-hf/gemma-7b",
}

SIZE_MAP = {
    "2b": ("gg-hf/gemma-2b", "gemma_2b_en"),
    "7b": ("gg-hf/gemma-7b", "gemma_7b_en"),
}

gemma_2b_config = transformers.GemmaConfig(
    num_hidden_layers=18,
    num_attention_heads=8,
    num_key_value_heads=1,
    hidden_size=2048,
    intermediate_size=16384,
)

gemma_7b_config = transformers.GemmaConfig()

CONFIG_MAPPING = {"2b": gemma_2b_config, "7b": gemma_7b_config}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "hf_token",
    None,
    "Your HuggingFace token. Needed for access to the HuggingFace Gemma"
    "implementation since the repository is private, for now.",
)
flags.DEFINE_string(
    "preset",
    None,
    f'Must be one of {",".join(PRESET_MAP.keys())}'
    " Alternatively, a Keras weights file (`.weights.h5`) can be passed"
    " to --weights_file flag.",
)
flags.DEFINE_string(
    "weights_file",
    None,
    "A Keras weights file (`.weights.h5`)."
    " Alternatively, a model preset can be passed to --preset flag.",
)
flags.DEFINE_string(
    "size",
    None,
    "Size of model. Must be passed if `weights_file` is passed. "
    "This should be either `2b` or `7b`.",
)
flags.DEFINE_string(
    "output_dir",
    "gg_hf",
    "An output directory for the converted HuggingFace model and tokenizer.",
)
flags.DEFINE_string(
    "vocab_path",
    None,
    "A path containing the vocabulary (must be a `.spm` file or equivalent). "
    "If not passed, the vocabulary of the preset will be used.",
)
flags.DEFINE_string(
    "dtype",
    "float32",
    "Set the precision of the converted checkpoint. Must be a valid PyTorch dtype.",
)


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def convert_checkpoints(preset, weights_file, size, output_dir, vocab_path):
    if preset is not None:
        hf_id = PRESET_MAP[preset]
        print(f"\n-> Loading KerasNLP Gemma model with preset `{preset}`...")
        keras_nlp_model = keras_nlp.models.GemmaCausalLM.from_preset(preset)
    else:
        hf_id, keras_preset = SIZE_MAP[size.lower()]
        print(f"\n-> Loading Keras weights from file `{weights_file}`...")
        keras_nlp_model = keras_nlp.models.GemmaCausalLM.from_preset(
            keras_preset
        )
        keras_nlp_model.load_weights(weights_file)

    print(f"\n-> Loading HuggingFace Gemma `{size.upper()}` model...")
    hf_model = transformers.GemmaForCausalLM(CONFIG_MAPPING[size.lower()])

    print("\n✅ Model loading complete.")
    print("\n-> Converting weights from KerasNLP Gemma to HuggingFace Gemma...")

    # Token embedding (with vocab size difference handling)
    keras_embedding = keras_nlp_model.backbone.token_embedding.weights[0]
    hf_vocab_size = hf_model.model.embed_tokens.weight.shape[0]
    keras_nlp_vocab_size = keras_embedding.value.shape[0]
    if hf_vocab_size < keras_nlp_vocab_size:
        diff = keras_nlp_vocab_size - hf_vocab_size
        update_state_dict(
            hf_model.model.embed_tokens,
            "weight",
            keras_embedding.value[:-diff, :],
        )
    else:
        update_state_dict(
            hf_model.model.embed_tokens,
            "weight",
            keras_embedding.value,
        )

    # Decoder blocks
    for i in range(keras_nlp_model.backbone.num_layers):
        decoder_block = keras_nlp_model.backbone.get_layer(f"decoder_block_{i}")

        # Pre-attention norm
        update_state_dict(
            hf_model.model.layers[i].input_layernorm,
            "weight",
            decoder_block.pre_attention_norm.weights[0].value,
        )

        # Attention
        query_target_shape = hf_model.model.layers[
            i
        ].self_attn.q_proj.weight.shape
        query_tensor = decoder_block.attention.query_dense.weights[0].value
        query_tensor = query_tensor.transpose(1, 2).reshape(query_target_shape)
        update_state_dict(
            hf_model.model.layers[i].self_attn.q_proj, "weight", query_tensor
        )

        key_target_shape = hf_model.model.layers[
            i
        ].self_attn.k_proj.weight.shape
        key_tensor = decoder_block.attention.key_dense.weights[0].value
        key_tensor = key_tensor.transpose(1, 2).reshape(key_target_shape)
        update_state_dict(
            hf_model.model.layers[i].self_attn.k_proj, "weight", key_tensor
        )

        value_target_shape = hf_model.model.layers[
            i
        ].self_attn.v_proj.weight.shape
        value_tensor = decoder_block.attention.value_dense.weights[0].value
        value_tensor = value_tensor.transpose(1, 2).reshape(value_target_shape)
        update_state_dict(
            hf_model.model.layers[i].self_attn.v_proj, "weight", value_tensor
        )

        out_target_shape = hf_model.model.layers[
            i
        ].self_attn.o_proj.weight.shape
        keras_out_tensor = decoder_block.attention.output_dense.weights[0].value
        out_tensor = keras_out_tensor.reshape(
            (out_target_shape[1], out_target_shape[0])  # Transpose target size
        ).transpose(0, 1)

        update_state_dict(
            hf_model.model.layers[i].self_attn.o_proj, "weight", out_tensor
        )

        # Post-attention norm
        update_state_dict(
            hf_model.model.layers[i].post_attention_layernorm,
            "weight",
            decoder_block.pre_ffw_norm.weights[0].value,
        )

        # MLP (Feed-forward)
        update_state_dict(
            hf_model.model.layers[i].mlp.gate_proj,
            "weight",
            decoder_block.gating_ffw.weights[0].value.transpose(0, 1),
        )
        update_state_dict(
            hf_model.model.layers[i].mlp.up_proj,
            "weight",
            decoder_block.gating_ffw_2.weights[0].value.transpose(0, 1),
        )
        update_state_dict(
            hf_model.model.layers[i].mlp.down_proj,
            "weight",
            decoder_block.ffw_linear.weights[0].value.transpose(0, 1),
        )

    # Final norm
    update_state_dict(
        hf_model.model.norm,
        "weight",
        keras_nlp_model.backbone.layers[-1].weights[0].value,
    )

    print("\n✅ Weights converted successfully.")
    print(f"\n-> Saving HuggingFace model to `{output_dir}`...")

    # Save model to HF Transformers format
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir)

    print(f"\n✅ Saving complete. Model saved at `{output_dir}`.")

    # Tokenizer

    if not vocab_path:
        tokenizer_preset = preset or SIZE_MAP[size.lower()]
        print(
            "\n-> Loading KerasNLP Gemma tokenizer with "
            f"preset `{tokenizer_preset}`..."
        )
        keras_nlp_tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(
            tokenizer_preset
        )
        # Save tokenizer state
        keras_nlp_tokenizer.save_assets(output_dir)
        vocab_path = os.path.join(output_dir, "vocabulary.spm")
        print("\n✅ Tokenizer loading complete.")

    hf_tokenizer = transformers.GemmaTokenizer(vocab_path)

    print(f"\n-> Saving HuggingFace Gemma tokenizer to `{output_dir}`...")
    # Save tokenizer to HF Transformers format
    hf_tokenizer.save_pretrained(output_dir)

    print(f"\n✅ Saving complete. Tokenizer saved at `{output_dir}`.")


def update_state_dict(layer, weight_name: str, tensor: torch.Tensor) -> None:
    """Updates the state dict for a weight given a tensor."""
    assert (
        tensor.shape == layer.state_dict()[weight_name].shape
    ), f"{tensor.shape} vs {layer.state_dict()[weight_name].shape}"
    layer.state_dict()[weight_name].copy_(tensor)


def flag_error_handler():
    if not FLAGS.preset and not FLAGS.weights_file:
        raise ValueError(
            "Please pass either a valid Keras preset to `--preset`"
            " or supply a Keras weights file (`.weights.h5`) and model size"
            " (`2b` or `7b`) to `--weights_file` and `--size`, respectively."
        )
    if FLAGS.weights_file:
        if FLAGS.preset:
            raise ValueError(
                "Both `--preset` and `--weights_file` flags cannot be supplied "
                "at the same time. Either supply a valid Keras preset to "
                "`--preset`or supply a Keras `.weights.h5` file and "
                "model size (`2b` or `7b`) to `--weights_file` and `--size`, "
                "respectively."
            )
        if not str(FLAGS.weights_file).endswith(".weights.h5"):
            raise ValueError(
                "Please pass a valid Keras weights file ending in `.weights.h5`."
            )
        if not FLAGS.size:
            raise ValueError(
                "The `size` flag must be passed if a weights file is passed. "
                "Please pass the appropriate size (`2b` or `7b`) for your "
                "model to the `--size` flag."
            )
        if FLAGS.size.lower() not in ["2b", "7b"]:
            raise ValueError(
                "Invalid `size`. Please pass the appropriate size (`2b` or `7b`) "
                "for your model to the `--size` flag."
            )
    if FLAGS.dtype:
        dtype = getattr(torch, FLAGS.dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(
                "Invalid `dtype`. Please pass a valid PyTorch data type (e.g. "
                "`float32', 'float16`, etc.) to the `--dtype` flag."
            )


def main(_):
    flag_error_handler()
    with _set_default_tensor_type(getattr(torch, FLAGS.dtype)):
        convert_checkpoints(
            FLAGS.preset,
            FLAGS.weights_file,
            FLAGS.size,
            FLAGS.output_dir,
            FLAGS.vocab_path,
        )


if __name__ == "__main__":
    flags.mark_flag_as_required("size")
    app.run(main)
