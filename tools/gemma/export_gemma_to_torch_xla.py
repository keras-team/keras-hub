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

"""
Prior to running this conversion script, please install the PyTorch
implementation of Gemma and `torch_xla`:

`pip install git+https://github.com/google/gemma_pytorch.git`
`pip install torch_xla`

Please also ensure that your installed versions of `torch_xla` and `torch` are
compatible.
"""

import contextlib
import os

import gemma
import torch
import torch_xla.core.xla_model as xm
from absl import app
from absl import flags
from gemma import model_xla as gemma_model

import keras_nlp

os.environ["KERAS_BACKEND"] = "torch"

"""
Sample usage:

For converting a Keras model to PyTorch format using a custom or fine-tuned
checkpoint from Keras, make sure to pass the path for the Keras weights file
(ending in `.weights.h5`) and the model size (`2b` or `7b`) to `--weights_file`
and `--size`, respectively.

Optionally, you can specify the output path for the converted model at
`--output_file`. (This defaults to `gemma.ckpt`)
```
python tools/gemma/export_gemma_to_torch_xla.py \
  --weights_file fine_tuned_imdb.weights.h5 \
  --size 2b \
  --output_file fine_tuned_imdb.ckpt
```

For converting a Keras model to PyTorch format from a preset,
simply pass the Keras preset name to `--preset`.
```
python tools/gemma/export_gemma_to_torch_xla.py   \
    --preset gemma_2b_en   \
    --output_file path/to/keras_torch_model.ckpt
```

Following this usage, you can run the verification script to confirm
functionality of the converted checkpoint:

```
python keras-nlp-gemma/tools/gemma/run_gemma_xla.py \
  --size 2b \
  --checkpoint_file fine_tuned_imdb.ckpt \
  --vocab_file gemma_tokenizer/vocabulary.spm \
  --prompt "Inception is about"
```
"""


PRESET_MAP = {
    "gemma_2b_en": gemma.config.get_config_for_2b(),
    "gemma_instruct_2b_en": gemma.config.get_config_for_2b(),
    "gemma_7b_en": gemma.config.get_config_for_7b(),
    "gemma_instruct_7b_en": gemma.config.get_config_for_7b(),
}

SIZE_MAP = {
    "2b": (gemma.config.get_config_for_2b(), "gemma_2b_en"),
    "7b": (gemma.config.get_config_for_7b(), "gemma_7b_en"),
}

FLAGS = flags.FLAGS
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
    "output_file",
    "gemma.ckpt",
    "An output file for the converted PyTorch checkpoint. Default: `gemma.ckpt`",
)
flags.DEFINE_string(
    "vocab_dir",
    "gemma_tokenizer",
    "A directory in which the vocabulary for the tokenizer will be stored.",
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


def _reconcile_attention_dims(qkv, target_shape):
    return torch.cat(qkv).reshape(tuple(target_shape))


def convert_checkpoints(preset, weights_file, size, output_file, vocab_dir):
    device = xm.xla_device()

    if preset is not None:
        print(
            f"\n-> Loading PyTorch Gemma model config for preset `{preset}`..."
        )
        model = gemma_model.GemmaForCausalLM(
            PRESET_MAP[preset], world_size=1, rank=0, device=device
        )
        print(f"\n-> Loading KerasNLP Gemma model with preset `{preset}`...")
        keras_nlp_model = keras_nlp.models.GemmaCausalLM.from_preset(preset)
    else:
        print(f"\n-> Loading PyTorch Gemma model config for `{size}` model...")
        config, size_preset = SIZE_MAP[size.lower()]
        model = gemma_model.GemmaForCausalLM(
            config, world_size=1, rank=0, device=device
        )
        print(f"\n-> Loading Keras weights from file `{weights_file}`...")
        keras_nlp_model = keras_nlp.models.GemmaCausalLM.from_preset(
            size_preset
        )
        keras_nlp_model.load_weights(weights_file)

    print("\n✅ Model loading complete.")
    print("\n-> Converting weights from KerasNLP Gemma to PyTorch Gemma...")

    # Token embedding (with vocab size difference handling)
    keras_embedding = keras_nlp_model.backbone.token_embedding.weights[0]
    torch_vocab_size = model.embedder.weight.shape[0]
    keras_nlp_vocab_size = keras_embedding.value.shape[0]
    if torch_vocab_size < keras_nlp_vocab_size:
        diff = keras_nlp_vocab_size - torch_vocab_size
        update_state_dict(
            model.embedder,
            "weight",
            keras_embedding.value[:-diff, :],
        )
    else:
        update_state_dict(
            model.embedder,
            "weight",
            keras_embedding.value,
        )

    # Decoder blocks
    for i in range(keras_nlp_model.backbone.num_layers):
        decoder_block = keras_nlp_model.backbone.get_layer(f"decoder_block_{i}")
        # Pre-attention norm
        update_state_dict(
            model.model.layers[i].input_layernorm,
            "weight",
            decoder_block.pre_attention_norm.weights[0].value,
        )

        # Attention
        qkv = (
            decoder_block.attention.query_dense.weights[0].value.transpose(
                1, 2
            ),
            decoder_block.attention.key_dense.weights[0].value.transpose(1, 2),
            decoder_block.attention.value_dense.weights[0].value.transpose(
                1, 2
            ),
        )
        qkv_target_shape = model.model.layers[i].self_attn.qkv_proj.weight.shape
        combined_tensor = _reconcile_attention_dims(qkv, qkv_target_shape)

        update_state_dict(
            model.model.layers[i].self_attn.qkv_proj, "weight", combined_tensor
        )

        out_target_shape = model.model.layers[i].self_attn.o_proj.weight.shape
        keras_out_tensor = decoder_block.attention.output_dense.weights[0].value
        out_tensor = keras_out_tensor.reshape(
            (out_target_shape[1], out_target_shape[0])  # Transpose target size
        ).transpose(0, 1)

        update_state_dict(
            model.model.layers[i].self_attn.o_proj, "weight", out_tensor
        )

        # Post-attention norm
        update_state_dict(
            model.model.layers[i].post_attention_layernorm,
            "weight",
            decoder_block.pre_ffw_norm.weights[0].value,
        )

        # MLP (Feed-forward)
        update_state_dict(
            model.model.layers[i].mlp.gate_proj,
            "weight",
            decoder_block.gating_ffw.weights[0].value.transpose(0, 1),
        )
        update_state_dict(
            model.model.layers[i].mlp.up_proj,
            "weight",
            decoder_block.gating_ffw_2.weights[0].value.transpose(0, 1),
        )
        update_state_dict(
            model.model.layers[i].mlp.down_proj,
            "weight",
            decoder_block.ffw_linear.weights[0].value.transpose(0, 1),
        )

    # Final norm
    update_state_dict(
        model.model.norm,
        "weight",
        keras_nlp_model.backbone.layers[-1].weights[0].value,
    )

    print("\n✅ Weights converted successfully.")
    print(f"\n-> Saving PyTorch model checkpoint to `{output_file}`...")

    # Save model checkpoint
    torch.save({"model_state_dict": model.state_dict()}, output_file)

    print(
        f"\n✅ Saving complete. Model checkpoint available at `{output_file}`."
    )

    if preset is not None:
        # Tokenizer
        print(
            f"\n-> Loading KerasNLP Gemma tokenizer with preset `{preset}`..."
        )
        keras_nlp_tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(
            preset
        )
        print("\n✅ Model loading complete.")
        print(f"\n-> Saving tokenizer state to directory `{vocab_dir}`...")

        # Save tokenizer state
        os.makedirs(vocab_dir, exist_ok=True)
        keras_nlp_tokenizer.save_assets(vocab_dir)

        print(
            "\n✅ Saving complete. Tokenizer state "
            f"available at `{vocab_dir}/vocabulary.spm`."
        )


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
            FLAGS.output_file,
            FLAGS.vocab_dir,
        )


if __name__ == "__main__":
    app.run(main)
