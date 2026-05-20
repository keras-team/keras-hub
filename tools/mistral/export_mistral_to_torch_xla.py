"""
Prior to running this conversion script, please install `torch_xla`:

`pip install torch_xla`

Please ensure that your installed versions of `torch_xla` and `torch` are
compatible. `mistral-inference` is NOT required for this script; it is only
needed by `run_mistral_xla.py`.

Sample usage:

For converting a Keras model to PyTorch format using a custom or fine-tuned
checkpoint from Keras, make sure to pass the path for the Keras weights file
(ending in `.weights.h5`) to `--weights_file` and a preset for the model
architecture to `--preset`.

Optionally, you can specify the output directory for the converted model at
`--output_dir`. (Defaults to `mistral_xla`)
```
python tools/mistral/export_mistral_to_torch_xla.py \
  --weights_file fine_tuned_mistral.weights.h5 \
  --preset mistral_7b_en \
  --output_dir fine_tuned_mistral_xla
```

For converting a Keras model to PyTorch format from a preset, simply pass
the Keras preset name to `--preset`.
```
python tools/mistral/export_mistral_to_torch_xla.py \
    --preset mistral_7b_en \
    --output_dir mistral_xla
```

Following this usage, you can run the verification script to confirm
functionality of the converted checkpoint:

```
python tools/mistral/run_mistral_xla.py \
  --checkpoint_dir mistral_xla \
  --prompt "Inception is about"
```
"""

import json
import os

import torch
from absl import app
from absl import flags

try:
    import torch_xla.core.xla_model as xm
except ImportError as e:
    raise ImportError(
        "torch_xla is required but could not be imported. "
        "Ensure torch and torch_xla versions match:\n"
        "  pip install torch_xla[tpu]==<torch_version> "
        "-f https://storage.googleapis.com/libtpu-releases/index.html"
    ) from e

os.environ["KERAS_BACKEND"] = "torch"

import keras  # noqa: E402
import keras.ops as ops  # noqa: E402
import keras_hub  # noqa: E402


PRESET_MAP = {
    "mistral_7b_en": "Base Mistral 7B",
    "mistral_0.3_7b_en": "Mistral 7B v0.3",
    "mistral_instruct_7b_en": "Instruction-tuned Mistral 7B",
    "mistral_0.2_instruct_7b_en": "Instruction-tuned Mistral 7B v0.2",
    "mistral_0.3_instruct_7b_en": "Instruction-tuned Mistral 7B v0.3",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}. "
    "Alternatively, a Keras weights file (`.weights.h5`) can be passed "
    "to --weights_file (preset still required to define architecture).",
)
flags.DEFINE_string(
    "weights_file",
    None,
    "A Keras weights file (`.weights.h5`). "
    "If supplied, --preset must also be given to define the architecture.",
)
flags.DEFINE_string(
    "output_dir",
    "mistral_xla",
    "Output directory for the converted PyTorch checkpoint, params.json, "
    "and tokenizer. Default: `mistral_xla`",
)
flags.DEFINE_string(
    "dtype",
    "float32",
    "Set the precision of the converted checkpoint. "
    "Must be a valid PyTorch dtype.",
)


def _to_torch(keras_var):
    """Convert a Keras variable to a CPU torch.Tensor."""
    return torch.from_numpy(ops.convert_to_numpy(keras_var))


def convert_checkpoints(preset, weights_file, output_dir):
    # ------------------------------------------------------------------ #
    # 1. Load KerasHub model
    # ------------------------------------------------------------------ #
    print(f"\n-> Loading KerasHub Mistral model from preset `{preset}`...")
    keras_model = keras_hub.models.MistralCausalLM.from_preset(preset)

    if weights_file:
        print(f"\n-> Loading custom weights from `{weights_file}`...")
        keras_model.load_weights(weights_file)

    print("\n✅ KerasHub model loaded.")

    backbone = keras_model.backbone
    head_dim = backbone.hidden_dim // backbone.num_query_heads

    # ------------------------------------------------------------------ #
    # 2. Build state dict directly (no mistral-inference needed at export)
    #
    # Key layout matches the mistral-inference Transformer state dict so
    # run_mistral_xla.py can load it with model.load_state_dict().
    # ------------------------------------------------------------------ #
    print("\n-> Converting weights from KerasHub Mistral to mistral-inference...")

    state_dict = {}

    # Token embedding  (vocab_size, dim)  — same layout in both
    state_dict["tok_embeddings.weight"] = _to_torch(
        backbone.token_embedding.embeddings
    )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.transformer_layers[i]
        attn = decoder_layer._self_attention_layer

        # ---- Attention projections ---- #
        # KerasHub Q kernel: (hidden_dim, num_query_heads, head_dim)
        # mistral-inference wq.weight: (num_query_heads * head_dim, hidden_dim)
        q = _to_torch(attn._query_dense.kernel)
        state_dict[f"layers.{i}.attention.wq.weight"] = (
            q.reshape(backbone.hidden_dim, -1).T.contiguous()
        )

        # KerasHub K kernel: (hidden_dim, num_key_value_heads, head_dim)
        # mistral-inference wk.weight: (num_kv_heads * head_dim, hidden_dim)
        k = _to_torch(attn._key_dense.kernel)
        state_dict[f"layers.{i}.attention.wk.weight"] = (
            k.reshape(backbone.hidden_dim, -1).T.contiguous()
        )

        # KerasHub V kernel: (hidden_dim, num_key_value_heads, head_dim)
        # mistral-inference wv.weight: (num_kv_heads * head_dim, hidden_dim)
        v = _to_torch(attn._value_dense.kernel)
        state_dict[f"layers.{i}.attention.wv.weight"] = (
            v.reshape(backbone.hidden_dim, -1).T.contiguous()
        )

        # KerasHub O kernel: (num_query_heads, head_dim, hidden_dim)
        # mistral-inference wo.weight: (hidden_dim, num_query_heads * head_dim)
        o = _to_torch(attn._output_dense.kernel)
        state_dict[f"layers.{i}.attention.wo.weight"] = (
            o.reshape(-1, backbone.hidden_dim).T.contiguous()
        )

        # ---- MLP (SwiGLU) ---- #
        # KerasHub gate kernel: (hidden_dim, intermediate_dim)
        # mistral-inference w1.weight (gate): (intermediate_dim, hidden_dim)
        gate = _to_torch(decoder_layer._feedforward_gate_dense.kernel)
        state_dict[f"layers.{i}.feed_forward.w1.weight"] = gate.T.contiguous()

        # KerasHub intermediate kernel: (hidden_dim, intermediate_dim)
        # mistral-inference w3.weight (up): (intermediate_dim, hidden_dim)
        up = _to_torch(decoder_layer._feedforward_intermediate_dense.kernel)
        state_dict[f"layers.{i}.feed_forward.w3.weight"] = up.T.contiguous()

        # KerasHub output kernel: (intermediate_dim, hidden_dim)
        # mistral-inference w2.weight (down): (hidden_dim, intermediate_dim)
        down = _to_torch(decoder_layer._feedforward_output_dense.kernel)
        state_dict[f"layers.{i}.feed_forward.w2.weight"] = down.T.contiguous()

        # ---- Layer norms ---- #
        state_dict[f"layers.{i}.attention_norm.weight"] = _to_torch(
            decoder_layer._self_attention_layernorm.scale
        )
        state_dict[f"layers.{i}.ffn_norm.weight"] = _to_torch(
            decoder_layer._feedforward_layernorm.scale
        )

    # Final layer norm
    state_dict["norm.weight"] = _to_torch(backbone.layer_norm.scale)

    # LM head: KerasHub reverse_embeddings (hidden_dim, vocab_size)
    # mistral-inference output.weight: (vocab_size, hidden_dim)
    state_dict["output.weight"] = _to_torch(
        backbone.token_embedding.reverse_embeddings
    ).T.contiguous()

    print("\n✅ Weights converted successfully.")

    # ------------------------------------------------------------------ #
    # 4. Save checkpoint + params.json + tokenizer
    # ------------------------------------------------------------------ #
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "mistral.ckpt")
    print(f"\n-> Saving PyTorch checkpoint to `{output_file}`...")
    torch.save({"model_state_dict": state_dict}, output_file)
    print(f"\n✅ Checkpoint saved to `{output_file}`.")

    params_file = os.path.join(output_dir, "params.json")
    print(f"\n-> Saving model config to `{params_file}`...")
    params = {
        "dim": backbone.hidden_dim,
        "n_layers": backbone.num_layers,
        "n_heads": backbone.num_query_heads,
        "n_kv_heads": backbone.num_key_value_heads,
        "head_dim": head_dim,
        "hidden_dim": backbone.intermediate_dim,
        "vocab_size": backbone.vocabulary_size,
        "norm_eps": backbone.layer_norm_epsilon,
        "sliding_window": backbone.sliding_window,
        "rope_theta": float(backbone.rope_max_wavelength),
    }
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\n✅ Config saved to `{params_file}`.")

    # Tokenizer
    print(f"\n-> Saving tokenizer to directory `{output_dir}`...")
    keras_tokenizer = keras_hub.models.MistralTokenizer.from_preset(preset)
    keras_tokenizer.save_assets(output_dir)
    # Rename vocabulary.spm → tokenizer.model (mistral-inference convention)
    spm_path = os.path.join(output_dir, "vocabulary.spm")
    tok_path = os.path.join(output_dir, "tokenizer.model")
    if os.path.exists(spm_path) and not os.path.exists(tok_path):
        os.rename(spm_path, tok_path)
    print(f"\n✅ Tokenizer saved. Vocabulary at `{tok_path}`.")

    print(f"\n✅ All done! Converted checkpoint available at `{output_dir}/`.")


def flag_error_handler():
    if not FLAGS.preset and not FLAGS.weights_file:
        raise ValueError(
            "Please pass either a valid Keras preset to `--preset` "
            "or supply a Keras weights file (`.weights.h5`) to `--weights_file`."
        )
    if FLAGS.weights_file:
        if not FLAGS.preset:
            raise ValueError(
                "The `--preset` flag must be given together with `--weights_file` "
                "to define the model architecture."
            )
        if not str(FLAGS.weights_file).endswith(".weights.h5"):
            raise ValueError(
                "Please pass a valid Keras weights file ending in `.weights.h5`."
            )
    if FLAGS.dtype:
        dtype = getattr(torch, FLAGS.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(
                "Invalid `dtype`. Please pass a valid PyTorch data type "
                "(e.g. `float32`, `float16`, `bfloat16`) to `--dtype`."
            )


def main(_):
    flag_error_handler()
    with torch.no_grad():
        convert_checkpoints(FLAGS.preset, FLAGS.weights_file, FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
