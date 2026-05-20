"""
Prior to running this conversion script, please install `torch_xla`:

`pip install torch_xla`

Please ensure that your installed versions of `torch_xla` and `torch` are
compatible. The HuggingFace `transformers` library is NOT required for this
script; it is only needed by `run_llama3_xla.py`.

Sample usage:

For converting a fine-tuned KerasHub checkpoint to XLA/PyTorch format:
```
python tools/llama3/export_llama3_to_torch_xla.py \
    --weights_file fine_tuned_llama3.weights.h5 \
    --preset llama3_8b_en \
    --output_dir fine_tuned_llama3_xla
```

For converting a KerasHub preset to XLA/PyTorch format:
```
python tools/llama3/export_llama3_to_torch_xla.py \
    --preset llama3_8b_en \
    --output_dir llama3_xla
```

Following this, run `run_llama3_xla.py` to verify the checkpoint:
```
python tools/llama3/run_llama3_xla.py \
    --checkpoint_dir llama3_xla \
    --prompt "The capital of France is"
```
"""

import json
import os

import torch
from absl import app
from absl import flags

try:
    import torch_xla  # noqa: F401
except ImportError as e:
    raise ImportError(
        "torch_xla is required but could not be imported. "
        "Ensure torch and torch_xla versions match:\n"
        "  pip install torch_xla[tpu]==<torch_version> "
        "-f https://storage.googleapis.com/libtpu-releases/index.html"
    ) from e

os.environ["KERAS_BACKEND"] = "torch"

import keras.ops as ops  # noqa: E402

import keras_hub  # noqa: E402

PRESET_MAP = {
    "llama3_8b_en": "Base Llama 3 8B model",
    "llama3_8b_en_int8": "Llama 3 8B (int8 quantized)",
    "llama3_instruct_8b_en": "Instruction-tuned Llama 3 8B",
    "llama3_instruct_8b_en_int8": (
        "Instruction-tuned Llama 3 8B (int8 quantized)"
    ),
    "llama3.1_8b": "Llama 3.1 8B base model",
    "llama3.1_instruct_8b": "Instruction-tuned Llama 3.1 8B",
    "llama3.1_guard_8b": "Llama Guard 3.1 8B",
    "llama3.2_1b": "Llama 3.2 1B base model",
    "llama3.2_instruct_1b": "Instruction-tuned Llama 3.2 1B",
    "llama3.2_3b": "Llama 3.2 3B base model",
    "llama3.2_instruct_3b": "Instruction-tuned Llama 3.2 3B",
    "llama3.2_guard_1b": "Llama Guard 3.2 1B",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {', '.join(PRESET_MAP.keys())}. "
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
    "llama3_xla",
    "Output directory for the converted PyTorch checkpoint, params.json, "
    "and tokenizer.json. Default: `llama3_xla`",
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
    print(f"\n-> Loading KerasHub Llama3 model from preset `{preset}`...")
    keras_model = keras_hub.models.Llama3CausalLM.from_preset(preset)

    if weights_file:
        print(f"\n-> Loading custom weights from `{weights_file}`...")
        keras_model.load_weights(weights_file)

    print("\n✅ KerasHub model loaded.")

    backbone = keras_model.backbone
    head_dim = backbone.hidden_dim // backbone.num_query_heads

    # ------------------------------------------------------------------ #
    # 2. Build state dict
    #
    # Key layout matches the LlamaModel state dict defined in
    # run_llama3_xla.py so that script can load it with
    # model.load_state_dict().
    # ------------------------------------------------------------------ #
    print(
        "\n-> Converting weights from KerasHub Llama3 to XLA/PyTorch format..."
    )

    state_dict = {}

    # Token embedding  (vocab_size, dim)
    state_dict["tok_embeddings.weight"] = _to_torch(
        backbone.token_embedding.embeddings
    )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")
        attn = decoder_layer._self_attention_layer

        # ---- Attention projections ---- #
        # KerasHub Q kernel: (hidden_dim, num_query_heads, head_dim)
        # LlamaModel wq.weight: (num_query_heads * head_dim, hidden_dim)
        q = _to_torch(attn._query_dense.kernel)
        state_dict[f"layers.{i}.attention.wq.weight"] = q.reshape(
            backbone.hidden_dim, -1
        ).T.contiguous()

        # KerasHub K kernel: (hidden_dim, num_key_value_heads, head_dim)
        # LlamaModel wk.weight: (num_kv_heads * head_dim, hidden_dim)
        k = _to_torch(attn._key_dense.kernel)
        state_dict[f"layers.{i}.attention.wk.weight"] = k.reshape(
            backbone.hidden_dim, -1
        ).T.contiguous()

        # KerasHub V kernel: (hidden_dim, num_key_value_heads, head_dim)
        # LlamaModel wv.weight: (num_kv_heads * head_dim, hidden_dim)
        v = _to_torch(attn._value_dense.kernel)
        state_dict[f"layers.{i}.attention.wv.weight"] = v.reshape(
            backbone.hidden_dim, -1
        ).T.contiguous()

        # KerasHub O kernel: (num_query_heads, head_dim, hidden_dim)
        # LlamaModel wo.weight: (hidden_dim, num_query_heads * head_dim)
        o = _to_torch(attn._output_dense.kernel)
        state_dict[f"layers.{i}.attention.wo.weight"] = o.reshape(
            -1, backbone.hidden_dim
        ).T.contiguous()

        # ---- MLP (SwiGLU) ---- #
        # KerasHub gate kernel: (hidden_dim, intermediate_dim)
        # LlamaModel w1.weight (gate): (intermediate_dim, hidden_dim)
        gate = _to_torch(decoder_layer._feedforward_gate_dense.kernel)
        state_dict[f"layers.{i}.feed_forward.w1.weight"] = gate.T.contiguous()

        # KerasHub up kernel: (hidden_dim, intermediate_dim)
        # LlamaModel w3.weight (up): (intermediate_dim, hidden_dim)
        up = _to_torch(decoder_layer._feedforward_intermediate_dense.kernel)
        state_dict[f"layers.{i}.feed_forward.w3.weight"] = up.T.contiguous()

        # KerasHub down kernel: (intermediate_dim, hidden_dim)
        # LlamaModel w2.weight (down): (hidden_dim, intermediate_dim)
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
    state_dict["norm.weight"] = _to_torch(
        backbone.get_layer("sequence_output_layernorm").scale
    )

    # LM head: KerasHub reverse_embeddings (hidden_dim, vocab_size)
    # LlamaModel output.weight: (vocab_size, hidden_dim)
    state_dict["output.weight"] = _to_torch(
        backbone.token_embedding.reverse_embeddings
    ).T.contiguous()

    print("\n✅ Weights converted successfully.")

    # ------------------------------------------------------------------ #
    # 4. Save checkpoint + params.json + tokenizer
    # ------------------------------------------------------------------ #
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "llama3.ckpt")
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
        "rope_theta": float(backbone.rope_max_wavelength),
    }
    # Llama 3.1+ uses scaled RoPE; save the adjustment factor if present.
    if backbone.rope_frequency_adjustment_factor is not None:
        params["rope_frequency_adjustment_factor"] = float(
            backbone.rope_frequency_adjustment_factor
        )
        params["rope_low_freq_factor"] = float(backbone.rope_low_freq_factor)
        params["rope_high_freq_factor"] = float(backbone.rope_high_freq_factor)
        params["rope_pretraining_sequence_length"] = int(
            backbone.rope_pretraining_sequence_length
        )
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\n✅ Config saved to `{params_file}`.")

    # Tokenizer: save BPE tokenizer.json (compatible with
    # transformers.PreTrainedTokenizerFast used by run_llama3_xla.py)
    print(f"\n-> Saving tokenizer to directory `{output_dir}`...")
    keras_tokenizer = keras_hub.models.Llama3Tokenizer.from_preset(preset)
    keras_tokenizer.save_assets(output_dir)
    # save_assets() writes vocabulary.json + merges.txt; rename/rebuild
    # to the tokenizer.json format expected by run_llama3_xla.py.
    # The easiest approach: export via hf_exporter which produces
    # tokenizer.json in one step.
    from keras_hub.src.utils.transformers.export.llama3 import (
        build_llama3_tokenizer_json,
    )
    from keras_hub.src.utils.transformers.export.llama3 import (
        get_llama3_tokenizer_config,
    )

    tokenizer_json = build_llama3_tokenizer_json(keras_tokenizer)
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

    tokenizer_config = get_llama3_tokenizer_config(keras_tokenizer)
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Tokenizer saved. Fast tokenizer at `{tokenizer_json_path}`.")

    print(f"\n✅ All done! Converted checkpoint available at `{output_dir}/`.")


def flag_error_handler():
    if not FLAGS.preset and not FLAGS.weights_file:
        raise ValueError(
            "Please pass either a valid Keras preset to `--preset` "
            "or supply a Keras weights file (`.weights.h5`) to"
            " `--weights_file`."
        )
    if FLAGS.weights_file:
        if not FLAGS.preset:
            raise ValueError(
                "The `--preset` flag must be given together with"
                " `--weights_file` to define the model architecture."
            )
        if not str(FLAGS.weights_file).endswith(".weights.h5"):
            raise ValueError(
                "Please pass a valid Keras weights file ending"
                " in `.weights.h5`."
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
