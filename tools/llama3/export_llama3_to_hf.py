"""
Export KerasHub Llama3 models to HuggingFace Transformers format.

This script converts Llama3 models from KerasHub format to HuggingFace
Transformers format with safetensors weights. The exported model can be
loaded directly with the `transformers` library.

Sample usage:

For converting a KerasHub preset to HuggingFace format:
```
python tools/llama3/export_llama3_to_hf.py \
    --preset llama3_8b_en \
    --output_dir ./llama3_hf/
```

For converting a custom fine-tuned checkpoint:
```
python tools/llama3/export_llama3_to_hf.py \
    --preset llama3_8b_en \
    --weights_file fine_tuned_llama3.weights.h5 \
    --output_dir ./fine_tuned_llama3_hf/
```

For converting with a specific precision:
```
python tools/llama3/export_llama3_to_hf.py \
    --preset llama3_instruct_8b_en \
    --output_dir ./llama3_hf/ \
    --dtype bfloat16
```
"""

import os
from typing import Optional

from absl import app
from absl import flags

os.environ["KERAS_BACKEND"] = "torch"

import keras  # noqa: E402

import keras_hub  # noqa: E402
from keras_hub.src.utils.transformers.export.hf_exporter import (  # noqa: E402
    export_to_safetensors,
)

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
    f"KerasHub preset name. Must be one of {', '.join(PRESET_MAP.keys())}. "
    "Alternatively, provide --weights_file for custom checkpoints.",
)
flags.DEFINE_string(
    "weights_file",
    None,
    "Path to a Keras weights file (`.weights.h5`). "
    "If provided with --preset, loads the preset architecture with "
    "custom weights. --preset must also be specified.",
)
flags.DEFINE_string(
    "output_dir",
    "llama3_hf",
    "Output directory for the converted HuggingFace model and tokenizer. "
    "Will be created if it doesn't exist.",
)
flags.DEFINE_string(
    "dtype",
    "float32",
    "Data type for the exported model. Must be a valid Keras floatx "
    "(e.g., 'float32', 'float16', 'bfloat16').",
)


def validate_flags():
    """Validate flag combinations and requirements."""
    if not FLAGS.preset and not FLAGS.weights_file:
        raise ValueError(
            "Either --preset or --weights_file must be provided. "
            f"Common presets: {', '.join(PRESET_MAP.keys())}"
        )

    if FLAGS.weights_file and not FLAGS.preset:
        raise ValueError(
            "When using --weights_file, --preset must also be provided "
            "to define the model architecture."
        )

    if FLAGS.weights_file and not FLAGS.weights_file.endswith(".weights.h5"):
        raise ValueError(
            "Weights file must have .weights.h5 extension. "
            f"Got: {FLAGS.weights_file}"
        )


def export_llama3(
    preset: Optional[str],
    weights_file: Optional[str],
    output_dir: str,
    dtype: str,
):
    """
    Export Llama3 model from KerasHub to HuggingFace format.

    Args:
        preset: KerasHub preset name or None
        weights_file: Path to custom weights file or None
        output_dir: Output directory for exported model
        dtype: Keras floatx string (e.g., 'float32')
    """
    print("=" * 80)
    print("Llama3 Model Export: KerasHub → HuggingFace Transformers")
    print("=" * 80)

    # Set precision
    print(f"\n📊 Setting precision to {dtype}")
    keras.config.set_floatx(dtype)

    # Load model
    print(f"\n📥 Loading KerasHub Llama3 model from preset: {preset}")
    if preset in PRESET_MAP:
        print(f"   Description: {PRESET_MAP[preset]}")
    model = keras_hub.models.Llama3CausalLM.from_preset(preset)

    if weights_file:
        print(f"\n⚙️  Loading custom weights from: {weights_file}")
        model.load_weights(weights_file)
        print("   ✅ Custom weights loaded")

    print("\n✅ Model loaded successfully")

    # Display model info
    backbone = model.backbone
    print("\n📋 Model Configuration:")
    print(f"   • Vocabulary size: {backbone.vocabulary_size:,}")
    print(f"   • Number of layers: {backbone.num_layers}")
    print(f"   • Query heads: {backbone.num_query_heads}")
    print(f"   • Key-value heads: {backbone.num_key_value_heads}")
    print(f"   • Hidden dimension: {backbone.hidden_dim:,}")
    print(f"   • Intermediate dimension: {backbone.intermediate_dim:,}")
    head_dim = backbone.hidden_dim // backbone.num_query_heads
    print(f"   • Head dimension: {head_dim}")
    print(f"   • RoPE max wavelength: {backbone.rope_max_wavelength:,}")
    if backbone.rope_frequency_adjustment_factor is not None:
        factor = backbone.rope_frequency_adjustment_factor
        print(f"   • RoPE scaling factor: {factor}")

    # Export to HuggingFace format
    print("\n🔄 Exporting to HuggingFace format...")
    print(f"   Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    export_to_safetensors(model, output_dir)

    print("\n✅ Export completed successfully!")

    # List exported files
    print(f"\n📁 Exported files in {output_dir}:")
    exported_files = sorted(os.listdir(output_dir))
    for file in exported_files:
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   • {file:30s} ({size_mb:.2f} MB)")

    print("\n" + "=" * 80)
    print("✅ Export Complete!")
    print("=" * 80)
    print("\nYou can now load the model with HuggingFace Transformers:")
    print("\n  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    print("\n  # Generate text")
    print("  inputs = tokenizer('Hello', return_tensors='pt')")
    print("  outputs = model.generate(**inputs, max_new_tokens=50)")
    print("  print(tokenizer.decode(outputs[0], skip_special_tokens=True))")
    print()


def main(_):
    validate_flags()

    export_llama3(
        preset=FLAGS.preset,
        weights_file=FLAGS.weights_file,
        output_dir=FLAGS.output_dir,
        dtype=FLAGS.dtype,
    )


if __name__ == "__main__":
    app.run(main)
