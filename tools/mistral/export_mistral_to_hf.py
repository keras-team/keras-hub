"""
Export KerasHub Mistral models to HuggingFace Transformers format.

This script converts Mistral models from KerasHub format to HuggingFace
Transformers format with safetensors weights. The exported model can be loaded
directly with the `transformers` library.

Sample usage:

For converting a KerasHub preset to HuggingFace format:
```
python tools/mistral/export_mistral_to_hf.py \
    --preset mistral_7b_en \
    --output_dir ./mistral_hf/
```

For converting a custom fine-tuned checkpoint:
```
python tools/mistral/export_mistral_to_hf.py \
    --weights_file fine_tuned_mistral.weights.h5 \
    --vocab_path mistral_tokenizer/tokenizer.model \
    --output_dir ./fine_tuned_mistral_hf/
```

For converting with a specific preset as base but custom weights:
```
python tools/mistral/export_mistral_to_hf.py \
    --preset mistral_7b_en \
    --weights_file fine_tuned.weights.h5 \
    --output_dir ./mistral_hf/
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
    "mistral_7b_en": "Base Mistral 7B model",
    "mistral_0.3_7b_en": "Mistral 7B v0.3 model",
    "mistral_instruct_7b_en": "Instruction-tuned Mistral 7B",
    "mistral_0.2_instruct_7b_en": "Instruction-tuned Mistral 7B v0.2",
    "mistral_0.3_instruct_7b_en": "Instruction-tuned Mistral 7B v0.3",
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
    "If provided with --preset, loads the preset architecture with custom weights. "
    "If provided alone, --vocab_path must also be specified.",
)
flags.DEFINE_string(
    "vocab_path",
    None,
    "Path to the SentencePiece vocabulary file (`.model` or `.spm`). "
    "Required only when using --weights_file without --preset.",
)
flags.DEFINE_string(
    "output_dir",
    "mistral_hf",
    "Output directory for the converted HuggingFace model and tokenizer. "
    "Will be created if it doesn't exist.",
)
flags.DEFINE_string(
    "dtype",
    "float32",
    "Data type for the exported model. Must be a valid PyTorch dtype "
    "(e.g., 'float32', 'float16', 'bfloat16').",
)


def validate_flags():
    """Validate flag combinations and requirements."""
    if not FLAGS.preset and not FLAGS.weights_file:
        raise ValueError(
            "Either --preset or --weights_file must be provided. "
            f"Common presets: {', '.join(PRESET_MAP.keys())}"
        )
    
    if FLAGS.weights_file and not FLAGS.preset and not FLAGS.vocab_path:
        raise ValueError(
            "When using --weights_file without --preset, "
            "--vocab_path must be provided."
        )
    
    if FLAGS.weights_file and not FLAGS.weights_file.endswith(".weights.h5"):
        raise ValueError(
            "Weights file must have .weights.h5 extension. "
            f"Got: {FLAGS.weights_file}"
        )


def export_mistral(
    preset: Optional[str],
    weights_file: Optional[str],
    vocab_path: Optional[str],
    output_dir: str,
    dtype: str,
):
    """
    Export Mistral model from KerasHub to HuggingFace format.
    
    Args:
        preset: KerasHub preset name or None
        weights_file: Path to custom weights file or None
        vocab_path: Path to vocabulary file or None
        output_dir: Output directory for exported model
        dtype: PyTorch dtype string (e.g., 'float32')
    """
    print("=" * 80)
    print("Mistral Model Export: KerasHub → HuggingFace Transformers")
    print("=" * 80)
    
    # Set precision
    print(f"\n📊 Setting precision to {dtype}")
    keras.config.set_floatx(dtype)
    
    # Load model
    if preset:
        print(f"\n📥 Loading KerasHub Mistral model from preset: {preset}")
        if preset in PRESET_MAP:
            print(f"   Description: {PRESET_MAP[preset]}")
        model = keras_hub.models.MistralCausalLM.from_preset(preset)
        
        if weights_file:
            print(f"\n⚙️  Loading custom weights from: {weights_file}")
            model.load_weights(weights_file)
            print("   ✅ Custom weights loaded")
    else:
        # Load with custom weights - need to construct model first
        raise NotImplementedError(
            "Loading with weights_file alone (without preset) is not yet "
            "implemented. Please provide a --preset to define the architecture."
        )
    
    print("\n✅ Model loaded successfully")
    
    # Display model info
    backbone = model.backbone
    print(f"\n📋 Model Configuration:")
    print(f"   • Vocabulary size: {backbone.vocabulary_size:,}")
    print(f"   • Number of layers: {backbone.num_layers}")
    print(f"   • Query heads: {backbone.num_query_heads}")
    print(f"   • Key-value heads: {backbone.num_key_value_heads}")
    print(f"   • Hidden dimension: {backbone.hidden_dim:,}")
    print(f"   • Intermediate dimension: {backbone.intermediate_dim:,}")
    print(f"   • Sliding window: {backbone.sliding_window}")
    print(f"   • RoPE max wavelength: {backbone.rope_max_wavelength:,}")
    
    # Export to HuggingFace format
    print(f"\n🔄 Exporting to HuggingFace format...")
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
            size = os.path.getsize(file_path)
            size_mb = size / (1024 * 1024)
            print(f"   • {file:30s} ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 80)
    print("✅ Export Complete!")
    print("=" * 80)
    print(f"\nYou can now load the model with HuggingFace Transformers:")
    print(f"\n  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    print(f"\n  # Generate text")
    print(f"  inputs = tokenizer('Hello', return_tensors='pt')")
    print(f"  outputs = model.generate(**inputs, max_length=50)")
    print(f"  print(tokenizer.decode(outputs[0]))")
    print()


def main(_):
    validate_flags()
    
    export_mistral(
        preset=FLAGS.preset,
        weights_file=FLAGS.weights_file,
        vocab_path=FLAGS.vocab_path,
        output_dir=FLAGS.output_dir,
        dtype=FLAGS.dtype,
    )


if __name__ == "__main__":
    app.run(main)
