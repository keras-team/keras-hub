"""
Convert Llama 3.2 Vision checkpoints from HuggingFace to Keras Hub format.

Usage:
    python tools/checkpoint_conversion/convert_llama3_vision_checkpoints.py \
        --preset meta-llama/Llama-3.2-11B-Vision-Instruct \
        --output_dir llama3_2_vision_11b_instruct

Requirements:
    pip install transformers torch accelerate pillow

Supported presets:
    - meta-llama/Llama-3.2-11B-Vision
    - meta-llama/Llama-3.2-11B-Vision-Instruct
    - meta-llama/Llama-3.2-90B-Vision
    - meta-llama/Llama-3.2-90B-Vision-Instruct
"""

import argparse
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import torch
from transformers import AutoProcessor
from transformers import MllamaForConditionalGeneration

from keras_hub.src.models.llama3.llama3_vision_backbone import (
    Llama3VisionBackbone,
)


def convert_backbone_config(hf_config):
    """Convert HuggingFace config to Keras Hub backbone kwargs."""
    vision_config = hf_config.get("vision_config", {})
    text_config = hf_config.get("text_config", {})

    num_text_layers = text_config.get("num_hidden_layers", 40)
    cross_attention_layers = hf_config.get(
        "cross_attention_layers",
        [i for i in range(3, num_text_layers, 5)],
    )

    return {
        "vocabulary_size": text_config.get("vocab_size", 128256),
        "num_layers": num_text_layers,
        "hidden_dim": text_config.get("hidden_size", 4096),
        "num_query_heads": text_config.get("num_attention_heads", 32),
        "num_key_value_heads": text_config.get("num_key_value_heads", 8),
        "intermediate_dim": text_config.get("intermediate_size", 14336),
        "rope_max_wavelength": text_config.get("rope_theta", 500000),
        "layer_norm_epsilon": text_config.get("rms_norm_eps", 1e-5),
        "vision_hidden_dim": vision_config.get("hidden_size", 1280),
        "vision_num_layers": vision_config.get("num_hidden_layers", 32),
        "vision_num_heads": vision_config.get("num_attention_heads", 16),
        "vision_intermediate_dim": vision_config.get("intermediate_size", 5120),
        "vision_patch_size": vision_config.get("patch_size", 14),
        "vision_image_size": vision_config.get("image_size", 560),
        "vision_num_channels": vision_config.get("num_channels", 3),
        "cross_attention_layers": cross_attention_layers,
    }


def convert_vision_encoder_weights(
    keras_encoder, hf_state_dict, prefix="vision_model"
):
    """Convert vision encoder weights."""

    def transpose_conv(x):
        return np.transpose(x, (2, 3, 1, 0))

    # Patch embedding
    keras_encoder.patch_embedding.kernel.assign(
        transpose_conv(
            hf_state_dict[f"{prefix}.patch_embedding.weight"].numpy()
        )
    )

    # Position embedding
    keras_encoder.position_embedding.embeddings.assign(
        hf_state_dict[f"{prefix}.position_embedding.weight"].numpy()
    )

    # Transformer layers
    if keras_encoder.is_two_stage:
        layers = keras_encoder.local_transformer_layers
        layer_prefix = f"{prefix}.encoder.layers"
    else:
        layers = keras_encoder.transformer_layers
        layer_prefix = f"{prefix}.encoder.layers"

    for i, layer in enumerate(layers):
        lp = f"{layer_prefix}.{i}"

        # Self-attention layer norm
        layer._self_attention_layernorm.gamma.assign(
            hf_state_dict[f"{lp}.layer_norm1.weight"].numpy()
        )
        layer._self_attention_layernorm.beta.assign(
            hf_state_dict[f"{lp}.layer_norm1.bias"].numpy()
        )

        # Self-attention
        layer._self_attention_layer._query_dense.kernel.assign(
            hf_state_dict[f"{lp}.self_attn.q_proj.weight"].numpy().T
        )
        layer._self_attention_layer._query_dense.bias.assign(
            hf_state_dict[f"{lp}.self_attn.q_proj.bias"].numpy()
        )
        layer._self_attention_layer._key_dense.kernel.assign(
            hf_state_dict[f"{lp}.self_attn.k_proj.weight"].numpy().T
        )
        layer._self_attention_layer._key_dense.bias.assign(
            hf_state_dict[f"{lp}.self_attn.k_proj.bias"].numpy()
        )
        layer._self_attention_layer._value_dense.kernel.assign(
            hf_state_dict[f"{lp}.self_attn.v_proj.weight"].numpy().T
        )
        layer._self_attention_layer._value_dense.bias.assign(
            hf_state_dict[f"{lp}.self_attn.v_proj.bias"].numpy()
        )
        layer._self_attention_layer._output_dense.kernel.assign(
            hf_state_dict[f"{lp}.self_attn.o_proj.weight"].numpy().T
        )
        layer._self_attention_layer._output_dense.bias.assign(
            hf_state_dict[f"{lp}.self_attn.o_proj.bias"].numpy()
        )

        # FFN layer norm
        layer._feedforward_layernorm.gamma.assign(
            hf_state_dict[f"{lp}.layer_norm2.weight"].numpy()
        )
        layer._feedforward_layernorm.beta.assign(
            hf_state_dict[f"{lp}.layer_norm2.bias"].numpy()
        )

        # FFN
        layer._feedforward_intermediate_dense.kernel.assign(
            hf_state_dict[f"{lp}.mlp.fc1.weight"].numpy().T
        )
        layer._feedforward_intermediate_dense.bias.assign(
            hf_state_dict[f"{lp}.mlp.fc1.bias"].numpy()
        )
        layer._feedforward_output_dense.kernel.assign(
            hf_state_dict[f"{lp}.mlp.fc2.weight"].numpy().T
        )
        layer._feedforward_output_dense.bias.assign(
            hf_state_dict[f"{lp}.mlp.fc2.bias"].numpy()
        )

    # Final layer norm
    keras_encoder.layer_norm.gamma.assign(
        hf_state_dict[f"{prefix}.post_layernorm.weight"].numpy()
    )
    keras_encoder.layer_norm.beta.assign(
        hf_state_dict[f"{prefix}.post_layernorm.bias"].numpy()
    )

    params = keras_encoder.count_params()
    print(f"  Vision encoder weights converted: {params:,} params")


def convert_vision_projector_weights(
    keras_projector, hf_state_dict, prefix="multi_modal_projector"
):
    """Convert vision projector weights."""
    keras_projector.dense_1.kernel.assign(
        hf_state_dict[f"{prefix}.linear_1.weight"].numpy().T
    )
    keras_projector.dense_1.bias.assign(
        hf_state_dict[f"{prefix}.linear_1.bias"].numpy()
    )
    keras_projector.dense_2.kernel.assign(
        hf_state_dict[f"{prefix}.linear_2.weight"].numpy().T
    )
    keras_projector.dense_2.bias.assign(
        hf_state_dict[f"{prefix}.linear_2.bias"].numpy()
    )

    params = keras_projector.count_params()
    print(f"  Vision projector weights converted: {params:,} params")


def convert_text_backbone_weights(
    keras_text_backbone, hf_state_dict, prefix="language_model"
):
    """Convert text backbone (Llama3) weights."""

    # Token embedding
    keras_text_backbone.token_embedding.embeddings.assign(
        hf_state_dict[f"{prefix}.model.embed_tokens.weight"].numpy()
    )

    # Transformer layers
    for i, layer in enumerate(keras_text_backbone.transformer_layers):
        lp = f"{prefix}.model.layers.{i}"

        # Input layer norm
        layer.attention_norm.weights[0].assign(
            hf_state_dict[f"{lp}.input_layernorm.weight"].numpy()
        )

        # Self-attention
        layer.attention.query_dense.kernel.assign(
            hf_state_dict[f"{lp}.self_attn.q_proj.weight"].numpy().T
        )
        layer.attention.key_dense.kernel.assign(
            hf_state_dict[f"{lp}.self_attn.k_proj.weight"].numpy().T
        )
        layer.attention.value_dense.kernel.assign(
            hf_state_dict[f"{lp}.self_attn.v_proj.weight"].numpy().T
        )
        layer.attention.output_dense.kernel.assign(
            hf_state_dict[f"{lp}.self_attn.o_proj.weight"].numpy().T
        )

        # Post-attention layer norm
        layer.post_attention_norm.weights[0].assign(
            hf_state_dict[f"{lp}.post_attention_layernorm.weight"].numpy()
        )

        # MLP
        layer.feedforward_gate_dense.kernel.assign(
            hf_state_dict[f"{lp}.mlp.gate_proj.weight"].numpy().T
        )
        layer.feedforward_intermediate_dense.kernel.assign(
            hf_state_dict[f"{lp}.mlp.up_proj.weight"].numpy().T
        )
        layer.feedforward_output_dense.kernel.assign(
            hf_state_dict[f"{lp}.mlp.down_proj.weight"].numpy().T
        )

    # Final layer norm
    keras_text_backbone.layer_norm.weights[0].assign(
        hf_state_dict[f"{prefix}.model.norm.weight"].numpy()
    )

    params = keras_text_backbone.count_params()
    print(f"  Text backbone weights converted: {params:,} params")


def convert_cross_attention_weights(
    keras_ca_blocks, hf_state_dict, prefix="language_model"
):
    """Convert cross-attention layer weights."""
    total_params = 0

    for layer_idx, keras_ca in keras_ca_blocks.items():
        lp = f"{prefix}.model.layers.{layer_idx}"

        # Query norm
        keras_ca.query_norm.gamma.assign(
            hf_state_dict[f"{lp}.cross_attn.q_norm.weight"].numpy()
        )

        # KV norm
        keras_ca.kv_norm.gamma.assign(
            hf_state_dict[f"{lp}.cross_attn.k_norm.weight"].numpy()
        )

        # Query projection
        keras_ca.query_dense.kernel.assign(
            hf_state_dict[f"{lp}.cross_attn.q_proj.weight"].numpy().T
        )

        # Key projection
        keras_ca.key_dense.kernel.assign(
            hf_state_dict[f"{lp}.cross_attn.k_proj.weight"].numpy().T
        )

        # Value projection
        keras_ca.value_dense.kernel.assign(
            hf_state_dict[f"{lp}.cross_attn.v_proj.weight"].numpy().T
        )

        # Output projection
        keras_ca.output_dense.kernel.assign(
            hf_state_dict[f"{lp}.cross_attn.o_proj.weight"].numpy().T
        )

        # Gate
        keras_ca.gate.assign(
            hf_state_dict[f"{lp}.cross_attn_attn_gate"].numpy()
        )

        total_params += keras_ca.count_params()

    num_layers = len(keras_ca_blocks)
    print(f"  Cross-attention: {total_params:,} ({num_layers} layers)")


def validate_outputs(hf_model, keras_backbone, processor, tolerance=1e-4):
    """Validate that Keras and HuggingFace outputs match."""
    print("\nValidating numerical parity...")

    # Create test input
    np.random.seed(42)
    test_image = np.random.randint(0, 255, (560, 560, 3), dtype=np.uint8)

    from PIL import Image

    pil_image = Image.fromarray(test_image)

    hf_inputs = processor(
        images=pil_image,
        text="Describe this image:",
        return_tensors="pt",
    )

    # HuggingFace forward pass
    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=hf_inputs["input_ids"],
            pixel_values=hf_inputs["pixel_values"],
            attention_mask=hf_inputs.get("attention_mask"),
            output_hidden_states=True,
        )
    hf_hidden = hf_outputs.hidden_states[-1].numpy()

    # Keras forward pass
    keras_inputs = {
        "images": test_image.astype(np.float32)[np.newaxis] / 255.0,
        "token_ids": hf_inputs["input_ids"].numpy(),
        "padding_mask": hf_inputs.get(
            "attention_mask", torch.ones_like(hf_inputs["input_ids"])
        ).numpy(),
    }
    keras_hidden = keras.ops.convert_to_numpy(keras_backbone(keras_inputs))

    # Compare
    min_len = min(hf_hidden.shape[1], keras_hidden.shape[1])
    max_diff = np.max(
        np.abs(hf_hidden[:, :min_len] - keras_hidden[:, :min_len])
    )
    mean_diff = np.mean(
        np.abs(hf_hidden[:, :min_len] - keras_hidden[:, :min_len])
    )

    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  Tolerance: {tolerance:.6e}")
    print(f"  Result: {'PASS' if max_diff <= tolerance else 'FAIL'}")

    return max_diff <= tolerance


def main(args):
    print("=" * 60)
    print("LLAMA 3.2 VISION CHECKPOINT CONVERSION")
    print("=" * 60)
    print(f"Preset: {args.preset}")
    print(f"Output: {args.output_dir}")
    print()

    # Load HuggingFace model
    print("Loading HuggingFace model...")
    processor = AutoProcessor.from_pretrained(args.preset)
    hf_model = MllamaForConditionalGeneration.from_pretrained(
        args.preset,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    hf_model.eval()
    hf_config = hf_model.config.to_dict()
    hf_state_dict = hf_model.state_dict()

    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"  HuggingFace model loaded: {hf_params:,} params")

    # Create Keras backbone
    print("\nCreating Keras backbone...")
    keras_config = convert_backbone_config(hf_config)
    backbone = Llama3VisionBackbone(**keras_config)
    print(f"  Keras backbone created: {backbone.count_params():,} params")

    # Convert weights
    print("\nConverting weights...")
    convert_vision_encoder_weights(backbone.vision_encoder, hf_state_dict)
    convert_vision_projector_weights(backbone.vision_projector, hf_state_dict)
    convert_text_backbone_weights(backbone.text_backbone, hf_state_dict)
    convert_cross_attention_weights(
        backbone.cross_attention_blocks, hf_state_dict
    )

    # Validate
    if args.validate:
        success = validate_outputs(hf_model, backbone, processor)
        if not success:
            print("\nWARNING: Validation failed! Outputs don't match.")

    # Save
    if args.output_dir:
        print(f"\nSaving to {args.output_dir}...")
        # Create CausalLM with preprocessor
        # TODO: Add proper tokenizer and preprocessor setup
        backbone.save_to_preset(args.output_dir)
        print("  Done!")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"HuggingFace params: {hf_params:,}")
    print(f"Keras params:       {backbone.count_params():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Llama 3.2 Vision weights to Keras Hub format."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="HuggingFace model ID or path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for Keras preset.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate outputs match HuggingFace (requires more memory).",
    )
    args = parser.parse_args()

    main(args)
