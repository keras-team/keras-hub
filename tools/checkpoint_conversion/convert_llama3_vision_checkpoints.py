"""
Convert Llama 3.2 Vision checkpoints from HuggingFace to Keras Hub format.

Usage:
    python tools/checkpoint_conversion/convert_llama3_vision_checkpoints.py \
        --preset llama3_2_vision_11b_instruct

Requirements:
    pip install transformers torch accelerate pillow
"""

import os
import traceback

import numpy as np
import torch
from absl import app
from absl import flags

os.environ["KERAS_BACKEND"] = "torch"

from keras import ops
from transformers import AutoProcessor
from transformers import MllamaForConditionalGeneration

from keras_hub.models import Llama3VisionBackbone

PRESET_MAP = {
    "llama3_2_vision_11b": "meta-llama/Llama-3.2-11B-Vision",
    "llama3_2_vision_11b_instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "llama3_2_vision_90b": "meta-llama/Llama-3.2-90B-Vision",
    "llama3_2_vision_90b_instruct": "meta-llama/Llama-3.2-90B-Vision-Instruct",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
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
        "vision_output_dim": vision_config.get("vision_output_dim", 7680),
        "cross_attention_layers": cross_attention_layers,
    }


def convert_vision_encoder_weights(keras_encoder, hf_model):
    """Convert vision encoder weights from HuggingFace to Keras."""
    hf_vision = hf_model.model.vision_model

    # Patch embedding
    keras_encoder.patch_embedding.kernel.assign(
        hf_vision.patch_embedding.weight.permute(2, 3, 1, 0)
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    # Position embedding - slice off CLS position if needed
    pos_emb = (
        hf_vision.gated_positional_embedding.embedding.detach()
        .cpu()
        .float()
        .numpy()
    )
    keras_shape = keras_encoder.position_embedding.embeddings.shape[0]
    if pos_emb.shape[0] == keras_shape + 1:
        pos_emb = pos_emb[1:]  # Skip CLS token position
    keras_encoder.position_embedding.embeddings.assign(pos_emb)

    # Transformer layers
    layers = (
        keras_encoder.local_transformer_layers
        if keras_encoder.is_two_stage
        else keras_encoder.transformer_layers
    )

    for i, keras_layer in enumerate(layers):
        hf_layer = hf_vision.transformer.layers[i]

        # Get attention config
        attn = keras_layer._self_attention_layer
        num_heads = attn._num_heads
        head_dim = attn._key_dim
        hidden_dim = num_heads * head_dim

        # Layer norms
        keras_layer._self_attention_layer_norm.gamma.assign(
            hf_layer.input_layernorm.weight.detach().cpu().float().numpy()
        )
        keras_layer._self_attention_layer_norm.beta.assign(
            hf_layer.input_layernorm.bias.detach().cpu().float().numpy()
        )

        # Self-attention QKV
        attn._query_dense.kernel.assign(
            hf_layer.self_attn.q_proj.weight.T.reshape(
                hidden_dim, num_heads, head_dim
            )
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        attn._key_dense.kernel.assign(
            hf_layer.self_attn.k_proj.weight.T.reshape(
                hidden_dim, num_heads, head_dim
            )
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        attn._value_dense.kernel.assign(
            hf_layer.self_attn.v_proj.weight.T.reshape(
                hidden_dim, num_heads, head_dim
            )
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        attn._output_dense.kernel.assign(
            hf_layer.self_attn.o_proj.weight.T.reshape(
                num_heads, head_dim, hidden_dim
            )
            .detach()
            .cpu()
            .float()
            .numpy()
        )

        # FFN layer norm
        keras_layer._feedforward_layer_norm.gamma.assign(
            hf_layer.post_attention_layernorm.weight.detach()
            .cpu()
            .float()
            .numpy()
        )
        keras_layer._feedforward_layer_norm.beta.assign(
            hf_layer.post_attention_layernorm.bias.detach()
            .cpu()
            .float()
            .numpy()
        )

        # FFN
        keras_layer._feedforward_intermediate_dense.kernel.assign(
            hf_layer.mlp.fc1.weight.T.detach().cpu().float().numpy()
        )
        keras_layer._feedforward_intermediate_dense.bias.assign(
            hf_layer.mlp.fc1.bias.detach().cpu().float().numpy()
        )
        keras_layer._feedforward_output_dense.kernel.assign(
            hf_layer.mlp.fc2.weight.T.detach().cpu().float().numpy()
        )
        keras_layer._feedforward_output_dense.bias.assign(
            hf_layer.mlp.fc2.bias.detach().cpu().float().numpy()
        )

    # Final layer norm - check available norm attributes
    if hasattr(hf_vision, "post_layernorm"):
        ln = hf_vision.post_layernorm
    elif hasattr(hf_vision, "layernorm_pre"):
        ln = hf_vision.layernorm_pre
    elif hasattr(hf_vision, "layernorm_post"):
        ln = hf_vision.layernorm_post
    else:
        print(
            "   Available vision attrs:",
            [a for a in dir(hf_vision) if "norm" in a.lower()],
        )
        raise AttributeError("Could not find layer norm in vision model")

    keras_encoder.layer_norm.gamma.assign(
        ln.weight.detach().cpu().float().numpy()
    )
    keras_encoder.layer_norm.beta.assign(ln.bias.detach().cpu().float().numpy())


def convert_vision_projector_weights(keras_projector, hf_model):
    """Convert vision projector weights."""
    # HF uses a single nn.Linear, Keras now uses single Dense layer
    hf_proj = hf_model.model.multi_modal_projector

    # Single Linear layer
    keras_projector.projection.kernel.assign(
        hf_proj.weight.T.detach().cpu().float().numpy()
    )
    keras_projector.projection.bias.assign(
        hf_proj.bias.detach().cpu().float().numpy()
    )


def convert_text_backbone_weights(keras_text, hf_model):
    """Convert text backbone (Llama3) weights."""
    hf_text = hf_model.model.language_model

    # Token embedding
    keras_text.token_embedding.embeddings.assign(
        hf_text.embed_tokens.weight.detach().cpu().float().numpy()
    )

    for i, keras_layer in enumerate(keras_text.transformer_layers):
        hf_layer = hf_text.layers[i]

        # Get attention config
        attn = keras_layer._self_attention_layer
        num_heads = attn.num_query_heads
        num_kv_heads = attn.num_key_value_heads
        head_dim = keras_text.hidden_dim // num_heads
        hidden_dim = keras_text.hidden_dim

        # Input layer norm (RMSNorm - only has scale weight)
        keras_layer._self_attention_layernorm.scale.assign(
            hf_layer.input_layernorm.weight.detach().cpu().float().numpy()
        )

        # Self-attention
        attn._query_dense.kernel.assign(
            hf_layer.self_attn.q_proj.weight.T.reshape(
                hidden_dim, num_heads, head_dim
            )
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        attn._key_dense.kernel.assign(
            hf_layer.self_attn.k_proj.weight.T.reshape(
                hidden_dim, num_kv_heads, head_dim
            )
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        attn._value_dense.kernel.assign(
            hf_layer.self_attn.v_proj.weight.T.reshape(
                hidden_dim, num_kv_heads, head_dim
            )
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        attn._output_dense.kernel.assign(
            hf_layer.self_attn.o_proj.weight.T.reshape(
                num_heads, head_dim, hidden_dim
            )
            .detach()
            .cpu()
            .float()
            .numpy()
        )

        # Post-attention layer norm (RMSNorm)
        keras_layer._feedforward_layernorm.scale.assign(
            hf_layer.post_attention_layernorm.weight.detach()
            .cpu()
            .float()
            .numpy()
        )

        # MLP - Llama uses gate_proj, up_proj, down_proj
        keras_layer._feedforward_gate_dense.kernel.assign(
            hf_layer.mlp.gate_proj.weight.T.detach().cpu().float().numpy()
        )
        keras_layer._feedforward_intermediate_dense.kernel.assign(
            hf_layer.mlp.up_proj.weight.T.detach().cpu().float().numpy()
        )
        keras_layer._feedforward_output_dense.kernel.assign(
            hf_layer.mlp.down_proj.weight.T.detach().cpu().float().numpy()
        )

    # Final layer norm (RMSNorm)
    keras_text.layer_norm.scale.assign(
        hf_text.norm.weight.detach().cpu().float().numpy()
    )


def convert_cross_attention_weights(keras_ca_blocks, hf_model):
    """Convert cross-attention layer weights."""
    hf_layers = hf_model.model.language_model.layers

    for layer_idx, keras_ca in keras_ca_blocks.items():
        hf_ca = hf_layers[layer_idx].cross_attn

        # Norms (LlamaLayerNorm uses .scale not .gamma)
        keras_ca.query_norm.scale.assign(
            hf_ca.q_norm.weight.detach().cpu().float().numpy()
        )
        keras_ca.kv_norm.scale.assign(
            hf_ca.k_norm.weight.detach().cpu().float().numpy()
        )

        # Projections
        keras_ca.query_dense.kernel.assign(
            hf_ca.q_proj.weight.T.detach().cpu().float().numpy()
        )
        keras_ca.key_dense.kernel.assign(
            hf_ca.k_proj.weight.T.detach().cpu().float().numpy()
        )
        keras_ca.value_dense.kernel.assign(
            hf_ca.v_proj.weight.T.detach().cpu().float().numpy()
        )
        keras_ca.output_dense.kernel.assign(
            hf_ca.o_proj.weight.T.detach().cpu().float().numpy()
        )

        # Gate
        keras_ca.gate.assign(
            hf_layers[layer_idx]
            .cross_attn_attn_gate.detach()
            .cpu()
            .float()
            .numpy()
        )


def convert_checkpoints(keras_model, hf_model):
    """Convert all weights from HuggingFace to Keras."""
    print("\n-> Converting vision encoder weights...")
    convert_vision_encoder_weights(keras_model.vision_encoder, hf_model)

    print("-> Converting vision projector weights...")
    convert_vision_projector_weights(keras_model.vision_projector, hf_model)

    print("-> Converting text backbone weights...")
    convert_text_backbone_weights(keras_model.text_backbone, hf_model)

    print("-> Converting cross-attention weights...")
    convert_cross_attention_weights(
        keras_model.cross_attention_blocks, hf_model
    )


def test_model(keras_model, hf_model, processor):
    """Test that the outputs of both models match."""
    print("\n-> Testing model outputs...")

    # Check parameter counts
    keras_params = keras_model.count_params()
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"   Keras params: {keras_params:,}")
    print(f"   HF params: {hf_params:,}")

    if keras_params != hf_params:
        print("   WARNING: Parameter count mismatch!")

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
    hf_hidden = hf_outputs.hidden_states[-1].detach().cpu().float().numpy()

    # Keras forward pass
    keras_inputs = {
        "images": test_image.astype(np.float32)[np.newaxis] / 255.0,
        "token_ids": hf_inputs["input_ids"].numpy(),
        "padding_mask": hf_inputs.get(
            "attention_mask", torch.ones_like(hf_inputs["input_ids"])
        ).numpy(),
    }
    keras_hidden = ops.convert_to_numpy(keras_model(keras_inputs))

    # Compare
    min_len = min(hf_hidden.shape[1], keras_hidden.shape[1])
    hf_slice = hf_hidden[:, :min_len]
    keras_slice = keras_hidden[:, :min_len]

    try:
        np.testing.assert_allclose(keras_slice, hf_slice, atol=1e-4)
        print("   Outputs match within tolerance 1e-4!")
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")


def main(_):
    # Validate preset
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. "
            f"Must be one of {','.join(PRESET_MAP.keys())}"
        )

    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    print("=" * 60)
    print("LLAMA 3.2 VISION CHECKPOINT CONVERSION")
    print("=" * 60)
    print(f"Preset: {preset}")
    print(f"HuggingFace: {hf_preset}")
    print()

    # Load HuggingFace model
    print("-> Loading HuggingFace model...")
    processor = AutoProcessor.from_pretrained(hf_preset)
    hf_model = MllamaForConditionalGeneration.from_pretrained(
        hf_preset,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    hf_model.eval()
    hf_config = hf_model.config.to_dict()
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"   HuggingFace model loaded: {hf_params:,} params")

    # Create Keras backbone
    print("\n-> Creating Keras backbone...")
    keras_config = convert_backbone_config(hf_config)
    keras_model = Llama3VisionBackbone(**keras_config)
    print(f"   Keras backbone created: {keras_model.count_params():,} params")

    # Convert weights
    convert_checkpoints(keras_model, hf_model)
    print("\n-> Weight transfer done.")

    # Test outputs
    test_model(keras_model, hf_model, processor)

    # Save preset
    keras_model.save_to_preset(preset)
    print(f"\n-> Saved the model preset to `{preset}`")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
