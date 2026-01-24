"""
Convert Llama 3.2 Vision checkpoints from HuggingFace to Keras Hub format.

Usage:
    python tools/checkpoint_conversion/convert_llama3_vision_checkpoints.py \
        --preset llama3_2_vision_11b_instruct

Requirements:
    pip install transformers torch accelerate pillow
"""

import os

import numpy as np
import torch
from absl import app
from absl import flags

os.environ["KERAS_BACKEND"] = "torch"

from keras import ops
from transformers import AutoProcessor
from transformers import MllamaForConditionalGeneration

from keras_hub.models import Llama3VisionBackbone
from keras_hub.models import (
    Llama3VisionCausalLM,  # Assuming you have this class
)

PRESET_MAP = {
    "llama3_2_vision_11b": "meta-llama/Llama-3.2-11B-Vision",
    "llama3_2_vision_11b_instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct",
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

    # HF stores cross attention indices in text_config.
    # Default to 11B structure (every 4th layer starting at 3)
    cross_attention_layers = text_config.get(
        "cross_attention_layers",
        [i for i in range(3, num_text_layers, 5)],
    )

    # HF calculates vision_output_dim as:
    # (num_intermediate_layers + 1) * hidden_size
    # intermediate_layers_indices defaults to [3, 7, 15, 23, 30] + final
    intermediate_layers = vision_config.get(
        "intermediate_layers_indices", [3, 7, 15, 23, 30]
    )
    vision_hidden = vision_config.get("hidden_size", 1280)
    # 6 * 1280 = 7680
    vision_output_dim = (len(intermediate_layers) + 1) * vision_hidden

    return {
        "vocabulary_size": text_config.get("vocab_size", 128256),
        "num_layers": num_text_layers,
        "hidden_dim": text_config.get("hidden_size", 4096),
        "num_query_heads": text_config.get("num_attention_heads", 32),
        "num_key_value_heads": text_config.get("num_key_value_heads", 8),
        "intermediate_dim": text_config.get("intermediate_size", 14336),
        "rope_max_wavelength": text_config.get("rope_theta", 500000),
        "layer_norm_epsilon": text_config.get("rms_norm_eps", 1e-5),
        "vision_hidden_dim": vision_hidden,
        "vision_num_layers": vision_config.get(
            "num_hidden_layers", 32
        ),  # Local layers
        "vision_global_layers": vision_config.get(
            "num_global_layers", 8
        ),  # Global layers
        "vision_num_heads": vision_config.get("attention_heads", 16),
        "vision_intermediate_dim": vision_config.get("intermediate_size", 5120),
        "vision_patch_size": vision_config.get("patch_size", 14),
        "vision_image_size": vision_config.get("image_size", 560),
        "vision_num_channels": vision_config.get("num_channels", 3),
        "vision_max_num_tiles": vision_config.get("max_num_tiles", 4),
        "vision_max_aspect_ratio_id": vision_config.get(
            "max_aspect_ratio_id", 8
        ),
        "vision_intermediate_layers_indices": intermediate_layers,
        "vision_output_dim": vision_output_dim,
        "cross_attention_layers": cross_attention_layers,
    }


def _convert_vision_transformer_layer(keras_layer, hf_layer, is_gated=False):
    """Convert a single vision transformer layer (local or global).

    Args:
        keras_layer: The Keras TransformerEncoder layer.
        hf_layer: The HuggingFace MllamaVisionEncoderLayer.
        is_gated: Whether this is a gated (global) layer with
            gate_attn/gate_ffn.
    """
    # Get attention config
    attn = keras_layer._self_attention_layer
    num_heads = attn._num_heads
    head_dim = attn._key_dim
    hidden_dim = num_heads * head_dim

    # Layer norms
    keras_layer._self_attention_layer_norm.gamma.assign(
        hf_layer.input_layernorm.weight.detach().cpu().numpy()
    )
    keras_layer._self_attention_layer_norm.beta.assign(
        hf_layer.input_layernorm.bias.detach().cpu().numpy()
    )

    # Self-attention QKV
    attn._query_dense.kernel.assign(
        hf_layer.self_attn.q_proj.weight.T.reshape(
            hidden_dim, num_heads, head_dim
        )
        .detach()
        .cpu()
        .numpy()
    )
    attn._key_dense.kernel.assign(
        hf_layer.self_attn.k_proj.weight.T.reshape(
            hidden_dim, num_heads, head_dim
        )
        .detach()
        .cpu()
        .numpy()
    )
    attn._value_dense.kernel.assign(
        hf_layer.self_attn.v_proj.weight.T.reshape(
            hidden_dim, num_heads, head_dim
        )
        .detach()
        .cpu()
        .numpy()
    )
    attn._output_dense.kernel.assign(
        hf_layer.self_attn.o_proj.weight.T.reshape(
            num_heads, head_dim, hidden_dim
        )
        .detach()
        .cpu()
        .numpy()
    )

    # Vision Gating (Only present in Global/Gated Layers)
    if is_gated and hasattr(hf_layer, "gate_attn"):
        # Note: Keras TransformerEncoder doesn't have gate_attn/gate_ffn
        # by default. This would require a custom layer.
        if hasattr(keras_layer, "gate_attn"):
            keras_layer.gate_attn.assign(
                hf_layer.gate_attn.detach().cpu().numpy()
            )
            keras_layer.gate_ffn.assign(
                hf_layer.gate_ffn.detach().cpu().numpy()
            )

    # FFN layer norm
    keras_layer._feedforward_layer_norm.gamma.assign(
        hf_layer.post_attention_layernorm.weight.detach().cpu().numpy()
    )
    keras_layer._feedforward_layer_norm.beta.assign(
        hf_layer.post_attention_layernorm.bias.detach().cpu().numpy()
    )

    # FFN (MLP)
    keras_layer._feedforward_intermediate_dense.kernel.assign(
        hf_layer.mlp.fc1.weight.T.detach().cpu().numpy()
    )
    keras_layer._feedforward_intermediate_dense.bias.assign(
        hf_layer.mlp.fc1.bias.detach().cpu().numpy()
    )
    keras_layer._feedforward_output_dense.kernel.assign(
        hf_layer.mlp.fc2.weight.T.detach().cpu().numpy()
    )
    keras_layer._feedforward_output_dense.bias.assign(
        hf_layer.mlp.fc2.bias.detach().cpu().numpy()
    )


def convert_vision_encoder_weights(keras_encoder, hf_model):
    """Convert vision encoder weights from HuggingFace to Keras."""
    hf_vision = hf_model.model.vision_model

    print("   Converting Patch & Position Embeddings...")
    # 1. Patch embedding (Conv2d)
    keras_encoder.patch_embedding.kernel.assign(
        hf_vision.patch_embedding.weight.permute(2, 3, 1, 0)
        .detach()
        .cpu()
        .numpy()
    )

    # 2. Class Embedding
    keras_encoder.class_embedding.assign(
        hf_vision.class_embedding.detach().cpu().numpy()
    )

    # 3. Gated Positional Embedding (MllamaPrecomputedPositionEmbedding)
    keras_encoder.gated_positional_embedding.embedding.assign(
        hf_vision.gated_positional_embedding.embedding.detach().cpu().numpy()
    )
    keras_encoder.gated_positional_embedding.gate.assign(
        hf_vision.gated_positional_embedding.gate.detach().cpu().numpy()
    )
    # Tile embedding inside gated pos embedding
    keras_encoder.gated_positional_embedding.tile_embedding.set_weights(
        [
            hf_vision.gated_positional_embedding.tile_embedding.weight.detach()
            .cpu()
            .numpy()
        ]
    )

    # 4. Pre/Post Tile Embeddings (MllamaPrecomputedAspectRatioEmbedding)
    keras_encoder.pre_tile_positional_embedding.embedding.set_weights(
        [
            hf_vision.pre_tile_positional_embedding.embedding.weight.detach()
            .cpu()
            .numpy()
        ]
    )
    keras_encoder.pre_tile_positional_embedding.gate.assign(
        hf_vision.pre_tile_positional_embedding.gate.detach().cpu().numpy()
    )

    keras_encoder.post_tile_positional_embedding.embedding.set_weights(
        [
            hf_vision.post_tile_positional_embedding.embedding.weight.detach()
            .cpu()
            .numpy()
        ]
    )
    keras_encoder.post_tile_positional_embedding.gate.assign(
        hf_vision.post_tile_positional_embedding.gate.detach().cpu().numpy()
    )

    print("   Converting Transformer Layers (Local)...")

    # 5. LOCAL Transformer Layers
    # HF: transformer.layers -> Keras: transformer_layers
    hf_local_layers = hf_vision.transformer.layers
    keras_local_layers = keras_encoder.transformer_layers

    if len(keras_local_layers) != len(hf_local_layers):
        print(
            f"WARNING: Local layer mismatch! Keras: {len(keras_local_layers)}, "
            f"HF: {len(hf_local_layers)}"
        )

    for i, (keras_layer, hf_layer) in enumerate(
        zip(keras_local_layers, hf_local_layers)
    ):
        _convert_vision_transformer_layer(keras_layer, hf_layer, is_gated=False)

    print("   Converting Transformer Layers (Global)...")

    # 6. GLOBAL Transformer Layers
    # HF: global_transformer.layers -> Keras: global_transformer_layers
    hf_global_layers = hf_vision.global_transformer.layers
    keras_global_layers = keras_encoder.global_transformer_layers

    if len(keras_global_layers) != len(hf_global_layers):
        print(
            f"WARNING: Global layer mismatch! "
            f"Keras: {len(keras_global_layers)}, "
            f"HF: {len(hf_global_layers)}"
        )

    for i, (keras_layer, hf_layer) in enumerate(
        zip(keras_global_layers, hf_global_layers)
    ):
        _convert_vision_transformer_layer(keras_layer, hf_layer, is_gated=True)

    # 6. Global Layer Norms (Pre/Post)
    print("   Converting Global Layer Norms...")
    keras_encoder.layernorm_pre.gamma.assign(
        hf_vision.layernorm_pre.weight.detach().cpu().numpy()
    )
    keras_encoder.layernorm_pre.beta.assign(
        hf_vision.layernorm_pre.bias.detach().cpu().numpy()
    )

    keras_encoder.layernorm_post.gamma.assign(
        hf_vision.layernorm_post.weight.detach().cpu().numpy()
    )
    keras_encoder.layernorm_post.beta.assign(
        hf_vision.layernorm_post.bias.detach().cpu().numpy()
    )


def convert_vision_projector_weights(keras_projector, hf_model):
    """Convert vision projector weights."""
    print("   Converting Vision Projector...")
    # Based on MllamaModel.__init__, line 1059:
    # self.multi_modal_projector = nn.Linear(...)
    hf_proj = hf_model.model.multi_modal_projector

    # Since it is a single Linear layer, map to Keras single Dense
    # NOTE: Ensure your Keras model uses a single Dense layer for 'projection'
    keras_projector.projection.kernel.assign(
        hf_proj.weight.T.detach().cpu().numpy()
    )
    keras_projector.projection.bias.assign(hf_proj.bias.detach().cpu().numpy())


def convert_text_backbone_weights(keras_text, hf_model):
    """Convert text backbone (Llama3) weights."""
    print("   Converting Text Backbone...")
    hf_text = hf_model.model.language_model

    # Token embedding
    keras_text.token_embedding.embeddings.assign(
        hf_text.embed_tokens.weight.detach().cpu().numpy()
    )

    # Iterate over Keras transformer layers (Standard Llama Layers)
    # We must skip the HF layers that are CrossAttentionDecoderLayers
    hf_layers = hf_text.layers

    # Identify which HF layers are standard SelfAttention layers
    hf_standard_layers = [
        layer
        for layer in hf_layers
        if "MllamaSelfAttentionDecoderLayer" in str(type(layer))
    ]

    if len(keras_text.transformer_layers) != len(hf_standard_layers):
        k_len = len(keras_text.transformer_layers)
        h_len = len(hf_standard_layers)
        print(f"WARNING: Text layer mismatch! Keras: {k_len}, HF: {h_len}")

    for i, keras_layer in enumerate(keras_text.transformer_layers):
        hf_layer = hf_standard_layers[i]

        # Get attention config
        attn = keras_layer._self_attention_layer
        num_heads = attn.num_query_heads
        num_kv_heads = attn.num_key_value_heads
        head_dim = keras_text.hidden_dim // num_heads
        hidden_dim = keras_text.hidden_dim

        # Input layer norm
        keras_layer._self_attention_layernorm.scale.assign(
            hf_layer.input_layernorm.weight.detach().cpu().numpy()
        )

        # Self-attention
        attn._query_dense.kernel.assign(
            hf_layer.self_attn.q_proj.weight.T.reshape(
                hidden_dim, num_heads, head_dim
            )
            .detach()
            .cpu()
            .numpy()
        )
        attn._key_dense.kernel.assign(
            hf_layer.self_attn.k_proj.weight.T.reshape(
                hidden_dim, num_kv_heads, head_dim
            )
            .detach()
            .cpu()
            .numpy()
        )
        attn._value_dense.kernel.assign(
            hf_layer.self_attn.v_proj.weight.T.reshape(
                hidden_dim, num_kv_heads, head_dim
            )
            .detach()
            .cpu()
            .numpy()
        )
        attn._output_dense.kernel.assign(
            hf_layer.self_attn.o_proj.weight.T.reshape(
                num_heads, head_dim, hidden_dim
            )
            .detach()
            .cpu()
            .numpy()
        )

        # Post-attention layer norm
        keras_layer._feedforward_layernorm.scale.assign(
            hf_layer.post_attention_layernorm.weight.detach().cpu().numpy()
        )

        # MLP
        keras_layer._feedforward_gate_dense.kernel.assign(
            hf_layer.mlp.gate_proj.weight.T.detach().cpu().numpy()
        )
        keras_layer._feedforward_intermediate_dense.kernel.assign(
            hf_layer.mlp.up_proj.weight.T.detach().cpu().numpy()
        )
        keras_layer._feedforward_output_dense.kernel.assign(
            hf_layer.mlp.down_proj.weight.T.detach().cpu().numpy()
        )

    # Final layer norm
    keras_text.layer_norm.scale.assign(
        hf_text.norm.weight.detach().cpu().numpy()
    )


def convert_cross_attention_weights(keras_ca_blocks, hf_model):
    """Convert cross-attention layer weights."""
    print("   Converting Cross Attention Blocks...")
    hf_layers = hf_model.model.language_model.layers

    # keras_ca_blocks is a dict {layer_idx: layer_obj}
    for layer_idx, keras_ca in keras_ca_blocks.items():
        # Access the HF layer by index (MllamaCrossAttentionDecoderLayer)
        hf_layer = hf_layers[layer_idx]
        hf_ca = hf_layer.cross_attn

        # Norms
        keras_ca.query_norm.scale.assign(
            hf_ca.q_norm.weight.detach().cpu().numpy()
        )
        keras_ca.kv_norm.scale.assign(
            hf_ca.k_norm.weight.detach().cpu().numpy()
        )

        # Projections
        keras_ca.query_dense.kernel.assign(
            hf_ca.q_proj.weight.T.detach().cpu().numpy()
        )
        keras_ca.key_dense.kernel.assign(
            hf_ca.k_proj.weight.T.detach().cpu().numpy()
        )
        keras_ca.value_dense.kernel.assign(
            hf_ca.v_proj.weight.T.detach().cpu().numpy()
        )
        keras_ca.output_dense.kernel.assign(
            hf_ca.o_proj.weight.T.detach().cpu().numpy()
        )

        # GATES - Crucial! HF has TWO gates.
        # 1. Attention Gate
        keras_ca.gate.assign(
            hf_layer.cross_attn_attn_gate.detach().cpu().numpy()
        )

        # 2. MLP Gate (Make sure your Keras layer has this!)
        if hasattr(keras_ca, "mlp_gate"):
            keras_ca.mlp_gate.assign(
                hf_layer.cross_attn_mlp_gate.detach().cpu().numpy()
            )
        else:
            print("WARNING: Keras CrossAttention missing 'mlp_gate'.")


def convert_checkpoints(keras_model, hf_model):
    """Convert all weights from HuggingFace to Keras."""
    print("\n-> Converting vision encoder weights...")
    convert_vision_encoder_weights(
        keras_model.backbone.vision_encoder, hf_model
    )

    print("-> Converting vision projector weights...")
    convert_vision_projector_weights(
        keras_model.backbone.vision_projector, hf_model
    )

    print("-> Converting text backbone weights...")
    convert_text_backbone_weights(keras_model.backbone.text_backbone, hf_model)

    print("-> Converting cross-attention weights...")
    convert_cross_attention_weights(
        keras_model.backbone.cross_attention_blocks, hf_model
    )


def test_model(keras_model, hf_model, processor):
    """Test that the outputs of both models match."""
    print("\n-> Testing model outputs...")

    # Create test input
    np.random.seed(42)
    test_image = np.random.randint(0, 255, (560, 560, 3), dtype=np.uint8)
    from PIL import Image

    pil_image = Image.fromarray(test_image)

    text_input = "<|image|>If I had to write a haiku for this one"
    hf_inputs = processor(
        images=pil_image,
        text=text_input,
        return_tensors="pt",
    )

    # HuggingFace forward pass
    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=hf_inputs["input_ids"],
            pixel_values=hf_inputs["pixel_values"],
            aspect_ratio_ids=hf_inputs["aspect_ratio_ids"],
            aspect_ratio_mask=hf_inputs["aspect_ratio_mask"],
            attention_mask=hf_inputs.get("attention_mask"),
            output_hidden_states=True,
        )
    hf_hidden = hf_outputs.logits.detach().cpu().numpy()

    # Keras forward pass
    # Ensure inputs match Keras model signature
    keras_inputs = {
        "images": test_image.astype(np.float32)[np.newaxis] / 255.0,
        "token_ids": hf_inputs["input_ids"].numpy(),
        "padding_mask": hf_inputs.get(
            "attention_mask", torch.ones_like(hf_inputs["input_ids"])
        ).numpy(),
    }

    keras_logits = ops.convert_to_numpy(keras_model(keras_inputs))

    # Compare Logits
    # We compare the last few tokens which are generated
    print(f"   HF Logits Shape: {hf_hidden.shape}")
    print(f"   Keras Logits Shape: {keras_logits.shape}")

    # Just check the max difference
    diff = np.abs(keras_logits - hf_hidden).max()
    print(f"   Max Absolute Difference: {diff}")

    if diff < 1e-4:
        print("   SUCCESS: Outputs match within tolerance!")
    else:
        print("   FAILURE: Outputs do not match.")


def main(_):
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset. Must be one of {list(PRESET_MAP.keys())}"
        )

    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    print("=" * 60)
    print(f"Converting: {preset} <- {hf_preset}")
    print("=" * 60)

    # Load HF Model
    processor = AutoProcessor.from_pretrained(hf_preset)
    hf_model = MllamaForConditionalGeneration.from_pretrained(
        hf_preset, torch_dtype=torch.float32, device_map="cpu"
    )
    hf_model.eval()
    hf_config = hf_model.config.to_dict()

    # Create Keras Model
    # Note: We usually instantiate the CausalLM, which contains the Backbone
    print("-> Creating Keras model...")
    keras_config = convert_backbone_config(hf_config)

    # Debug: Print vision_output_dim to verify
    print(f"   DEBUG: vision_output_dim = {keras_config['vision_output_dim']}")

    # Instantiate the backbone first
    backbone = Llama3VisionBackbone(**keras_config)

    # Wrap in CausalLM (this adds the LM Head)
    keras_model = Llama3VisionCausalLM(backbone=backbone, preprocessor=None)

    # Convert weights
    convert_checkpoints(keras_model, hf_model)

    # Load LM Head (Part of CausalLM, not backbone)
    print("-> Converting LM Head...")
    keras_model.output_dense.kernel.assign(
        hf_model.lm_head.weight.T.detach().cpu().numpy()
    )

    # Test
    test_model(keras_model, hf_model, processor)

    # Save
    print(f"-> Saving to {preset}.keras")
    keras_model.save(f"{preset}.keras")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
