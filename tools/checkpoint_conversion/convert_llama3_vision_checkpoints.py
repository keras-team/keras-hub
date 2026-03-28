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

    # HF MllamaTextModel uses vocab_size directly (128256)
    vocab_size = text_config.get("vocab_size", 128256)

    # Keras text backbone has ONLY self-attention layers
    # HF CrossAttentionDecoderLayer has NO self_attn, only cross_attn + mlp
    # So we need num_text_layers - num_cross_attention_layers for Keras
    num_self_attention_layers = num_text_layers - len(cross_attention_layers)

    return {
        "vocabulary_size": vocab_size,
        "num_layers": num_self_attention_layers,  # Only self-attention layers
        "hidden_dim": text_config.get("hidden_size", 4096),
        "num_query_heads": text_config.get("num_attention_heads", 32),
        "num_key_value_heads": text_config.get("num_key_value_heads", 8),
        "intermediate_dim": text_config.get("intermediate_size", 14336),
        "rope_max_wavelength": text_config.get("rope_theta", 500000),
        "layer_norm_epsilon": text_config.get("rms_norm_eps", 1e-5),
        "vision_layer_norm_epsilon": vision_config.get("layer_norm_eps", 1e-6),
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
    # === 1. Fix Normalization Names ===
    # Use 'norm1' and 'norm2' to match Keras Llama3VisionTransformerLayer
    keras_layer.norm1.scale.assign(
        hf_layer.input_layernorm.weight.detach().cpu().numpy()
    )

    # Use direct 'attn' access matching the Keras layer structure
    # No reshaping needed as dimensions align after transpose

    # Query
    keras_layer.attn._query_dense.kernel.assign(
        hf_layer.self_attn.q_proj.weight.T.detach().cpu().numpy()
    )
    # Key
    keras_layer.attn._key_dense.kernel.assign(
        hf_layer.self_attn.k_proj.weight.T.detach().cpu().numpy()
    )
    # Value
    keras_layer.attn._value_dense.kernel.assign(
        hf_layer.self_attn.v_proj.weight.T.detach().cpu().numpy()
    )
    # Output
    keras_layer.attn._output_dense.kernel.assign(
        hf_layer.self_attn.o_proj.weight.T.detach().cpu().numpy()
    )

    # Gating for Global Layers
    if is_gated:
        keras_layer.gate_attn.assign(hf_layer.gate_attn.detach().cpu().numpy())
        keras_layer.gate_ffn.assign(hf_layer.gate_ffn.detach().cpu().numpy())

    # === 2. Fix MLP Names ===
    # Use 'norm2' for the post-attention norm
    keras_layer.norm2.scale.assign(
        hf_layer.post_attention_layernorm.weight.detach().cpu().numpy()
    )

    # MLP (SwiGLU)
    # Use direct attribute names 'gate_proj', 'up_proj', 'down_proj'
    keras_layer.gate_proj.kernel.assign(
        hf_layer.mlp.gate_proj.weight.T.detach().cpu().numpy()
    )
    keras_layer.up_proj.kernel.assign(
        hf_layer.mlp.up_proj.weight.T.detach().cpu().numpy()
    )
    keras_layer.down_proj.kernel.assign(
        hf_layer.mlp.down_proj.weight.T.detach().cpu().numpy()
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
    keras_encoder.layernorm_pre.scale.assign(
        hf_vision.layernorm_pre.weight.detach().cpu().numpy()
    )

    keras_encoder.layernorm_post.scale.assign(
        hf_vision.layernorm_post.weight.detach().cpu().numpy()
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
    # Ensure we only copy up to the Keras vocab size (128256)
    # HF might have 128264 (8 extra special tokens)
    vocab_size = keras_text.token_embedding.input_dim
    keras_text.token_embedding.embeddings.assign(
        hf_text.embed_tokens.weight[:vocab_size].detach().cpu().numpy()
    )

    # Iterate over Keras transformer layers (Standard Llama Layers)
    # HF CrossAttentionDecoderLayer has NO self_attn, only cross_attn + mlp
    # So we must only copy from MllamaSelfAttentionDecoderLayer
    hf_layers = list(hf_text.layers)
    hf_self_attn_layers = [
        layer
        for layer in hf_layers
        if "MllamaSelfAttentionDecoderLayer" in str(type(layer))
    ]

    if len(keras_text.transformer_layers) != len(hf_self_attn_layers):
        k_len = len(keras_text.transformer_layers)
        h_len = len(hf_self_attn_layers)
        print(
            f"WARNING: Text layer mismatch! "
            f"Keras: {k_len}, HF SelfAttn: {h_len}"
        )

    for i, keras_layer in enumerate(keras_text.transformer_layers):
        hf_layer = hf_self_attn_layers[i]

        # Get attention config
        attn = keras_layer._self_attention_layer

        # Input layer norm
        keras_layer._self_attention_layernorm.scale.assign(
            hf_layer.input_layernorm.weight.detach().cpu().numpy()
        )

        # Self-attention
        attn._query_dense.kernel.assign(
            hf_layer.self_attn.q_proj.weight.T.detach().cpu().numpy()
        )
        attn._key_dense.kernel.assign(
            hf_layer.self_attn.k_proj.weight.T.detach().cpu().numpy()
        )
        attn._value_dense.kernel.assign(
            hf_layer.self_attn.v_proj.weight.T.detach().cpu().numpy()
        )
        attn._output_dense.kernel.assign(
            hf_layer.self_attn.o_proj.weight.T.detach().cpu().numpy()
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

        # Input layer norm (applied BEFORE cross-attention in HF)
        keras_ca.input_layernorm.scale.assign(
            hf_layer.input_layernorm.weight.detach().cpu().numpy()
        )

        # Per-head norms for Q/K
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

        # 2. MLP Gate
        keras_ca.mlp_gate.assign(
            hf_layer.cross_attn_mlp_gate.detach().cpu().numpy()
        )

        # === MLP weights (HF CrossAttentionDecoderLayer has mlp) ===
        hf_mlp = hf_layer.mlp

        # Post-attention layer norm
        keras_ca.post_attention_layernorm.scale.assign(
            hf_layer.post_attention_layernorm.weight.detach().cpu().numpy()
        )

        # MLP projections (SwiGLU)
        keras_ca.mlp_gate_proj.kernel.assign(
            hf_mlp.gate_proj.weight.T.detach().cpu().numpy()
        )
        keras_ca.mlp_up_proj.kernel.assign(
            hf_mlp.up_proj.weight.T.detach().cpu().numpy()
        )
        keras_ca.mlp_down_proj.kernel.assign(
            hf_mlp.down_proj.weight.T.detach().cpu().numpy()
        )


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


def compare_tensors(hf_tensor, keras_tensor, name):
    """Compares HF and Keras tensors."""
    if isinstance(hf_tensor, torch.Tensor):
        hf_val = hf_tensor.detach().cpu().numpy()
    else:
        hf_val = hf_tensor

    if hasattr(keras_tensor, "numpy"):
        keras_val = keras_tensor.numpy()
    else:
        keras_val = keras_tensor

    diff = 0.0
    # Try to shape match
    if hf_val.shape == keras_val.shape:
        diff = np.abs(hf_val - keras_val).max()
    elif hf_val.size == keras_val.size:
        # Try flattening
        diff = np.abs(hf_val.flatten() - keras_val.flatten()).max()
    else:
        print(
            f"[{name}] SHAPE MISMATCH: "
            f"HF {hf_val.shape} vs Keras {keras_val.shape}"
        )
        return

    print(f"[{name}] Max Diff: {diff:.8f}")
    if diff > 1e-4:
        print("   !!! HIGH DIVERGENCE !!!")


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
    # Cast pixel values to float16 for memory efficiency
    hf_inputs["pixel_values"] = hf_inputs["pixel_values"].to(torch.float16)

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
    # HF pixel_values shape: (batch, num_images, num_tiles, channels, H, W)
    # Keras expects: (batch, num_tiles, H, W, channels)
    hf_pixel_values = hf_inputs["pixel_values"].numpy()
    print(f"   HF pixel_values shape: {hf_pixel_values.shape}")

    # Extract first image (all tiles) and transpose to Keras format
    # HF: (batch, num_images, num_tiles, C, H, W) -> (batch, num_tiles, C, H, W)
    pixel_values_keras = hf_pixel_values[:, 0]  # (batch, num_tiles, C, H, W)
    # Transpose to (batch, num_tiles, H, W, C)
    pixel_values_keras = np.transpose(pixel_values_keras, (0, 1, 3, 4, 2))
    print(f"   Keras pixel_values shape: {pixel_values_keras.shape}")

    # Aspect ratio IDs
    # HF: (batch, num_images, num_tiles) -> (batch, num_tiles)
    aspect_ratio_ids = hf_inputs["aspect_ratio_ids"].numpy().squeeze(1)
    aspect_ratio_mask = hf_inputs["aspect_ratio_mask"].numpy().squeeze(1)

    # Ensure 2D (batch, num_tiles) even if batch=1 and squeeze made it 1D
    if aspect_ratio_ids.ndim == 1:
        aspect_ratio_ids = aspect_ratio_ids[np.newaxis, :]
    if aspect_ratio_mask.ndim == 1:
        aspect_ratio_mask = aspect_ratio_mask[np.newaxis, :]

    # Pad aspect_ratio_ids to match pixel_values num_tiles (max_num_tiles)
    # HF might return IDs only for used tiles, but Keras input expects uniform
    # shape matching pixel_values
    num_tiles_keras = pixel_values_keras.shape[1]  # (B, T, H, W, C)
    current_tiles = aspect_ratio_ids.shape[1]
    if current_tiles < num_tiles_keras:
        pad_len = num_tiles_keras - current_tiles
        # Pad with 1 (default square aspect ratio ID)
        pad_ids = np.ones(
            (aspect_ratio_ids.shape[0], pad_len), dtype=aspect_ratio_ids.dtype
        )
        aspect_ratio_ids = np.concatenate([aspect_ratio_ids, pad_ids], axis=1)

    keras_inputs = {
        "pixel_values": pixel_values_keras,
        "token_ids": np.clip(hf_inputs["input_ids"].numpy(), 0, 128255),
        "padding_mask": hf_inputs.get(
            "attention_mask", torch.ones_like(hf_inputs["input_ids"])
        ).numpy(),
        "aspect_ratio_ids": aspect_ratio_ids,
        "aspect_ratio_mask": aspect_ratio_mask,
    }

    keras_logits = ops.convert_to_numpy(keras_model(keras_inputs))

    # Compare Logits
    # We compare the last few tokens which are generated

    print(f"   HF Logits Shape: {hf_hidden.shape}")
    print(f"   Keras Logits Shape: {keras_logits.shape}")

    # Compare only common vocab indices (HF vocab size is base, Keras has +8)
    hf_vocab_size = hf_hidden.shape[-1]
    keras_logits_common = keras_logits[:, :, :hf_vocab_size]

    # Just check the max difference
    diff = np.abs(keras_logits_common - hf_hidden).max()
    print(f"   Max Absolute Difference: {diff}")

    if diff < 1e-5:
        print("   SUCCESS: Outputs match within tolerance!")
    else:
        print("   WARNING: Outputs differ. Check conversion quality.")


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
        hf_preset, torch_dtype=torch.float16, device_map="cpu"
    )
    hf_model.eval()
    hf_config = hf_model.config.to_dict()

    # Create Keras Model
    # Note: We usually instantiate the CausalLM, which contains the Backbone
    print("-> Creating Keras model...")
    keras_config = convert_backbone_config(hf_config)

    # Instantiate the backbone first
    # Use float16 to match HF and save memory
    keras_config["dtype"] = "float16"
    backbone = Llama3VisionBackbone(**keras_config)

    # Wrap in CausalLM (this adds the LM Head)
    keras_model = Llama3VisionCausalLM(backbone=backbone, preprocessor=None)

    # Convert weights
    convert_checkpoints(keras_model, hf_model)

    # LM Head: Weights are tied to token_embedding (ReversibleEmbedding)
    # No separate conversion needed - already set via token_embedding
    print("-> LM Head uses tied weights (no separate conversion needed)")

    # Test
    test_model(keras_model, hf_model, processor)

    # Save using KerasHub preset format
    print(f"-> Saving model preset to {preset}/")
    keras_model.backbone.save_to_preset(preset)
    print("-> Saved the model preset")

    # Save preprocessor if available
    if keras_model.preprocessor is not None:
        keras_model.preprocessor.save_to_preset(preset)
        print("-> Saved the preprocessor")
    else:
        print("-> No preprocessor to save (was set to None)")

    print(f"\n=== Conversion complete! Preset saved to: {preset}/ ===")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
