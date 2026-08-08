import argparse
import os

import numpy as np
import torch
from transformers import (
    AutoModel,  # Assuming AutoModel works, adjust if specific class is needed
)

# Set backend to torch for easy conversion if needed, or stick to default
os.environ["KERAS_BACKEND"] = "torch"

from keras_hub.models import Qwen3ASRBackbone
from keras_hub.models import Qwen3ASRCausalLM


def _to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def map_text_backbone(keras_backbone, hf_model):
    """Maps standard Qwen3 text backbone weights."""
    print("Mapping text backbone...")

    # Embedding
    keras_backbone.token_embedding.embeddings.assign(
        _to_np(hf_model.model.embed_tokens.weight)
    )

    if not keras_backbone.tie_word_embeddings:
        keras_backbone.token_embedding.reverse_embeddings.assign(
            np.transpose(_to_np(hf_model.lm_head.weight))
        )

    # Transformer Layers
    for i in range(keras_backbone.num_layers):
        print(f"Mapping transformer layer {i}...")
        kh_layer = keras_backbone.transformer_layers[i]
        hf_layer = hf_model.model.layers[i]

        # Attention
        kh_layer._self_attention_layernorm.scale.assign(
            _to_np(hf_layer.input_layernorm.weight)
        )

        # QKV Attention
        kh_layer._self_attention_layer._query_dense.kernel.assign(
            np.transpose(_to_np(hf_layer.self_attn.q_proj.weight))
        )
        if hasattr(hf_layer.self_attn, "q_norm"):
            kh_layer._self_attention_layer._query_dense_layer_norm.scale.assign(
                _to_np(hf_layer.self_attn.q_norm.weight)
            )

        kh_layer._self_attention_layer._key_dense.kernel.assign(
            np.transpose(_to_np(hf_layer.self_attn.k_proj.weight))
        )
        if hasattr(hf_layer.self_attn, "k_norm"):
            kh_layer._self_attention_layer._key_dense_layer_norm.scale.assign(
                _to_np(hf_layer.self_attn.k_norm.weight)
            )

        kh_layer._self_attention_layer._value_dense.kernel.assign(
            np.transpose(_to_np(hf_layer.self_attn.v_proj.weight))
        )

        kh_layer._self_attention_layer._output_dense.kernel.assign(
            np.transpose(_to_np(hf_layer.self_attn.o_proj.weight))
        )

        # MLP
        kh_layer._feedforward_intermediate_dense.kernel.assign(
            np.transpose(_to_np(hf_layer.mlp.up_proj.weight))
        )
        kh_layer._feedforward_output_dense.kernel.assign(
            np.transpose(_to_np(hf_layer.mlp.down_proj.weight))
        )
        kh_layer._feedforward_gate_dense.kernel.assign(
            np.transpose(_to_np(hf_layer.mlp.gate_proj.weight))
        )

        kh_layer._feedforward_layernorm.scale.assign(
            _to_np(hf_layer.post_attention_layernorm.weight)
        )

    # Final Norm
    keras_backbone.layer_norm.scale.assign(_to_np(hf_model.model.norm.weight))


def map_audio_encoder(keras_audio_encoder, hf_audio_encoder):
    """Maps Qwen3-ASR Audio Encoder weights."""
    print("Mapping audio encoder...")

    # Note: This mapping is based on standard naming conventions for audio
    # encoders in HF.
    # If the HF model uses different naming (e.g., audio_tower), adjust keys
    # accordingly.

    # Convolutional Layers
    try:
        keras_audio_encoder.conv2d1.kernel.assign(
            np.transpose(
                _to_np(hf_audio_encoder.conv1.weight), (2, 3, 1, 0)
            )  # Assuming Conv2D format
        )
        keras_audio_encoder.conv2d1.bias.assign(
            _to_np(hf_audio_encoder.conv1.bias)
        )

        keras_audio_encoder.conv2d2.kernel.assign(
            np.transpose(_to_np(hf_audio_encoder.conv2.weight), (2, 3, 1, 0))
        )
        keras_audio_encoder.conv2d2.bias.assign(
            _to_np(hf_audio_encoder.conv2.bias)
        )

        keras_audio_encoder.conv2d3.kernel.assign(
            np.transpose(_to_np(hf_audio_encoder.conv3.weight), (2, 3, 1, 0))
        )
        keras_audio_encoder.conv2d3.bias.assign(
            _to_np(hf_audio_encoder.conv3.bias)
        )
    except AttributeError as e:
        print(f"Warning: Audio Convolutional layers might differ: {e}")
        print("Please check `hf_audio_encoder` keys and map manually.")

    # Output Projection
    if hasattr(keras_audio_encoder, "conv_out") and hasattr(
        hf_audio_encoder, "conv_out"
    ):
        keras_audio_encoder.conv_out.kernel.assign(
            np.transpose(_to_np(hf_audio_encoder.conv_out.weight))
        )
        if hasattr(keras_audio_encoder.conv_out, "bias") and hasattr(
            hf_audio_encoder.conv_out, "bias"
        ):
            keras_audio_encoder.conv_out.bias.assign(
                _to_np(hf_audio_encoder.conv_out.bias)
            )

    # Transformer Encoder Layers
    # Qwen3ASRAudioEncoder transformer layers are listed in
    # self.transformer_layers
    if hasattr(keras_audio_encoder, "transformer_layers") and hasattr(
        hf_audio_encoder, "layers"
    ):
        for i in range(len(keras_audio_encoder.transformer_layers)):
            print(f"Mapping audio transformer layer {i}...")
            kh_enc_layer = keras_audio_encoder.transformer_layers[i]
            hf_enc_layer = hf_audio_encoder.layers[i]  # Adjust if different

            # Self Attention
            kh_enc_layer._self_attention_layernorm.scale.assign(
                _to_np(hf_enc_layer.input_layernorm.weight)
            )

            kh_enc_layer._self_attention_layer._query_dense.kernel.assign(
                np.transpose(_to_np(hf_enc_layer.self_attn.q_proj.weight))
            )
            kh_enc_layer._self_attention_layer._key_dense.kernel.assign(
                np.transpose(_to_np(hf_enc_layer.self_attn.k_proj.weight))
            )
            kh_enc_layer._self_attention_layer._value_dense.kernel.assign(
                np.transpose(_to_np(hf_enc_layer.self_attn.v_proj.weight))
            )
            kh_enc_layer._self_attention_layer._output_dense.kernel.assign(
                np.transpose(_to_np(hf_enc_layer.self_attn.o_proj.weight))
            )

            # Feedforward
            kh_enc_layer._feedforward_layernorm.scale.assign(
                _to_np(hf_enc_layer.post_attention_layernorm.weight)
            )
            kh_enc_layer._feedforward_intermediate_dense.kernel.assign(
                np.transpose(_to_np(hf_enc_layer.mlp.up_proj.weight))
            )
            kh_enc_layer._feedforward_output_dense.kernel.assign(
                np.transpose(_to_np(hf_enc_layer.mlp.down_proj.weight))
            )
            kh_enc_layer._feedforward_gate_dense.kernel.assign(
                np.transpose(_to_np(hf_enc_layer.mlp.gate_proj.weight))
            )

    # Final Norm
    if hasattr(keras_audio_encoder, "ln_post") and hasattr(
        hf_audio_encoder, "ln_post"
    ):
        keras_audio_encoder.ln_post.scale.assign(
            _to_np(hf_audio_encoder.ln_post.weight)
        )


def convert_checkpoints(keras_model, hf_model):
    """Main conversion function."""
    map_text_backbone(keras_model.backbone, hf_model)

    if hasattr(hf_model, "audio_encoder") or hasattr(hf_model, "audio_tower"):
        hf_audio = getattr(
            hf_model, "audio_encoder", getattr(hf_model, "audio_tower", None)
        )
        if hf_audio is not None and hasattr(
            keras_model.backbone, "audio_encoder"
        ):
            map_audio_encoder(keras_model.backbone.audio_encoder, hf_audio)
    else:
        print(
            "Warning: No audio encoder found in HF model or "
            "Keras model backbone."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-ASR checkpoints."
    )
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to local HF checkpoint or HuggingFace handle.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the converted KerasHub preset.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="qwen3_asr_1.7b",
        help="Preset name to use (e.g., qwen3_asr_1.7b, qwen3_asr_0.6b).",
    )

    args = parser.parse_args()

    print(f"Loading HF model from {args.hf_model_path}...")
    # Adjust trust_remote_code if needed
    hf_model = AutoModel.from_pretrained(
        args.hf_model_path, trust_remote_code=True
    )
    print(
        f"Initializing Keras model architecture from preset: {args.preset}..."
    )
    try:
        # Load architecture without weights
        backbone = Qwen3ASRBackbone.from_preset(args.preset, load_weights=False)
        keras_model = Qwen3ASRCausalLM(backbone=backbone)
    except Exception as e:
        print(f"Failed to initialize model from preset: {e}")
        print("Please ensure the preset is available in qwen3_asr_presets.py")
        return

    print("Mapping weights...")
    convert_checkpoints(keras_model, hf_model)

    print(f"Saving converted preset to {args.save_path}...")
    keras_model.save_to_preset(args.save_path)
    print("Conversion complete!")


if __name__ == "__main__":
    main()
