import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM
from transformers import AutoProcessor

# Set backend to torch for easy comparison
os.environ["KERAS_BACKEND"] = "torch"

from keras_hub.models import Qwen3ASRCausalLM

# Seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    if isinstance(tensor, (list, tuple)):
        return np.array(tensor)
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def verify_parity(keras_model, hf_model, hf_processor):
    """Verifies numerical parity between Keras and HF models."""
    print("\n--- Starting Numerical Parity Check ---")

    # Force CPU for verification to avoid MPS precision issues
    device = torch.device("cpu")
    hf_model = hf_model.to(device)

    # Check unsplittable tokens
    tokenizer = keras_model.preprocessor.tokenizer
    print(f"Unsplittable tokens count: {len(tokenizer.unsplittable_tokens)}")

    # 1. Create Dummy Input (Text Only first)
    print("\n--- Text Only Parity Check ---")
    dummy_prompts = "This is a test of numerical parity."
    keras_tokens = tokenizer(dummy_prompts)
    keras_tokens_np = np.expand_dims(keras_tokens, 0)

    with torch.no_grad():
        keras_tokens_pt = torch.from_numpy(keras_tokens_np).long().to(device)
        keras_emb = keras_model.backbone.token_embedding(keras_tokens_pt)
        keras_x = keras_emb
        for layer in keras_model.backbone.transformer_layers:
            keras_x = layer(keras_x)
        keras_x = keras_model.backbone.layer_norm(keras_x)
        keras_logits = (
            to_numpy(keras_x)
            @ to_numpy(keras_model.backbone.token_embedding.embeddings).T
        )

        hf_tokens = torch.from_numpy(keras_tokens_np).long().to(device)
        hf_full_outputs = hf_model(input_ids=hf_tokens)
        hf_full_logits = to_numpy(hf_full_outputs.logits)

    print(
        f"Max Text-Only Logits Difference: {np.max(np.abs(keras_logits - hf_full_logits)):.6e}"
    )

    # 2. Check RoPE parity
    print("\n--- RoPE Parity Check (Layer 0) ---")
    with torch.no_grad():
        # Get hidden states after embedding for both
        k_h = keras_model.backbone.token_embedding(keras_tokens_pt)
        h_h = hf_model.model.language_model.embed_tokens(hf_tokens)

        # Apply first layer LN
        k_h_ln = keras_model.backbone.transformer_layers[
            0
        ]._self_attention_layernorm(k_h)
        h_h_ln = hf_model.model.language_model.layers[0].input_layernorm(h_h)

        # Compute Q and K
        k_q = keras_model.backbone.transformer_layers[
            0
        ]._self_attention_layer._query_dense(k_h_ln)
        k_q = keras_model.backbone.transformer_layers[
            0
        ]._self_attention_layer._query_dense_layer_norm(k_q)

        h_q = hf_model.model.language_model.layers[0].self_attn.q_proj(h_h_ln)
        # HF Qwen3Attention might do head-wise norm manually or use a layer
        h_q = h_q.view(
            1,
            -1,
            hf_model.config.text_config.num_attention_heads,
            hf_model.config.text_config.head_dim,
        )
        h_q = hf_model.model.language_model.layers[0].self_attn.q_norm(h_q)

        q_diff_pre_rope = np.max(np.abs(to_numpy(k_q) - to_numpy(h_q)))
        print(f"Max Q Difference (Pre-RoPE): {q_diff_pre_rope:.6e}")

        # Apply RoPE
        k_q_rope = keras_model.backbone.transformer_layers[
            0
        ]._self_attention_layer.rotary_embedding_layer(k_q)

        # HF RoPE
        position_ids = torch.arange(h_q.shape[1]).unsqueeze(0).to(device)
        cos, sin = hf_model.model.language_model.rotary_emb(h_h, position_ids)

        # Manually apply RoPE in HF style for comparison
        def apply_rotary_pos_emb(q, cos, sin, position_ids, unsqueeze_dim=2):
            # q: [B, L, N, H]
            # cos/sin are (1, L, H)
            cos = cos.unsqueeze(unsqueeze_dim)  # [1, L, 1, H]
            sin = sin.unsqueeze(unsqueeze_dim)  # [1, L, 1, H]
            q_embed = (q * cos) + (rotate_half(q) * sin)
            return q_embed

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        h_q_rope = apply_rotary_pos_emb(h_q, cos, sin, position_ids)

        q_diff_post_rope = np.max(
            np.abs(to_numpy(k_q_rope) - to_numpy(h_q_rope))
        )
        print(f"Max Q Difference (Post-RoPE): {q_diff_post_rope:.6e}")

    # 3. Multimodal Parity Check
    print("\n--- Multimodal Parity Check ---")
    sample_rate = 16000
    duration = 1
    total_samples = sample_rate * duration
    dummy_audio = np.random.uniform(-1.0, 1.0, size=(1, total_samples)).astype(
        np.float32
    )
    dummy_prompts = "<audio>transcribe"
    preprocessed_x = keras_model.preprocessor.generate_preprocess(
        {"audio": dummy_audio, "prompts": dummy_prompts}
    )

    keras_inputs = {}
    for k, v in preprocessed_x.items():
        v_np = to_numpy(v)
        if len(v_np.shape) == (1 if k != "audio_mel" else 2):
            v_np = np.expand_dims(v_np, 0)
        if k in ["token_ids", "padding_mask", "audio_mask"]:
            keras_inputs[k] = torch.from_numpy(v_np).long().to(device)
        else:
            keras_inputs[k] = torch.from_numpy(v_np).float().to(device)

    keras_len = int(np.sum(to_numpy(preprocessed_x["padding_mask"])))
    hf_inputs = {
        "input_ids": keras_inputs["token_ids"][:, :keras_len],
        "attention_mask": keras_inputs["padding_mask"][:, :keras_len],
        "input_features": keras_inputs["audio_mel"].transpose(1, 2),
        "input_features_mask": keras_inputs["audio_mask"],
    }

    with torch.no_grad():
        keras_logits = keras_model(keras_inputs)
        keras_logits_np = to_numpy(keras_logits)
        hf_outputs = hf_model(**hf_inputs)
        hf_logits = to_numpy(hf_outputs.logits)

    k_log = keras_logits_np[:, :keras_len, :]
    max_diff = np.max(np.abs(k_log - hf_logits))
    print(f"Max Absolute Difference: {max_diff:.6e}")

    if max_diff < 1e-3:
        print("✅ Numerical Parity Achieved!")
    else:
        print("❌ Numerical Parity Failed!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert and verify Qwen3-ASR 0.6B."
    )
    parser.add_argument("--hf_model_path", type=str, required=True)
    parser.add_argument("--keras_model_path", type=str, default=None)
    args = parser.parse_args()

    hf_model = AutoModelForMultimodalLM.from_pretrained(
        args.hf_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    hf_model.eval()
    hf_processor = AutoProcessor.from_pretrained(args.hf_model_path)

    keras_model_path = args.keras_model_path or f"hf://{args.hf_model_path}"
    keras_model = Qwen3ASRCausalLM.from_preset(
        keras_model_path, dtype="float32"
    )

    keras_model.preprocessor.sequence_length = 256
    if (
        hasattr(keras_model.preprocessor, "audio_converter")
        and keras_model.preprocessor.audio_converter
    ):
        keras_model.preprocessor.audio_converter.max_audio_length = 1.0
        keras_model.preprocessor.audio_converter.num_samples = 16000

    verify_parity(keras_model, hf_model, hf_processor)


if __name__ == "__main__":
    main()
