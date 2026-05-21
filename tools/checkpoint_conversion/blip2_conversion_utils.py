"""Shared utilities for BLIP-2 checkpoint conversion scripts.

Functions for EVA-CLIP vision encoder, Q-Former, and projection layer weight
transfer and validation are identical across all BLIP-2 variants (OPT, Flan-T5)
and live here to avoid duplication.
"""

import random

import numpy as np
import torch
from PIL import Image

import keras


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    keras.utils.set_random_seed(seed)


def to_np(tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    try:
        return tensor.detach().cpu().numpy()
    except AttributeError:
        return np.array(tensor)


def print_header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def report_diff(
    label: str,
    keras_out: np.ndarray,
    pt_out: np.ndarray,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    abs_diff = np.abs(keras_out - pt_out)
    mean_diff = float(np.mean(abs_diff))
    max_diff = float(np.max(abs_diff))
    print(f"\n🔶 {label}")
    print(f"   -> mean absolute diff : {mean_diff:.8f}")
    print(f"   -> max  absolute diff : {max_diff:.8f}")
    try:
        np.testing.assert_allclose(keras_out, pt_out, atol=atol, rtol=rtol)
        print(f"   -> ✅ within atol={atol}, rtol={rtol}")
        return True
    except AssertionError as err:
        print(f"   -> ⚠️  tolerance check failed: {err.args[0][:120]}")
        return False


# ── Validation ────────────────────────────────────────────────────────────────

def validate_vision_encoder(keras_vision, hf_model) -> bool:
    print_header("VISION ENCODER VALIDATION")

    image_pil = Image.new("RGB", (224, 224), color=(114, 114, 114))
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    image_np = image_np[None]

    image_pt = torch.tensor(image_np).permute(0, 3, 1, 2)
    with torch.no_grad():
        pt_out = to_np(
            hf_model.vision_model(pixel_values=image_pt).last_hidden_state
        )

    keras_out = to_np(keras_vision(image_np))

    print(f"   -> HF    shape : {pt_out.shape}")
    print(f"   -> Keras shape : {keras_out.shape}")

    return report_diff("Vision encoder hidden states", keras_out, pt_out)


def validate_qformer(
    keras_qformer, hf_model, vision_features_np: np.ndarray
) -> bool:
    print_header("Q-FORMER VALIDATION")

    vision_pt = torch.tensor(vision_features_np)
    query_tokens_pt = hf_model.query_tokens.expand(vision_pt.shape[0], -1, -1)

    with torch.no_grad():
        pt_out = to_np(
            hf_model.qformer(
                query_embeds=query_tokens_pt,
                encoder_hidden_states=vision_pt,
            ).last_hidden_state
        )

    keras_out = to_np(keras_qformer(vision_features_np, training=False))

    print(f"   -> HF    shape : {pt_out.shape}")
    print(f"   -> Keras shape : {keras_out.shape}")

    return report_diff("Q-Former query outputs", keras_out, pt_out)


def validate_projection(
    keras_proj, hf_model, qformer_out_np: np.ndarray
) -> bool:
    print_header("PROJECTION VALIDATION")

    qformer_pt = torch.tensor(qformer_out_np)
    with torch.no_grad():
        pt_out = to_np(hf_model.language_projection(qformer_pt))

    keras_inp = keras.Input(shape=(None, qformer_out_np.shape[-1]))
    keras_out_tensor = keras_proj(keras_inp)
    proj_model = keras.Model(inputs=keras_inp, outputs=keras_out_tensor)

    keras_out = to_np(proj_model(qformer_out_np))

    print(f"   -> HF    shape : {pt_out.shape}")
    print(f"   -> Keras shape : {keras_out.shape}")

    return report_diff("Projection outputs", keras_out, pt_out)


# ── Weight transfer ───────────────────────────────────────────────────────────

def transfer_vision_weights(keras_vision, hf_vision) -> None:
    print("Transferring Vision Encoder (EVA-CLIP) weights...")
    pt_state = hf_vision.state_dict()
    k_weights = keras_vision.weights

    hidden_dim = keras_vision.hidden_dim
    num_layers = keras_vision.num_layers
    num_heads = keras_vision.num_heads
    head_dim = hidden_dim // num_heads

    k_weights[0].assign(
        to_np(pt_state["embeddings.class_embedding"]).reshape(1, 1, hidden_dim)
    )
    k_weights[1].assign(
        to_np(pt_state["embeddings.patch_embedding.weight"]).transpose(
            2, 3, 1, 0
        )
    )
    k_weights[2].assign(to_np(pt_state["embeddings.patch_embedding.bias"]))
    k_weights[3].assign(
        to_np(pt_state["embeddings.position_embedding"]).reshape(-1, hidden_dim)
    )

    idx = 4
    for i in range(num_layers):
        pt_prefix = f"encoder.layers.{i}."

        k_weights[idx].assign(to_np(pt_state[f"{pt_prefix}layer_norm1.weight"]))
        k_weights[idx + 1].assign(
            to_np(pt_state[f"{pt_prefix}layer_norm1.bias"])
        )

        qkv_w = pt_state[f"{pt_prefix}self_attn.qkv.weight"]
        qkv_b = pt_state[f"{pt_prefix}self_attn.qkv.bias"]
        q_w, k_w, v_w = torch.chunk(qkv_w, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(qkv_b, 3, dim=0)

        k_weights[idx + 2].assign(
            to_np(q_w).T.reshape(hidden_dim, num_heads, head_dim)
        )
        k_weights[idx + 3].assign(to_np(q_b).reshape(num_heads, head_dim))
        k_weights[idx + 4].assign(
            to_np(k_w).T.reshape(hidden_dim, num_heads, head_dim)
        )
        k_weights[idx + 5].assign(to_np(k_b).reshape(num_heads, head_dim))
        k_weights[idx + 6].assign(
            to_np(v_w).T.reshape(hidden_dim, num_heads, head_dim)
        )
        k_weights[idx + 7].assign(to_np(v_b).reshape(num_heads, head_dim))

        k_weights[idx + 8].assign(
            to_np(
                pt_state[f"{pt_prefix}self_attn.projection.weight"]
            ).T.reshape(num_heads, head_dim, hidden_dim)
        )
        k_weights[idx + 9].assign(
            to_np(pt_state[f"{pt_prefix}self_attn.projection.bias"])
        )

        k_weights[idx + 10].assign(
            to_np(pt_state[f"{pt_prefix}layer_norm2.weight"])
        )
        k_weights[idx + 11].assign(
            to_np(pt_state[f"{pt_prefix}layer_norm2.bias"])
        )

        k_weights[idx + 12].assign(
            to_np(pt_state[f"{pt_prefix}mlp.fc1.weight"]).T
        )
        k_weights[idx + 13].assign(to_np(pt_state[f"{pt_prefix}mlp.fc1.bias"]))
        k_weights[idx + 14].assign(
            to_np(pt_state[f"{pt_prefix}mlp.fc2.weight"]).T
        )
        k_weights[idx + 15].assign(to_np(pt_state[f"{pt_prefix}mlp.fc2.bias"]))
        idx += 16

    k_weights[idx].assign(to_np(pt_state["post_layernorm.weight"]))
    k_weights[idx + 1].assign(to_np(pt_state["post_layernorm.bias"]))
    print("✓ Vision weights transferred")


def transfer_qformer_weights(keras_qformer, hf_model) -> None:
    print("Transferring Q-Former weights...")
    pt_qf = hf_model.qformer
    pt_state = pt_qf.state_dict()
    hidden_dim = keras_qformer.hidden_dim
    num_heads = keras_qformer.num_heads
    head_dim = hidden_dim // num_heads
    vision_dim = keras_qformer.vision_dim

    keras_qformer.query_tokens.assign(to_np(hf_model.query_tokens))
    keras_qformer.layer_norm.weights[0].assign(to_np(pt_qf.layernorm.weight))
    keras_qformer.layer_norm.weights[1].assign(to_np(pt_qf.layernorm.bias))

    def copy_attention(
        keras_attn, pt_prefix: str, is_cross: bool = False
    ) -> None:
        mha_w = keras_attn.attention.weights
        ln_w = keras_attn.layer_norm.weights
        kv_dim = vision_dim if is_cross else hidden_dim

        mha_w[0].assign(
            to_np(pt_state[f"{pt_prefix}attention.query.weight"]).T.reshape(
                hidden_dim, num_heads, head_dim
            )
        )
        mha_w[1].assign(
            to_np(pt_state[f"{pt_prefix}attention.query.bias"]).reshape(
                num_heads, head_dim
            )
        )
        mha_w[2].assign(
            to_np(pt_state[f"{pt_prefix}attention.key.weight"]).T.reshape(
                kv_dim, num_heads, head_dim
            )
        )
        mha_w[3].assign(
            to_np(pt_state[f"{pt_prefix}attention.key.bias"]).reshape(
                num_heads, head_dim
            )
        )
        mha_w[4].assign(
            to_np(pt_state[f"{pt_prefix}attention.value.weight"]).T.reshape(
                kv_dim, num_heads, head_dim
            )
        )
        mha_w[5].assign(
            to_np(pt_state[f"{pt_prefix}attention.value.bias"]).reshape(
                num_heads, head_dim
            )
        )
        mha_w[6].assign(
            to_np(pt_state[f"{pt_prefix}output.dense.weight"]).T.reshape(
                num_heads, head_dim, hidden_dim
            )
        )
        mha_w[7].assign(to_np(pt_state[f"{pt_prefix}output.dense.bias"]))
        ln_w[0].assign(to_np(pt_state[f"{pt_prefix}output.LayerNorm.weight"]))
        ln_w[1].assign(to_np(pt_state[f"{pt_prefix}output.LayerNorm.bias"]))

    for i, keras_layer in enumerate(keras_qformer.transformer_layers):
        pt_prefix = f"encoder.layer.{i}."
        copy_attention(keras_layer.self_attention, f"{pt_prefix}attention.")

        has_cross_attention = (
            keras_layer.has_cross_attention
            and f"{pt_prefix}crossattention.attention.query.weight" in pt_state
        )
        if has_cross_attention:
            copy_attention(
                keras_layer.cross_attention,
                f"{pt_prefix}crossattention.",
                is_cross=True,
            )

        keras_layer.intermediate_dense.weights[0].assign(
            to_np(pt_state[f"{pt_prefix}intermediate_query.dense.weight"]).T
        )
        keras_layer.intermediate_dense.weights[1].assign(
            to_np(pt_state[f"{pt_prefix}intermediate_query.dense.bias"])
        )
        keras_layer.output_dense.weights[0].assign(
            to_np(pt_state[f"{pt_prefix}output_query.dense.weight"]).T
        )
        keras_layer.output_dense.weights[1].assign(
            to_np(pt_state[f"{pt_prefix}output_query.dense.bias"])
        )
        keras_layer.output_layer_norm.weights[0].assign(
            to_np(pt_state[f"{pt_prefix}output_query.LayerNorm.weight"])
        )
        keras_layer.output_layer_norm.weights[1].assign(
            to_np(pt_state[f"{pt_prefix}output_query.LayerNorm.bias"])
        )

    print("✓ Q-Former weights transferred")


def transfer_projection_weights(keras_proj, hf_proj) -> None:
    print("Transferring Projection weights...")
    keras_proj.weights[0].assign(to_np(hf_proj.weight).T)
    keras_proj.weights[1].assign(to_np(hf_proj.bias))
    print("✓ Projection weights transferred")
