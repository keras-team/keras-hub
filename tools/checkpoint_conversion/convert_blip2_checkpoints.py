"""
Convert BLIP-2 checkpoints to KerasHub format.

Usage:
```shell
python convert_blip2_checkpoints.py \
    --model_id Salesforce/blip2-opt-2.7b \
    --output_dir blip2_opt_2_7b_converted
```
"""

import gc
import os
import random
import json

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide any CUDA devices

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import Blip2ForConditionalGeneration  # noqa: E402
from transformers import Blip2Processor  # noqa: E402

from keras_hub.src.models.blip2.blip2_backbone import Blip2Backbone  # noqa: E402
from keras_hub.src.models.blip2.blip2_causal_lm import Blip2CausalLM  # noqa: E402
from keras_hub.src.models.blip2.blip2_causal_lm_preprocessor import (  # noqa: E402
    Blip2CausalLMPreprocessor,
)
from keras_hub.src.models.blip2.blip2_custom_opt import Blip2CustomOPT  # noqa: E402
from keras_hub.src.models.blip2.blip2_image_converter import (  # noqa: E402
    Blip2ImageConverter,
)
from keras_hub.src.models.blip2.blip2_qformer import Blip2QFormer  # noqa: E402
from keras_hub.src.models.blip2.blip2_tokenizer import Blip2Tokenizer  # noqa: E402
from keras_hub.src.models.blip2.blip2_vision_encoder import (  # noqa: E402
    Blip2VisionEncoder,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_id",
    "Salesforce/blip2-opt-2.7b",
    "Hugging Face model ID.",
)
flags.DEFINE_string(
    "output_dir",
    "blip2_opt_2_7b_converted",
    "Output directory for converted weights and tokenizer assets.",
)

_VALIDATION_PROMPT = "Question: What is in this picture? Answer:"
_NUM_QUERY_TOKENS = 32


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


def _print_header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _report_diff(
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


def validate_vision_encoder(keras_vision, hf_model) -> bool:
    _print_header("VISION ENCODER VALIDATION")

    image_pil = Image.new("RGB", (224, 224), color=(114, 114, 114))
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    image_np = image_np[None]

    image_pt = torch.tensor(image_np).permute(0, 3, 1, 2)
    with torch.no_grad():
        pt_out = to_np(hf_model.vision_model(pixel_values=image_pt).last_hidden_state)

    keras_out = to_np(keras_vision(image_np))

    print(f"   -> HF    shape : {pt_out.shape}")
    print(f"   -> Keras shape : {keras_out.shape}")

    return _report_diff("Vision encoder hidden states", keras_out, pt_out)


def validate_qformer(keras_qformer, hf_model, vision_features_np: np.ndarray) -> bool:
    _print_header("Q-FORMER VALIDATION")

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

    return _report_diff("Q-Former query outputs", keras_out, pt_out)


def validate_projection(keras_proj, hf_model, qformer_out_np: np.ndarray) -> bool:
    _print_header("PROJECTION VALIDATION")

    qformer_pt = torch.tensor(qformer_out_np)
    with torch.no_grad():
        pt_out = to_np(hf_model.language_projection(qformer_pt))

    keras_inp = keras.Input(shape=(None, 768))
    keras_out_tensor = keras_proj(keras_inp)
    proj_model = keras.Model(inputs=keras_inp, outputs=keras_out_tensor)

    keras_out = to_np(proj_model(qformer_out_np))

    print(f"   -> HF    shape : {pt_out.shape}")
    print(f"   -> Keras shape : {keras_out.shape}")

    return _report_diff("Projection outputs", keras_out, pt_out)


def validate_opt(keras_opt, hf_model, qformer_out_np: np.ndarray, hf_processor) -> bool:
    _print_header("OPT (LANGUAGE MODEL) VALIDATION")

    hf_inputs = hf_processor(
        images=Image.new("RGB", (224, 224), color=(114, 114, 114)),
        text=_VALIDATION_PROMPT,
        return_tensors="pt",
        padding=False,
    )
    text_ids_pt = hf_inputs["input_ids"]
    text_len = int(text_ids_pt.shape[1])

    # HF expects projected features
    with torch.no_grad():
        projected_pt = hf_model.language_projection(torch.tensor(qformer_out_np))
        text_embeds = hf_model.language_model.model.decoder.embed_tokens(text_ids_pt)
        inputs_embeds = torch.cat([projected_pt, text_embeds], dim=1)
        attention_mask = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long)
        hf_lm_out = hf_model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
    hf_text_logits = hf_lm_out.logits.float().numpy()[:, _NUM_QUERY_TOKENS:, :]

    token_ids = text_ids_pt.numpy()
    padding_mask = np.ones_like(token_ids, dtype=bool)
    keras_hidden = keras_opt(
        {
            "qformer_features": qformer_out_np,  # Raw Q-Former features, internally projected
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
    )
    keras_logits_full = keras_opt.embeddings_layer.token_embedding(
        keras_hidden, reverse=True
    )
    keras_text_logits = to_np(keras_logits_full[:, _NUM_QUERY_TOKENS:, :])

    print(f"   -> HF    logits shape (text only) : {hf_text_logits.shape}")
    print(f"   -> Keras logits shape (text only) : {keras_text_logits.shape}")
    print(f"   -> HF    logits[:5] : {hf_text_logits[0, 0, :5]}")
    print(f"   -> Keras logits[:5] : {keras_text_logits[0, 0, :5]}")

    hf_pred = np.argmax(hf_text_logits[0], axis=-1)
    keras_pred = np.argmax(keras_text_logits[0], axis=-1)
    match = int(np.sum(hf_pred == keras_pred))
    print(f"   -> Argmax token match : {match}/{text_len}")

    return _report_diff(
        "OPT logits (text positions)", keras_text_logits, hf_text_logits
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline validator
# ─────────────────────────────────────────────────────────────────────────────


def validate_output(keras_backbone, keras_preprocessor, hf_model, hf_processor) -> None:
    _print_header("FULL PIPELINE — FORWARD PASS")

    keras_params = keras_backbone.count_params()
    hf_params = sum(
        p.numel() for name, p in hf_model.named_parameters() if "lm_head" not in name
    )
    print("🔶 Parameter count comparison:")
    print(f"   -> KerasHub    : {keras_params:,}")
    print(f"   -> HuggingFace : {hf_params:,}")
    if keras_params != hf_params:
        print("   -> ⚠️  counts differ (expected: vocab padding only)")
    else:
        print("   -> ✅ counts match")

    image_pil = Image.new("RGB", (224, 224), color=(114, 114, 114))
    image_np = np.array(image_pil)

    hf_inputs = hf_processor(
        images=image_pil,
        text=_VALIDATION_PROMPT,
        return_tensors="pt",
        padding=False,
    )
    text_len = hf_inputs["input_ids"].shape[1] - _NUM_QUERY_TOKENS
    print(f"\n   -> Text length : {text_len} tokens")

    with torch.no_grad():
        hf_out = hf_model(**hf_inputs)

    hf_logits = hf_out.logits.detach().cpu().float().numpy()
    hf_text_logits = hf_logits[:, _NUM_QUERY_TOKENS:, :]

    keras_inputs = keras_preprocessor.generate_preprocess(
        {"images": image_np[None], "text": [_VALIDATION_PROMPT]},
        sequence_length=text_len,
    )
    keras_hidden = keras_backbone(keras_inputs)

    lm = keras_backbone.language_model
    keras_logits_full = lm.embeddings_layer.token_embedding(keras_hidden, reverse=True)
    keras_text_logits = to_np(keras_logits_full[:, _NUM_QUERY_TOKENS:, :])

    keras_vocab = keras_text_logits.shape[-1]

    print(f"   -> HF    logits shape : {hf_text_logits.shape}")
    print(f"   -> Keras logits shape : {keras_text_logits.shape}")
    print(f"   -> HF    logits[:5]   : {hf_text_logits[0, 0, :5]}")
    print(f"   -> Keras logits[:5]   : {keras_text_logits[0, 0, :5]}")

    hf_pred = np.argmax(hf_text_logits[0], axis=-1)
    keras_pred = np.argmax(keras_text_logits[0], axis=-1)
    match = int(np.sum(hf_pred == keras_pred))
    total = len(hf_pred)
    print(f"   -> Argmax token match : {match}/{total}")
    print(f"   -> {'✅ Yes' if match == total else '⚠️  No'}")

    _report_diff("Full pipeline logits", keras_text_logits, hf_text_logits)


def validate_generate(keras_causal_lm, hf_model, hf_processor) -> None:
    _print_header("GENERATION VALIDATION")

    image_pil = Image.new("RGB", (224, 224), color=(114, 114, 114))
    image_np = np.array(image_pil)
    max_new_tokens = 30

    keras_causal_lm.compile(sampler="greedy")
    keras_output = keras_causal_lm.generate(
        {"images": image_np, "text": [_VALIDATION_PROMPT]},
        max_length=max_new_tokens,
        strip_prompt=True,
    )
    keras_text = keras_output[0] if isinstance(keras_output, (list, tuple)) else keras_output

    hf_inputs = hf_processor(
        images=image_pil,
        text=_VALIDATION_PROMPT,
        return_tensors="pt",
        padding=False,
    )
    with torch.no_grad():
        hf_generated = hf_model.generate(
            **hf_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    prompt_len = hf_inputs["input_ids"].shape[1]
    hf_text = hf_processor.batch_decode(
        hf_generated[:, prompt_len:],
        skip_special_tokens=True,
    )[0]

    print(f"\n   -> Keras output       : {keras_text!r}")
    print(f"   -> HuggingFace output : {hf_text!r}")
    match = keras_text.strip() == hf_text.strip()
    print(f"   -> {'✅ Match' if match else '⚠️  No match'}")


# ─────────────────────────────────────────────────────────────────────────────
# Weight transfer — Vision Encoder
# ─────────────────────────────────────────────────────────────────────────────


def transfer_vision_weights(keras_vision, hf_vision) -> None:
    print("Transferring Vision Encoder (EVA-CLIP) weights...")
    pt_state = hf_vision.state_dict()
    k_weights = keras_vision.weights

    k_weights[0].assign(
        to_np(pt_state["embeddings.class_embedding"]).reshape(1, 1, 1408)
    )
    k_weights[1].assign(
        to_np(pt_state["embeddings.patch_embedding.weight"]).transpose(2, 3, 1, 0)
    )
    k_weights[2].assign(to_np(pt_state["embeddings.patch_embedding.bias"]))
    k_weights[3].assign(
        to_np(pt_state["embeddings.position_embedding"]).reshape(257, 1408)
    )

    idx = 4
    for i in range(39):
        pt_prefix = f"encoder.layers.{i}."

        k_weights[idx].assign(to_np(pt_state[f"{pt_prefix}layer_norm1.weight"]))
        k_weights[idx + 1].assign(to_np(pt_state[f"{pt_prefix}layer_norm1.bias"]))

        qkv_w = pt_state[f"{pt_prefix}self_attn.qkv.weight"]
        qkv_b = pt_state[f"{pt_prefix}self_attn.qkv.bias"]
        q_w, k_w, v_w = torch.chunk(qkv_w, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(qkv_b, 3, dim=0)

        k_weights[idx + 2].assign(to_np(q_w).T.reshape(1408, 16, 88))
        k_weights[idx + 3].assign(to_np(q_b).reshape(16, 88))
        k_weights[idx + 4].assign(to_np(k_w).T.reshape(1408, 16, 88))
        k_weights[idx + 5].assign(to_np(k_b).reshape(16, 88))
        k_weights[idx + 6].assign(to_np(v_w).T.reshape(1408, 16, 88))
        k_weights[idx + 7].assign(to_np(v_b).reshape(16, 88))

        k_weights[idx + 8].assign(
            to_np(pt_state[f"{pt_prefix}self_attn.projection.weight"]).T.reshape(
                16, 88, 1408
            )
        )
        k_weights[idx + 9].assign(
            to_np(pt_state[f"{pt_prefix}self_attn.projection.bias"])
        )

        k_weights[idx + 10].assign(to_np(pt_state[f"{pt_prefix}layer_norm2.weight"]))
        k_weights[idx + 11].assign(to_np(pt_state[f"{pt_prefix}layer_norm2.bias"]))

        k_weights[idx + 12].assign(to_np(pt_state[f"{pt_prefix}mlp.fc1.weight"]).T)
        k_weights[idx + 13].assign(to_np(pt_state[f"{pt_prefix}mlp.fc1.bias"]))
        k_weights[idx + 14].assign(to_np(pt_state[f"{pt_prefix}mlp.fc2.weight"]).T)
        k_weights[idx + 15].assign(to_np(pt_state[f"{pt_prefix}mlp.fc2.bias"]))
        idx += 16

    k_weights[idx].assign(to_np(pt_state["post_layernorm.weight"]))
    k_weights[idx + 1].assign(to_np(pt_state["post_layernorm.bias"]))
    print("✓ Vision weights transferred")


# ─────────────────────────────────────────────────────────────────────────────
# Weight transfer — Q-Former
# ─────────────────────────────────────────────────────────────────────────────


def transfer_qformer_weights(keras_qformer, hf_model) -> None:
    print("Transferring Q-Former weights...")
    pt_qf = hf_model.qformer
    pt_state = pt_qf.state_dict()
    hidden_dim = 768
    num_heads = 12
    head_dim = 64
    vision_dim = 1408

    keras_qformer.query_tokens.assign(to_np(hf_model.query_tokens))
    keras_qformer.layernorm.weights[0].assign(to_np(pt_qf.layernorm.weight))
    keras_qformer.layernorm.weights[1].assign(to_np(pt_qf.layernorm.bias))

    def copy_attention(keras_attn, pt_prefix: str, is_cross: bool = False) -> None:
        mha_w = keras_attn.mha.weights
        ln_w = keras_attn.LayerNorm.weights
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

    for i, keras_layer in enumerate(keras_qformer.qformer_layers):
        pt_prefix = f"encoder.layer.{i}."
        copy_attention(keras_layer.attention, f"{pt_prefix}attention.")
        if keras_layer.has_cross_attention:
            copy_attention(
                keras_layer.crossattention,
                f"{pt_prefix}crossattention.",
                is_cross=True,
            )
        keras_layer.intermediate_query.weights[0].assign(
            to_np(pt_state[f"{pt_prefix}intermediate_query.dense.weight"]).T
        )
        keras_layer.intermediate_query.weights[1].assign(
            to_np(pt_state[f"{pt_prefix}intermediate_query.dense.bias"])
        )
        keras_layer.output_query_dense.weights[0].assign(
            to_np(pt_state[f"{pt_prefix}output_query.dense.weight"]).T
        )
        keras_layer.output_query_dense.weights[1].assign(
            to_np(pt_state[f"{pt_prefix}output_query.dense.bias"])
        )
        keras_layer.output_query_LN.weights[0].assign(
            to_np(pt_state[f"{pt_prefix}output_query.LayerNorm.weight"])
        )
        keras_layer.output_query_LN.weights[1].assign(
            to_np(pt_state[f"{pt_prefix}output_query.LayerNorm.bias"])
        )

    print("✓ Q-Former weights transferred")


def transfer_projection_weights(keras_proj, hf_proj) -> None:
    print("Transferring Projection weights...")
    keras_proj.weights[0].assign(to_np(hf_proj.weight).T)
    keras_proj.weights[1].assign(to_np(hf_proj.bias))
    print("✓ Projection weights transferred")


def transfer_opt_weights(keras_opt, hf_opt) -> None:
    """Transfer weights from HuggingFace OPT into Blip2CustomOPT.

    Position embedding note
    -----------------------
    HuggingFace OPT's ``embed_positions`` table has shape ``(2050, 2560)``:
    rows 0-1 are *reserved* and never used; real positions start at row 2.

    ``Blip2OPTEmbeddings`` uses a plain ``keras.layers.Embedding`` of size
    ``max_sequence_length + 2 = 2050`` and looks up indices that already
    include the HF +2 offset (``position_offset = num_query_tokens + 2``).
    Therefore we load the **full** 2050-row table without stripping any rows.
    """
    print("Transferring OPT weights...")

    pt_dec = hf_opt.model.decoder
    pt_state = pt_dec.state_dict()

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Token embeddings: HF stores 50304 rows.
    keras_opt.embeddings_layer.token_embedding.embeddings.assign(
        to_np(pt_dec.embed_tokens.weight)
    )

    # Position embeddings: load the FULL (2050, 2560) table.
    # Keras indices already include the +2 HF offset, so no row-stripping here.
    keras_opt.embeddings_layer.position_embedding.embeddings.assign(
        to_np(pt_dec.embed_positions.weight)  # shape (2050, 2560)
    )

    # ── Transformer layers ────────────────────────────────────────────────────
    for i in range(keras_opt.num_layers):
        p = f"layers.{i}."
        layer = keras_opt.transformer_layers[i]

        # Self-attention layer norm (pre-norm)
        layer.self_attn_layer_norm.gamma.assign(
            to_np(pt_state[f"{p}self_attn_layer_norm.weight"])
        )
        layer.self_attn_layer_norm.beta.assign(
            to_np(pt_state[f"{p}self_attn_layer_norm.bias"])
        )

        # Q / K / V projections — HF shape (out, in), Keras needs (in, out)
        layer.self_attn.q_proj.kernel.assign(
            to_np(pt_state[f"{p}self_attn.q_proj.weight"]).T
        )
        layer.self_attn.q_proj.bias.assign(
            to_np(pt_state[f"{p}self_attn.q_proj.bias"])
        )
        layer.self_attn.k_proj.kernel.assign(
            to_np(pt_state[f"{p}self_attn.k_proj.weight"]).T
        )
        layer.self_attn.k_proj.bias.assign(
            to_np(pt_state[f"{p}self_attn.k_proj.bias"])
        )
        layer.self_attn.v_proj.kernel.assign(
            to_np(pt_state[f"{p}self_attn.v_proj.weight"]).T
        )
        layer.self_attn.v_proj.bias.assign(
            to_np(pt_state[f"{p}self_attn.v_proj.bias"])
        )

        # Output projection
        layer.self_attn.out_proj.kernel.assign(
            to_np(pt_state[f"{p}self_attn.out_proj.weight"]).T
        )
        layer.self_attn.out_proj.bias.assign(
            to_np(pt_state[f"{p}self_attn.out_proj.bias"])
        )

        # FFN layer norm (pre-norm)
        layer.final_layer_norm.gamma.assign(
            to_np(pt_state[f"{p}final_layer_norm.weight"])
        )
        layer.final_layer_norm.beta.assign(
            to_np(pt_state[f"{p}final_layer_norm.bias"])
        )

        # Feed-forward network
        layer.fc1.kernel.assign(to_np(pt_state[f"{p}fc1.weight"]).T)
        layer.fc1.bias.assign(to_np(pt_state[f"{p}fc1.bias"]))
        layer.fc2.kernel.assign(to_np(pt_state[f"{p}fc2.weight"]).T)
        layer.fc2.bias.assign(to_np(pt_state[f"{p}fc2.bias"]))

    # ── Final decoder layer norm ──────────────────────────────────────────────
    keras_opt.layer_norm.gamma.assign(to_np(pt_dec.final_layer_norm.weight))
    keras_opt.layer_norm.beta.assign(to_np(pt_dec.final_layer_norm.bias))

    # ── Spot-check layer 0 ────────────────────────────────────────────────────
    print("\n── Layer 0 weight spot-check (first 5 values) ──")
    layer0 = keras_opt.transformer_layers[0]
    _spot = [
        (
            "token_embedding",
            keras_opt.embeddings_layer.token_embedding.embeddings,
            to_np(pt_dec.embed_tokens.weight),
        ),
        (
            "position_embedding (row 34 = first text pos)",
            keras_opt.embeddings_layer.position_embedding.embeddings,
            to_np(pt_dec.embed_positions.weight),
        ),
        (
            "self_attn_layer_norm gamma",
            layer0.self_attn_layer_norm.gamma,
            to_np(pt_state["layers.0.self_attn_layer_norm.weight"]),
        ),
        (
            "q_proj kernel",
            layer0.self_attn.q_proj.kernel,
            to_np(pt_state["layers.0.self_attn.q_proj.weight"]).T,
        ),
        (
            "k_proj kernel",
            layer0.self_attn.k_proj.kernel,
            to_np(pt_state["layers.0.self_attn.k_proj.weight"]).T,
        ),
        (
            "v_proj kernel",
            layer0.self_attn.v_proj.kernel,
            to_np(pt_state["layers.0.self_attn.v_proj.weight"]).T,
        ),
        (
            "out_proj kernel",
            layer0.self_attn.out_proj.kernel,
            to_np(pt_state["layers.0.self_attn.out_proj.weight"]).T,
        ),
        (
            "final_layer_norm gamma",
            layer0.final_layer_norm.gamma,
            to_np(pt_state["layers.0.final_layer_norm.weight"]),
        ),
        (
            "fc1 kernel",
            layer0.fc1.kernel,
            to_np(pt_state["layers.0.fc1.weight"]).T,
        ),
        (
            "fc2 kernel",
            layer0.fc2.kernel,
            to_np(pt_state["layers.0.fc2.weight"]).T,
        ),
    ]
    all_ok = True
    for label, keras_w, hf_arr in _spot:
        k_vals = to_np(keras_w).flatten()[:5]
        h_vals = hf_arr.flatten()[:5]
        ok = np.allclose(k_vals, h_vals, atol=1e-6)
        status = "✅" if ok else "⚠️  MISMATCH"
        print(f"  {status}  {label}")
        if not ok:
            print(f"       Keras : {k_vals}")
            print(f"       HF    : {h_vals}")
            all_ok = False
    if all_ok:
        print("  All spot-checks passed.")
    else:
        print("  ⚠️  Fix failing weights before continuing.")

    print("✓ OPT weights transferred")


def main(_) -> None:
    set_seed(42)
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    print(f"Loading HF model: {FLAGS.model_id}")
    hf_model = Blip2ForConditionalGeneration.from_pretrained(
        FLAGS.model_id, torch_dtype=torch.float32
    )
    # Force every submodule to float32. Some transformers versions silently
    # keep layernorms or buffers in a different dtype even when torch_dtype
    # is set at load time. This is a no-op when everything is already fp32.
    hf_model = hf_model.float()
    hf_model.eval()
    hf_processor = Blip2Processor.from_pretrained(FLAGS.model_id)

    # ── Verify all HF parameters are fp32 ────────────────────────────────────
    bad_params = [
        (n, p.dtype)
        for n, p in hf_model.named_parameters()
        if p.dtype != torch.float32
    ]
    if bad_params:
        print("⚠️  Non-fp32 parameters found after cast — forcing individually:")
        for name, dtype in bad_params:
            print(f"   {name}: {dtype}")
        # Should not happen after .float(), but guard anyway
        hf_model = hf_model.to(torch.float32)
    else:
        print("✅ All HF parameters confirmed float32")

    vision_config = {
        "image_size": 224,
        "patch_size": 14,
        "num_layers": 39,
        "num_heads": 16,
        "hidden_dim": 1408,
        "intermediate_dim": 6144,
        "use_patch_bias": True,
        "use_class_token": True,
        "use_mha_bias": True,
        "use_mlp_bias": True,
        "dropout_rate": 0.0,
        "layer_norm_epsilon": 1e-6,
    }
    v_enc = Blip2VisionEncoder(**vision_config)
    v_enc.build((None, 224, 224, 3))
    transfer_vision_weights(v_enc, hf_model.vision_model)

    qf = Blip2QFormer(num_query_tokens=_NUM_QUERY_TOKENS)
    qf.build((None, 257, 1408))
    transfer_qformer_weights(qf, hf_model)

    opt_config = {
        "vocabulary_size": 50304,
        "num_layers": 32,
        "num_heads": 32,
        "hidden_dim": 2560,
        "intermediate_dim": 10240,
        "num_query_tokens": _NUM_QUERY_TOKENS,  # sets position_offset = 34
        "dropout": 0.1,
        "max_sequence_length": 2048,
    }
    opt = Blip2CustomOPT(**opt_config)
    opt.build(
        {
            "qformer_features": (None, _NUM_QUERY_TOKENS, 768),
            "token_ids": (None, 10),
            "padding_mask": (None, 10),
        }
    )
    transfer_opt_weights(opt, hf_model.language_model)
    transfer_projection_weights(opt.language_projection, hf_model.language_projection)

    backbone = Blip2Backbone(
        vision_encoder=v_enc,
        qformer=qf,
        language_model=opt,
    )

    print("\nExtracting tokenizer assets...")
    tokenizer_json_path = hf_hub_download(
        repo_id=FLAGS.model_id, filename="tokenizer.json"
    )
    with open(tokenizer_json_path) as f:
        tok_config = json.load(f)

    full_vocab = tok_config["model"]["vocab"].copy()
    added_tokens_list = tok_config.get("added_tokens", [])
    for token_data in added_tokens_list:
        full_vocab[token_data["content"]] = token_data["id"]

    vocab_path = os.path.join(FLAGS.output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(full_vocab, f)

    raw_merges = tok_config["model"]["merges"]
    cleaned_merges = [" ".join(m) if isinstance(m, list) else m for m in raw_merges]
    merges_path = os.path.join(FLAGS.output_dir, "merges.txt")
    with open(merges_path, "w") as f:
        f.write("\n".join(cleaned_merges))

    unsplittable = [t["content"] for t in added_tokens_list]
    unsplittable_path = os.path.join(FLAGS.output_dir, "unsplittable.json")
    with open(unsplittable_path, "w") as f:
        json.dump(unsplittable, f)
    print(f"✓ Tokenizer assets saved to {FLAGS.output_dir}")

    tokenizer = Blip2Tokenizer(
        vocabulary=vocab_path,
        merges=merges_path,
        unsplittable_tokens=unsplittable,
    )
    preprocessor = Blip2CausalLMPreprocessor(
        tokenizer=tokenizer,
        image_converter=Blip2ImageConverter(image_size=(224, 224)),
    )
    causal_lm = Blip2CausalLM(backbone=backbone, preprocessor=preprocessor)

    # ── Per-submodule validation ───────────────────────────────────────────
    image_pil = Image.new("RGB", (224, 224), color=(114, 114, 114))
    image_np = np.array(image_pil).astype(np.float32) / 255.0

    vision_ok = validate_vision_encoder(v_enc, hf_model)

    image_pt = torch.tensor(image_np[None]).permute(0, 3, 1, 2)
    with torch.no_grad():
        vision_features_np = to_np(
            hf_model.vision_model(pixel_values=image_pt).last_hidden_state
        )

    qformer_ok = validate_qformer(qf, hf_model, vision_features_np)

    with torch.no_grad():
        query_tokens_pt = hf_model.query_tokens.expand(1, -1, -1)
        qformer_out_np = to_np(
            hf_model.qformer(
                query_embeds=query_tokens_pt,
                encoder_hidden_states=torch.tensor(vision_features_np),
            ).last_hidden_state
        )

    proj_ok = validate_projection(opt.language_projection, hf_model, qformer_out_np)

    with torch.no_grad():
        projected_np = to_np(
            hf_model.language_projection(torch.tensor(qformer_out_np))
        )

    opt_ok = validate_opt(opt, hf_model, qformer_out_np, hf_processor)

    # ── Full pipeline validation ───────────────────────────────────────────
    validate_output(backbone, preprocessor, hf_model, hf_processor)
    validate_generate(causal_lm, hf_model, hf_processor)

    # ── Debug Pipeline ────────────────────────────────────────────────────
    debug_pipeline(backbone, preprocessor, hf_model, hf_processor)

    # ── Summary ───────────────────────────────────────────────────────────
    _print_header("SUBMODULE VALIDATION SUMMARY")
    for name, ok in [
        ("Vision Encoder", vision_ok),
        ("Q-Former", qformer_ok),
        ("Projection", proj_ok),
        ("OPT", opt_ok),
    ]:
        status = "✅ PASS" if ok else "⚠️  FAIL"
        print(f"   {status}  {name}")

    # ── Save ──────────────────────────────────────────────────────────────
    _print_header("SAVING WEIGHTS")
    weights_path = os.path.join(FLAGS.output_dir, "model.weights.h5")
    backbone.save_weights(weights_path)
    print(f"✓ Weights saved to {weights_path}")

    del hf_model, causal_lm, backbone
    gc.collect()

    print("\n✓ Conversion complete.")


def _stats(arr, label):
    print(
        f"  {label:45s}  mean={arr.mean():.5f}  std={arr.std():.5f}  "
        f"min={arr.min():.5f}  max={arr.max():.5f}"
    )


def debug_pipeline(keras_backbone, keras_preprocessor, hf_model, hf_processor):
    print("\n" + "=" * 70)
    print("  PIPELINE DIVERGENCE DEBUG")
    print("=" * 70)

    image_pil = Image.new("RGB", (224, 224), color=(114, 114, 114))
    image_np = np.array(image_pil)

    with torch.no_grad():
        # ── Step 1: Compare pixel values after preprocessing ──────────────────
        print("\n[1] IMAGE PREPROCESSING")

        hf_inputs = hf_processor(
            images=image_pil,
            text=_VALIDATION_PROMPT,
            return_tensors="pt",
            padding=False,
        )
        hf_pixels = hf_inputs["pixel_values"].numpy()  # (1, 3, H, W)
        hf_pixels_nhwc = hf_pixels.transpose(0, 2, 3, 1)  # (1, H, W, 3)

        keras_inputs = keras_preprocessor.generate_preprocess(
            {"images": image_np[None], "text": [_VALIDATION_PROMPT]},
            sequence_length=hf_inputs["input_ids"].shape[1] - 32,
        )
        keras_pixels = to_np(keras_inputs["images"])  # (1, H, W, 3)

        _stats(hf_pixels_nhwc, "HF pixel_values (NHWC)")
        _stats(keras_pixels, "Keras image pixels")

        pixel_match = np.allclose(hf_pixels_nhwc, keras_pixels, atol=1e-3)
        print(
            f"  -> pixels match: {'✅' if pixel_match else '⚠️  MISMATCH — preprocessor bug'}"
        )
        if not pixel_match:
            diff = np.abs(hf_pixels_nhwc - keras_pixels)
            print(f"     max pixel diff: {diff.max():.5f}")

        # ── Step 2: Vision encoder — feed HF pixels to Keras encoder ──────────
        print("\n[2] VISION ENCODER  (fed HF pixels)")

        hf_vis = hf_model.vision_model(
            pixel_values=hf_inputs["pixel_values"]
        ).last_hidden_state
        hf_vis_np = hf_vis.numpy()

        keras_vis_from_hf_pixels = to_np(
            keras_backbone.vision_encoder(hf_pixels_nhwc)
        )
        keras_vis_from_keras_pixels = to_np(
            keras_backbone.vision_encoder(keras_pixels)
        )

        _stats(hf_vis_np, "HF   vision features")
        _stats(keras_vis_from_hf_pixels, "Keras vision (HF pixels)")
        _stats(keras_vis_from_keras_pixels, "Keras vision (Keras pixels)")

        vis_match_hf_in = np.allclose(hf_vis_np, keras_vis_from_hf_pixels, atol=1e-2)
        vis_match_own_in = np.allclose(
            hf_vis_np, keras_vis_from_keras_pixels, atol=1e-2
        )
        print(
            f"  -> Keras(HF pixels)   vs HF: {'✅' if vis_match_hf_in  else '⚠️  MISMATCH'}"
        )
        print(
            f"  -> Keras(own pixels)  vs HF: {'✅' if vis_match_own_in else '⚠️  MISMATCH — pixel preprocessing causes this'}"
        )

        # ── Step 3: Q-Former — feed HF vision features ────────────────────────
        print("\n[3] Q-FORMER  (fed HF vision features)")

        qt = hf_model.query_tokens.expand(1, -1, -1)
        hf_qf_np = (
            hf_model.qformer(
                query_embeds=qt,
                encoder_hidden_states=hf_vis,
            )
            .last_hidden_state.numpy()
        )

        keras_qf_from_hf = to_np(keras_backbone.qformer(hf_vis_np))

        _stats(hf_qf_np, "HF   Q-Former output")
        _stats(keras_qf_from_hf, "Keras Q-Former (HF vis feats)")
        qf_match = np.allclose(hf_qf_np, keras_qf_from_hf, atol=1e-2)
        print(f"  -> match: {'✅' if qf_match else '⚠️  MISMATCH'}")

        # ── Step 4: Projection ─────────────────────────────────────────────────
        print("\n[4] PROJECTION  (fed HF Q-Former output)")

        hf_proj_np = hf_model.language_projection(torch.tensor(hf_qf_np)).numpy()

        keras_proj_from_hf = to_np(keras_backbone.language_model.language_projection(hf_qf_np))

        _stats(hf_proj_np, "HF   projection output")
        _stats(keras_proj_from_hf, "Keras projection (HF Q-Former)")
        proj_match = np.allclose(hf_proj_np, keras_proj_from_hf, atol=1e-2)
        print(f"  -> match: {'✅' if proj_match else '⚠️  MISMATCH'}")

        # ── Step 5: OPT — feed HF projected features + Keras token ids ────────
        print("\n[5] OPT  (HF Q-Former features, Keras token_ids/mask)")

        token_ids = to_np(keras_inputs["token_ids"])
        padding_mask = to_np(keras_inputs["padding_mask"])

        # 5a: use HF Q-Former features
        lm = keras_backbone.language_model
        keras_hidden_hf_proj = to_np(
            lm(
                {
                    "qformer_features": hf_qf_np,
                    "token_ids": token_ids,
                    "padding_mask": padding_mask,
                }
            )
        )
        keras_logits_hf_proj = to_np(
            lm.embeddings_layer.token_embedding(keras_hidden_hf_proj, reverse=True)
        )[:, 32:, :]

        # 5b: use Keras end-to-end
        keras_hidden_own = to_np(keras_backbone(keras_inputs))
        keras_logits_own = to_np(
            lm.embeddings_layer.token_embedding(keras_hidden_own, reverse=True)
        )[:, 32:, :]

        # 5c: HF reference logits
        hf_text_embeds = hf_model.language_model.model.decoder.embed_tokens(
            hf_inputs["input_ids"][:, 32:]
        )
        full_embeds = torch.cat([torch.tensor(hf_proj_np), hf_text_embeds], dim=1)
        attn_mask = torch.ones(1, full_embeds.shape[1], dtype=torch.long)
        hf_logits_np = (
            hf_model.language_model(inputs_embeds=full_embeds, attention_mask=attn_mask)
            .logits.float()
            .numpy()[:, 32:, :]
        )

        _stats(hf_logits_np, "HF   logits")
        _stats(keras_logits_hf_proj, "Keras logits (HF proj features)")
        _stats(keras_logits_own, "Keras logits (own features)")

        opt_match_hf_proj = np.allclose(hf_logits_np, keras_logits_hf_proj, atol=1e-2)
        opt_match_own = np.allclose(hf_logits_np, keras_logits_own, atol=1e-2)
        print(
            f"  -> Keras(HF proj)  vs HF: {'✅' if opt_match_hf_proj else '⚠️  MISMATCH — OPT or token_ids differ'}"
        )
        print(
            f"  -> Keras(own proj) vs HF: {'✅' if opt_match_own     else '⚠️  MISMATCH'}"
        )

        # ── Step 6: Token ID comparison ────────────────────────────────────────
        print("\n[6] TOKEN IDS")
        hf_token_ids = hf_inputs["input_ids"][:, 32:].numpy()
        print(f"  HF   token_ids : {hf_token_ids[0]}")
        print(f"  Keras token_ids: {token_ids[0]}")
        ids_match = np.array_equal(hf_token_ids, token_ids)
        print(
            f"  -> match: {'✅' if ids_match else '⚠️  MISMATCH — tokenizer bug'}"
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("SUMMARY")
    print(f"  [1] Pixels match          : {'✅' if pixel_match        else '⚠️'}")
    print(f"  [2] Vision (HF pixels)    : {'✅' if vis_match_hf_in    else '⚠️'}")
    print(f"  [2] Vision (own pixels)   : {'✅' if vis_match_own_in   else '⚠️'}")
    print(f"  [3] Q-Former              : {'✅' if qf_match            else '⚠️'}")
    print(f"  [4] Projection            : {'✅' if proj_match          else '⚠️'}")
    print(f"  [5] OPT (HF proj feats)   : {'✅' if opt_match_hf_proj  else '⚠️'}")
    print(f"  [5] OPT (own proj feats)  : {'✅' if opt_match_own      else '⚠️'}")
    print(f"  [6] Token IDs             : {'✅' if ids_match           else '⚠️'}")
    print("-" * 70)


if __name__ == "__main__":
    app.run(main)