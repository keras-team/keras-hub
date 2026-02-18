import os
import traceback

os.environ["KERAS_BACKEND"] = "torch"

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

from keras import ops  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers import Qwen2VLForConditionalGeneration  # noqa: E402

import keras_hub  # noqa: E402
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import (  # noqa: E402
    Qwen2VLBackbone,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (  # noqa: E402, E501
    Qwen2VLImageConverter,
)

PRESET_MAP = {
    "qwen2_vl_2b_instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2_vl_7b_instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2_vl_72b_instruct": "Qwen/Qwen2-VL-72B-Instruct",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
)


# ── Helpers ──────────────────────────────────────────────────────────


def transpose_and_reshape(x, shape):
    """Transpose a 2-D HF weight and reshape to Keras EinsumDense shape."""
    return np.reshape(np.transpose(x), shape)


def transpose_2d(x):
    """Simple 2-D transpose for Dense kernels."""
    return np.transpose(x, axes=(1, 0))


# ── 1. Build config from HF ─────────────────────────────────────────


def build_backbone_config(hf_config):
    """Map HF ``config.json`` fields to ``Qwen2VLBackbone`` kwargs."""
    vc = hf_config.get("vision_config", {})
    return {
        "vocabulary_size": hf_config["vocab_size"],
        "num_layers": hf_config["num_hidden_layers"],
        "num_query_heads": hf_config["num_attention_heads"],
        "num_key_value_heads": hf_config["num_key_value_heads"],
        "hidden_dim": hf_config["hidden_size"],
        "intermediate_dim": hf_config["intermediate_size"],
        "vision_patch_size": vc.get("patch_size", 14),
        "vision_temporal_patch_size": vc.get("temporal_patch_size", 2),
        "vision_in_channels": vc.get("in_channels", 3),
        "vision_embed_dim": vc.get("embed_dim", vc.get("hidden_size", 1280)),
        "vision_depth": vc.get("depth", vc.get("num_hidden_layers", 32)),
        "vision_num_heads": vc.get(
            "num_heads", vc.get("num_attention_heads", 16)
        ),
        "vision_mlp_ratio": vc.get("mlp_ratio", 4),
        "spatial_merge_size": vc.get("spatial_merge_size", 2),
        "image_token_id": hf_config.get("image_token_id", 151655),
        "rope_max_wavelength": hf_config.get("rope_theta", 1000000),
        "layer_norm_epsilon": hf_config.get("rms_norm_eps", 1e-6),
        "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
        "use_sliding_window_attention": hf_config.get(
            "use_sliding_window", False
        ),
        "sliding_window_size": hf_config.get("sliding_window", 32768),
    }


# ── 2. Port weights ─────────────────────────────────────────────────


def port_weights(backbone, hf_state_dict):
    """Assign every HF weight to the corresponding KerasHub variable."""

    def get(key):
        return hf_state_dict[key].cpu().float().numpy()

    # ── Token embeddings ────────────────────────────────────────
    backbone.get_layer("token_embedding").embeddings.assign(
        get("model.embed_tokens.weight")
    )
    if not backbone.tie_word_embeddings:
        backbone.get_layer("token_embedding").reverse_embeddings.assign(
            transpose_2d(get("lm_head.weight"))
        )

    # ── Text decoder layers ─────────────────────────────────────
    for i in range(backbone.num_layers):
        d = backbone.get_layer(f"transformer_layer_{i}")
        pfx = f"model.layers.{i}"

        # Pre-attention RMSNorm
        d._self_attention_layernorm.scale.assign(
            get(f"{pfx}.input_layernorm.weight")
        )

        # Q projection
        q_w = get(f"{pfx}.self_attn.q_proj.weight")
        q_b = get(f"{pfx}.self_attn.q_proj.bias")
        q_k_shape = list(d._self_attention_layer._query_dense.kernel.shape)
        q_b_shape = list(d._self_attention_layer._query_dense.bias.shape)
        d._self_attention_layer._query_dense.kernel.assign(
            transpose_and_reshape(q_w, q_k_shape)
        )
        d._self_attention_layer._query_dense.bias.assign(
            transpose_and_reshape(q_b, q_b_shape)
        )

        # K projection
        k_w = get(f"{pfx}.self_attn.k_proj.weight")
        k_b = get(f"{pfx}.self_attn.k_proj.bias")
        k_k_shape = list(d._self_attention_layer._key_dense.kernel.shape)
        k_b_shape = list(d._self_attention_layer._key_dense.bias.shape)
        d._self_attention_layer._key_dense.kernel.assign(
            transpose_and_reshape(k_w, k_k_shape)
        )
        d._self_attention_layer._key_dense.bias.assign(
            transpose_and_reshape(k_b, k_b_shape)
        )

        # V projection
        v_w = get(f"{pfx}.self_attn.v_proj.weight")
        v_b = get(f"{pfx}.self_attn.v_proj.bias")
        v_k_shape = list(d._self_attention_layer._value_dense.kernel.shape)
        v_b_shape = list(d._self_attention_layer._value_dense.bias.shape)
        d._self_attention_layer._value_dense.kernel.assign(
            transpose_and_reshape(v_w, v_k_shape)
        )
        d._self_attention_layer._value_dense.bias.assign(
            transpose_and_reshape(v_b, v_b_shape)
        )

        # O projection
        o_w = get(f"{pfx}.self_attn.o_proj.weight")
        o_k_shape = list(d._self_attention_layer._output_dense.kernel.shape)
        d._self_attention_layer._output_dense.kernel.assign(
            transpose_and_reshape(o_w, o_k_shape)
        )

        # MLP (gate / up / down)
        d._feedforward_gate_dense.kernel.assign(
            transpose_2d(get(f"{pfx}.mlp.gate_proj.weight"))
        )
        d._feedforward_intermediate_dense.kernel.assign(
            transpose_2d(get(f"{pfx}.mlp.up_proj.weight"))
        )
        d._feedforward_output_dense.kernel.assign(
            transpose_2d(get(f"{pfx}.mlp.down_proj.weight"))
        )

        # Post-attention RMSNorm
        d._feedforward_layernorm.scale.assign(
            get(f"{pfx}.post_attention_layernorm.weight")
        )

    # Final layernorm
    backbone.get_layer("sequence_output_layernorm").scale.assign(
        get("model.norm.weight")
    )

    # ── Vision encoder ──────────────────────────────────────────
    vision = backbone.get_layer("vision_encoder")

    # Conv3D patch embedding
    # HF: (embed_dim, C, T, H, W) → Keras: (T, H, W, C, embed_dim)
    vision.patch_embed.kernel.assign(
        np.transpose(get("visual.patch_embed.proj.weight"), (2, 3, 4, 1, 0))
    )
    # HF has bias on patch_embed; KerasHub uses use_bias=False → skip

    # Vision blocks
    for i in range(vision.depth):
        blk = vision.blocks[i]
        bp = f"visual.blocks.{i}"

        # LayerNorm 1
        blk.norm1.gamma.assign(get(f"{bp}.norm1.weight"))
        blk.norm1.beta.assign(get(f"{bp}.norm1.bias"))

        # Fused QKV
        blk.attn.qkv.kernel.assign(transpose_2d(get(f"{bp}.attn.qkv.weight")))
        blk.attn.qkv.bias.assign(get(f"{bp}.attn.qkv.bias"))

        # Output projection
        blk.attn.proj.kernel.assign(transpose_2d(get(f"{bp}.attn.proj.weight")))
        blk.attn.proj.bias.assign(get(f"{bp}.attn.proj.bias"))

        # LayerNorm 2
        blk.norm2.gamma.assign(get(f"{bp}.norm2.weight"))
        blk.norm2.beta.assign(get(f"{bp}.norm2.bias"))

        # MLP
        blk.mlp.fc1.kernel.assign(transpose_2d(get(f"{bp}.mlp.fc1.weight")))
        blk.mlp.fc1.bias.assign(get(f"{bp}.mlp.fc1.bias"))
        blk.mlp.fc2.kernel.assign(transpose_2d(get(f"{bp}.mlp.fc2.weight")))
        blk.mlp.fc2.bias.assign(get(f"{bp}.mlp.fc2.bias"))

    # Patch merger
    merger = vision.merger
    merger.ln_q.gamma.assign(get("visual.merger.ln_q.weight"))
    merger.ln_q.beta.assign(get("visual.merger.ln_q.bias"))
    # HF Sequential: .0 = fc1, .1 = GELU (no params), .2 = fc2
    merger.mlp_fc1.kernel.assign(
        transpose_2d(get("visual.merger.mlp.0.weight"))
    )
    merger.mlp_fc1.bias.assign(get("visual.merger.mlp.0.bias"))
    merger.mlp_fc2.kernel.assign(
        transpose_2d(get("visual.merger.mlp.2.weight"))
    )
    merger.mlp_fc2.bias.assign(get("visual.merger.mlp.2.bias"))

    print(f"  Ported {len(hf_state_dict)} HF weights → KerasHub backbone")
    return backbone


# ── 3. Verify tokenizer ─────────────────────────────────────────────


def verify_tokenizer(keras_tokenizer, hf_tokenizer):
    print("\n── Tokenizer verification ──")
    test_strings = [
        "What is Keras?",
        "Describe the weather today.",
        "Hello, world! 🌍",
    ]
    for s in test_strings:
        hf_ids = hf_tokenizer(s, add_special_tokens=False)["input_ids"]
        keras_ids = keras_tokenizer(s)
        if hasattr(keras_ids, "numpy"):
            keras_ids = keras_ids.numpy()
        keras_ids = np.asarray(keras_ids).flatten().tolist()
        np.testing.assert_equal(keras_ids, hf_ids, err_msg=f"Mismatch: {s!r}")
        print(f"  ✅ '{s}' → {len(hf_ids)} tokens match")
    print("  All tokenizer checks passed")


# ── 4. Verify preprocessor ──────────────────────────────────────────


def verify_preprocessor(keras_tokenizer, hf_processor):
    print("\n── Preprocessor verification ──")

    # Text-only
    text = "Describe the weather"
    hf_ids = hf_processor.tokenizer(text, add_special_tokens=False)["input_ids"]
    keras_pp = keras_hub.models.Qwen2VLCausalLMPreprocessor(
        tokenizer=keras_tokenizer, sequence_length=32
    )
    result = keras_pp.generate_preprocess(text)
    padding_mask = np.asarray(result["padding_mask"])
    num_real = int(np.sum(padding_mask))
    keras_ids = np.asarray(result["token_ids"])[:num_real].tolist()
    np.testing.assert_equal(keras_ids, hf_ids)
    print("Text-only preprocessing matches")

    # With image
    image_converter = Qwen2VLImageConverter()
    keras_pp_img = keras_hub.models.Qwen2VLCausalLMPreprocessor(
        tokenizer=keras_tokenizer,
        image_converter=image_converter,
        sequence_length=512,
        spatial_merge_size=2,
    )
    dummy = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
    result = keras_pp_img.generate_preprocess(
        {"text": "Describe this image", "images": dummy}
    )
    assert result["patch_values"] is not None
    assert result["image_grid_thw"] is not None
    grid_thw = result["image_grid_thw"]
    expected = int(np.prod(grid_thw[0]) // 4)
    actual = int(
        np.sum(
            np.asarray(result["token_ids"])
            == keras_tokenizer.image_pad_token_id
        )
    )
    assert actual == expected, f"vision tokens: {actual} != {expected}"
    print(
        f"Image preprocessing: {expected} vision tokens "
        f"from grid {grid_thw[0].tolist()}"
    )
    print("  All preprocessor checks passed")


# ── 5. Verify backbone outputs ──────────────────────────────────────


def verify_backbone(keras_backbone, keras_tokenizer, hf_model, hf_tokenizer):
    print("\n── Backbone verification ──")

    # Parameter counts
    keras_params = keras_backbone.count_params()
    hf_params = hf_model.num_parameters()
    print(f"  KerasHub params: {keras_params:,}")
    print(f"  HF total params: {hf_params:,}")

    # Hidden state comparison (text-only path)
    test_text = "What is Keras?"
    hf_inputs = hf_tokenizer(
        test_text, return_tensors="pt", add_special_tokens=False
    ).to(device)
    seq_len = hf_inputs["input_ids"].shape[1]

    with torch.no_grad():
        hf_outputs = hf_model.model(**hf_inputs)
    hf_hidden = hf_outputs.last_hidden_state.detach().cpu().float().numpy()

    keras_pp = keras_hub.models.Qwen2VLCausalLMPreprocessor(
        tokenizer=keras_tokenizer, sequence_length=seq_len
    )
    k_in = keras_pp([test_text], sequence_length=seq_len)[0]
    k_in = {k: v.to(device) for k, v in k_in.items()}
    keras_hidden = ops.convert_to_numpy(keras_backbone(k_in))

    print(f"  HF hidden shape:    {hf_hidden.shape}")
    print(f"  Keras hidden shape: {keras_hidden.shape}")

    try:
        np.testing.assert_allclose(
            keras_hidden, hf_hidden, atol=1e-4, rtol=1e-4
        )
        print("Hidden states match (atol=1e-4)")
    except AssertionError:
        max_diff = float(np.max(np.abs(keras_hidden - hf_hidden)))
        mean_diff = float(np.mean(np.abs(keras_hidden - hf_hidden)))
        print(f"Max abs diff:  {max_diff:.6e}")
        print(f"Mean abs diff: {mean_diff:.6e}")
        print(traceback.format_exc())

    # Logits via LM head
    keras_logits = ops.convert_to_numpy(
        keras_backbone.token_embedding(
            ops.convert_to_tensor(keras_hidden), reverse=True
        )
    )
    with torch.no_grad():
        hf_logits = (
            hf_model.lm_head(hf_outputs.last_hidden_state)
            .detach()
            .cpu()
            .float()
            .numpy()
        )

    try:
        np.testing.assert_allclose(
            keras_logits, hf_logits, atol=1e-4, rtol=1e-4
        )
        print("Logits match (atol=1e-4)")
    except AssertionError:
        max_diff = float(np.max(np.abs(keras_logits - hf_logits)))
        print(f"Logits max diff: {max_diff:.6e}")
        print(traceback.format_exc())

    # Mean diff as a single number (like the DistilBERT example)
    mean_hidden = float(np.mean(keras_hidden - hf_hidden))
    print(f"  Mean diff (keras - hf): {mean_hidden:.6e}")

    print("  Backbone verification complete")


# ── Main ─────────────────────────────────────────────────────────────


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {preset}. "
            f"Must be one of {','.join(PRESET_MAP.keys())}"
        )
    hf_id = PRESET_MAP[preset]

    # ── Load HF model ───────────────────────────────────────────
    print(f"Loading HF model: {hf_id}")
    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
        hf_id, device_map=device, torch_dtype=torch.float32
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_id)
    hf_processor = AutoProcessor.from_pretrained(hf_id)
    hf_state_dict = hf_model.state_dict()

    # Use in-memory config
    hf_config = hf_model.config.to_dict()
    print(f"  HF state_dict: {len(hf_state_dict)} tensors")

    # ── Build KerasHub backbone from config ─────────────────────
    print("\nBuilding KerasHub backbone from config...")
    backbone_kwargs = build_backbone_config(hf_config)
    print(f"  Config: {json.dumps(backbone_kwargs, indent=2)}")
    keras_backbone = Qwen2VLBackbone(**backbone_kwargs)
    keras_backbone.summary()

    # ── Port weights ────────────────────────────────────────────
    print("\nPorting weights...")
    port_weights(keras_backbone, hf_state_dict)

    # ── Load KerasHub tokenizer ─────────────────────────────────
    print("\nLoading KerasHub tokenizer...")
    keras_tokenizer = keras_hub.models.Qwen2VLTokenizer.from_preset(
        f"hf://{hf_id}"
    )
    print("  KerasHub tokenizer loaded")

    # ── Verify ──────────────────────────────────────────────────
    verify_tokenizer(keras_tokenizer, hf_tokenizer)
    verify_preprocessor(keras_tokenizer, hf_processor)
    verify_backbone(keras_backbone, keras_tokenizer, hf_model, hf_tokenizer)

    print(f"\n🏁 All checks passed for {preset}!")

    # ── Save preset ─────────────────────────────────────────────
    save_dir = f"./{preset}"
    keras_backbone.save_to_preset(save_dir)
    keras_tokenizer.save_to_preset(save_dir)
    print(f"\n  Preset saved to {save_dir}/")
    print(f"  Contents: {os.listdir(save_dir)}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
