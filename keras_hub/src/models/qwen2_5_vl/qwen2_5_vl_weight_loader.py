"""
Qwen2.5-VL Weight Loader

Loads pretrained weights from a HuggingFace safetensors checkpoint
into a Qwen2_5_VLCausalLM Keras model.

Usage
-----
from qwen2_5_vl_causal_lm import Qwen2_5_VLCausalLM
from qwen2_5_vl_weight_loader import load_weights_from_hf

model = Qwen2_5_VLCausalLM.from_preset("qwen2_5_vl_7b")
load_weights_from_hf(
    model,
    hf_repo="Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir="/content/qwen_weights",
)
"""

import os
import json
import numpy as np


# ── Safetensors loader ─────────────────────────────────────────────────────────

def _load_safetensors(path):
    """
    Load a single safetensors file into a dict of {name: np.ndarray}.
    Uses the safetensors library which is available in all HF environments.
    """
    from safetensors import safe_open
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).float().numpy()
    return tensors


def _load_all_shards(index_path, shard_dir):
    """
    Load all safetensors shards listed in model.safetensors.index.json.
    Returns a single merged dict of {param_name: np.ndarray}.
    """
    with open(index_path) as f:
        index = json.load(f)

    # Collect unique shard filenames
    shard_files = sorted(set(index["weight_map"].values()))
    all_weights = {}
    for shard_file in shard_files:
        shard_path = os.path.join(shard_dir, shard_file)
        print(f"  Loading shard: {shard_file}")
        all_weights.update(_load_safetensors(shard_path))

    return all_weights


# ── HF → Keras variable mapping ────────────────────────────────────────────────

def _build_mapping(model):
    """
    Build a dict mapping HuggingFace parameter names to Keras variables.

    HF name                                      → Keras variable
    ─────────────────────────────────────────────────────────────
    Text decoder
    model.embed_tokens.weight                    → token_embedding.embeddings
    model.norm.weight                            → decoder_stack.final_norm.weight
    model.layers.{i}.input_layernorm.weight      → decoder_stack.layers_list[i].input_norm.weight
    model.layers.{i}.post_attention_layernorm.weight → decoder_stack.layers_list[i].post_norm.weight
    model.layers.{i}.self_attn.q_proj.weight     → decoder_stack.layers_list[i].self_attn.q_proj.kernel
    model.layers.{i}.self_attn.q_proj.bias       → decoder_stack.layers_list[i].self_attn.q_proj.bias
    model.layers.{i}.self_attn.k_proj.weight     → decoder_stack.layers_list[i].self_attn.k_proj.kernel
    model.layers.{i}.self_attn.k_proj.bias       → decoder_stack.layers_list[i].self_attn.k_proj.bias
    model.layers.{i}.self_attn.v_proj.weight     → decoder_stack.layers_list[i].self_attn.v_proj.kernel
    model.layers.{i}.self_attn.v_proj.bias       → decoder_stack.layers_list[i].self_attn.v_proj.bias
    model.layers.{i}.self_attn.o_proj.weight     → decoder_stack.layers_list[i].self_attn.o_proj.kernel
    model.layers.{i}.mlp.gate_proj.weight        → decoder_stack.layers_list[i].mlp.gate_proj.kernel
    model.layers.{i}.mlp.up_proj.weight          → decoder_stack.layers_list[i].mlp.up_proj.kernel
    model.layers.{i}.mlp.down_proj.weight        → decoder_stack.layers_list[i].mlp.down_proj.kernel

    Vision encoder
    visual.patch_embed.proj.weight               → vision_encoder.patch_embed.kernel
    visual.blocks.{i}.norm1.weight               → vision_encoder.blocks[i].norm1.weight
    visual.blocks.{i}.norm2.weight               → vision_encoder.blocks[i].norm2.weight
    visual.blocks.{i}.attn.qkv.weight            → vision_encoder.blocks[i].attn.qkv.kernel
    visual.blocks.{i}.attn.qkv.bias              → vision_encoder.blocks[i].attn.qkv.bias
    visual.blocks.{i}.attn.proj.weight           → vision_encoder.blocks[i].attn.proj.kernel
    visual.blocks.{i}.attn.proj.bias             → vision_encoder.blocks[i].attn.proj.bias
    visual.blocks.{i}.mlp.gate_proj.weight/bias  → vision_encoder.blocks[i].mlp.gate_proj.kernel/bias
    visual.blocks.{i}.mlp.up_proj.weight/bias    → vision_encoder.blocks[i].mlp.up_proj.kernel/bias
    visual.blocks.{i}.mlp.down_proj.weight/bias  → vision_encoder.blocks[i].mlp.down_proj.kernel/bias
    visual.merger.ln_q.weight                    → (skipped — PatchMerger has no weights)
    visual.merger.mlp.0.weight/bias              → vision_projector.dense1.kernel/bias
    visual.merger.mlp.2.weight/bias              → vision_projector.dense2.kernel/bias

    lm_head.weight is tied to embed_tokens.weight — skipped.
    """
    backbone = model.backbone
    mapping  = {}

    # ── Token embedding ────────────────────────────────────────────────────
    mapping["model.embed_tokens.weight"] = (
        backbone.token_embedding.embeddings, False
    )

    # ── Decoder final norm ─────────────────────────────────────────────────
    mapping["model.norm.weight"] = (
        backbone.decoder_stack.final_norm.weight, False
    )

    # ── Decoder layers ─────────────────────────────────────────────────────
    for i, block in enumerate(backbone.decoder_stack.layers_list):
        p = f"model.layers.{i}"
        k = block  # shorthand

        mapping[f"{p}.input_layernorm.weight"] = (
            k.input_norm.weight, False
        )
        mapping[f"{p}.post_attention_layernorm.weight"] = (
            k.post_norm.weight, False
        )

        # Attention projections — Linear weights are transposed in HF
        # (HF stores [out_features, in_features], Keras expects [in, out])
        mapping[f"{p}.self_attn.q_proj.weight"] = (
            k.self_attn.q_proj.kernel, True   # needs transpose
        )
        mapping[f"{p}.self_attn.q_proj.bias"] = (
            k.self_attn.q_proj.bias, False
        )
        mapping[f"{p}.self_attn.k_proj.weight"] = (
            k.self_attn.k_proj.kernel, True
        )
        mapping[f"{p}.self_attn.k_proj.bias"] = (
            k.self_attn.k_proj.bias, False
        )
        mapping[f"{p}.self_attn.v_proj.weight"] = (
            k.self_attn.v_proj.kernel, True
        )
        mapping[f"{p}.self_attn.v_proj.bias"] = (
            k.self_attn.v_proj.bias, False
        )
        mapping[f"{p}.self_attn.o_proj.weight"] = (
            k.self_attn.o_proj.kernel, True
        )

        # MLP projections
        mapping[f"{p}.mlp.gate_proj.weight"] = (
            k.mlp.gate_proj.kernel, True
        )
        mapping[f"{p}.mlp.up_proj.weight"] = (
            k.mlp.up_proj.kernel, True
        )
        mapping[f"{p}.mlp.down_proj.weight"] = (
            k.mlp.down_proj.kernel, True
        )

    # ── Vision patch embedding ─────────────────────────────────────────────
    # Conv3D kernel: HF stores (out_ch, in_ch, kT, kH, kW)
    # Keras expects (kT, kH, kW, in_ch, out_ch)
    mapping["visual.patch_embed.proj.weight"] = (
        backbone.vision_encoder.patch_embed.kernel, "conv3d"
    )

    # ── Vision blocks ──────────────────────────────────────────────────────
    for i, block in enumerate(backbone.vision_encoder.blocks):
        p = f"visual.blocks.{i}"

        mapping[f"{p}.norm1.weight"] = (block.norm1.weight, False)
        mapping[f"{p}.norm2.weight"] = (block.norm2.weight, False)

        # QKV is a single fused Dense in both WindowAttention and GlobalAttention
        mapping[f"{p}.attn.qkv.weight"] = (block.attn.qkv.kernel, True)
        mapping[f"{p}.attn.qkv.bias"]   = (block.attn.qkv.bias,   False)
        mapping[f"{p}.attn.proj.weight"] = (block.attn.proj.kernel, True)
        mapping[f"{p}.attn.proj.bias"]   = (block.attn.proj.bias,   False)

        # Vision MLP has bias (unlike decoder MLP)
        mapping[f"{p}.mlp.gate_proj.weight"] = (block.mlp.gate_proj.kernel, True)
        mapping[f"{p}.mlp.gate_proj.bias"]   = (block.mlp.gate_proj.bias,   False)
        mapping[f"{p}.mlp.up_proj.weight"]   = (block.mlp.up_proj.kernel,   True)
        mapping[f"{p}.mlp.up_proj.bias"]     = (block.mlp.up_proj.bias,     False)
        mapping[f"{p}.mlp.down_proj.weight"] = (block.mlp.down_proj.kernel, True)
        mapping[f"{p}.mlp.down_proj.bias"]   = (block.mlp.down_proj.bias,   False)

    # ── Vision projector (merger MLP) ──────────────────────────────────────
    # visual.merger.ln_q.weight → PatchMerger has no weights, skip
    mapping["visual.merger.mlp.0.weight"] = (
        backbone.vision_projector.dense1.kernel, True
    )
    mapping["visual.merger.mlp.0.bias"] = (
        backbone.vision_projector.dense1.bias, False
    )
    mapping["visual.merger.mlp.2.weight"] = (
        backbone.vision_projector.dense2.kernel, True
    )
    mapping["visual.merger.mlp.2.bias"] = (
        backbone.vision_projector.dense2.bias, False
    )

    return mapping


# ── Weight assignment ──────────────────────────────────────────────────────────

def _assign_weight(keras_var, hf_array, transform):
    """
    Assign hf_array to keras_var, applying the required shape transform.

    transform:
        False    — assign as-is (RMSNorm weights, biases, embeddings)
        True     — transpose last two dims (Linear: [out, in] → [in, out])
        "conv3d" — permute Conv3D kernel (out, in, kT, kH, kW) → (kT, kH, kW, in, out)
    """
    if transform == "conv3d":
        # HF: (out_channels, in_channels, kT, kH, kW)
        # Keras: (kT, kH, kW, in_channels, out_channels)
        hf_array = np.transpose(hf_array, (2, 3, 4, 1, 0))
    elif transform is True:
        # HF Linear: (out_features, in_features)
        # Keras Dense kernel: (in_features, out_features)
        hf_array = hf_array.T

    expected = tuple(keras_var.shape)
    actual   = tuple(hf_array.shape)
    if expected != actual:
        raise ValueError(
            f"Shape mismatch for variable '{keras_var.path}':\n"
            f"  Keras expects: {expected}\n"
            f"  HF provides:   {actual}"
        )

    keras_var.assign(hf_array.astype(keras_var.dtype))


# ── Public API ─────────────────────────────────────────────────────────────────

def load_weights_from_hf(
    model,
    hf_repo="Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir="/content/qwen_weights",
    verbose=True,
):
    """
    Download and load pretrained HuggingFace weights into a Keras model.

    The model must have been built (i.e. called at least once with dummy
    inputs) before calling this function so all Keras variables exist.

    Parameters
    ----------
    model : Qwen2_5_VLCausalLM
        A built Keras model instance.
    hf_repo : str
        HuggingFace repo id, e.g. "Qwen/Qwen2.5-VL-7B-Instruct".
    cache_dir : str
        Local directory to cache downloaded weight files.
    verbose : bool
        If True, prints progress messages.

    Example
    -------
    model = Qwen2_5_VLCausalLM.from_preset("qwen2_5_vl_7b")

    # Build the model with a dummy forward pass
    import keras, numpy as np
    model({"token_ids": keras.ops.zeros((1, 4), dtype="int32")})

    load_weights_from_hf(model, hf_repo="Qwen/Qwen2.5-VL-7B-Instruct")
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    os.makedirs(cache_dir, exist_ok=True)

    # ── Step 1: Download index file ────────────────────────────────────────
    if verbose:
        print(f"Fetching weight index from {hf_repo} ...")

    index_path = hf_hub_download(
        repo_id=hf_repo,
        filename="model.safetensors.index.json",
        local_dir=cache_dir,
    )

    with open(index_path) as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))

    # ── Step 2: Download all shards ────────────────────────────────────────
    if verbose:
        print(f"Downloading {len(shard_files)} weight shard(s) ...")

    for shard_file in shard_files:
        hf_hub_download(
            repo_id=hf_repo,
            filename=shard_file,
            local_dir=cache_dir,
        )

    # ── Step 3: Load all shards into memory ───────────────────────────────
    if verbose:
        print("Loading weights into memory ...")

    all_weights = _load_all_shards(index_path, cache_dir)

    # ── Step 4: Build mapping and assign weights ───────────────────────────
    if verbose:
        print("Building HF → Keras weight mapping ...")

    mapping = _build_mapping(model)

    skipped  = []
    assigned = 0
    errors   = []

    # Weights in HF but not in our mapping
    known_skips = {
        "lm_head.weight",          # tied to embed_tokens
        "visual.merger.ln_q.weight",  # PatchMerger has no weights
    }

    for hf_name, hf_array in all_weights.items():
        if hf_name in known_skips:
            skipped.append(hf_name)
            continue

        if hf_name not in mapping:
            skipped.append(hf_name)
            if verbose:
                print(f"  [WARN] No mapping for HF param: {hf_name}")
            continue

        keras_var, transform = mapping[hf_name]
        try:
            _assign_weight(keras_var, hf_array, transform)
            assigned += 1
            if verbose:
                print(f"  ✓ {hf_name}")
        except Exception as e:
            errors.append((hf_name, str(e)))
            print(f"  ✗ {hf_name}: {e}")

    # ── Step 5: Report ─────────────────────────────────────────────────────
    if verbose:
        print(f"\nWeight loading complete:")
        print(f"  Assigned : {assigned}")
        print(f"  Skipped  : {len(skipped)}")
        print(f"  Errors   : {len(errors)}")

    if errors:
        raise RuntimeError(
            f"{len(errors)} weight(s) failed to load:\n" +
            "\n".join(f"  {name}: {msg}" for name, msg in errors)
        )

    return assigned


def load_weights_from_local(
    model,
    checkpoint_dir,
    verbose=True,
):
    """
    Load weights from a locally downloaded HuggingFace checkpoint directory.

    Loads and assigns one shard at a time to minimise peak memory usage.
    Safe to use on 16GB GPUs with 3B/7B models.

    Parameters
    ----------
    model : Qwen2_5_VLCausalLM
    checkpoint_dir : str
        Path to a directory containing model.safetensors.index.json
        and all shard files.
    verbose : bool
    """
    import gc
    known_skips = {"lm_head.weight", "visual.merger.ln_q.weight"}
    mapping     = _build_mapping(model)
    assigned    = 0
    errors      = []

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")

    if not os.path.exists(index_path):
        single = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(single):
            shard_files = [os.path.basename(single)]
            shard_dir   = checkpoint_dir
            index       = {"weight_map": {}}
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}. "
                "Expected model.safetensors.index.json or model.safetensors."
            )
    else:
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        shard_dir   = checkpoint_dir

    if verbose:
        print(f"Loading {len(shard_files)} shard(s) from {checkpoint_dir} ...")

    for shard_file in shard_files:
        shard_path = os.path.join(shard_dir, shard_file)
        if verbose:
            print(f"  Shard: {shard_file}")

        # Load one shard into CPU RAM
        shard_weights = _load_safetensors(shard_path)

        # Assign each weight immediately then discard
        for hf_name, hf_array in shard_weights.items():
            if hf_name in known_skips:
                continue
            if hf_name not in mapping:
                if verbose:
                    print(f"    [WARN] No mapping for: {hf_name}")
                continue
            keras_var, transform = mapping[hf_name]
            try:
                _assign_weight(keras_var, hf_array, transform)
                assigned += 1
            except Exception as e:
                errors.append((hf_name, str(e)))
                print(f"    ✗ {hf_name}: {e}")

        # Free shard from CPU RAM before loading next one
        del shard_weights
        gc.collect()

        if verbose:
            print(f"    ✓ shard assigned ({assigned} weights so far)")

    if verbose:
        print(f"\nWeight loading complete: {assigned} assigned, {len(errors)} errors")

    if errors:
        raise RuntimeError(
            f"{len(errors)} weight(s) failed:\n" +
            "\n".join(f"  {n}: {m}" for n, m in errors)
        )

    return assigned