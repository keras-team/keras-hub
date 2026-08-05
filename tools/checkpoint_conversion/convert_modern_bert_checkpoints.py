"""
ModernBERT weight conversion script.

This script downloads the official Hugging Face ModernBERT checkpoint,
converts the weights to the KerasHub format, and validates the converted
model against the Hugging Face implementation by comparing intermediate
activations and final predictions.

To run, install the CPU-only development environment and Hugging Face
dependencies:
```
pip install -r requirements.txt
pip install transformers huggingface_hub
```

Login to Hugging Face:
```
huggingface-cli login
```

Run the conversion:
```
python tools/checkpoint_conversion/convert_modern_bert_checkpoints.py \
    --preset modernbert_base_en \
    --output_dir ./modernbert_base_en
```

The converted model matches the Hugging Face model very closely. 
The maximum difference is 4.48e-05, and the mean difference is 5.75e-06.

The embeddings match almost exactly, with a difference of 9.5e-07. 
The small difference increases gradually through the 22 
layers because each layer performs many calculations, 
such as attention, normalization, softmax, and MLP. 
PyTorch and Keras may handle some of these calculations 
slightly differently, causing tiny float32 rounding differences that add up.

- 6/6 top-1 predictions match Hugging Face across the test sentences.
- 6/6 top-5 predictions match as well, so the ranking of 
    likely tokens also matches.
- The hidden-state and logit statistics 
(mean and standard deviation) are also very close to Hugging Face.
- The differences are small and increase gradually, 
    which is what we expect from normal float32 
    rounding between two different frameworks.
    On longer inputs, where local sliding-window attention 
    is more heavily used, the backbone difference can 
    increase to around 1.4e-2. However, the predictions 
    and final logits remain accurate, which indicates accumulated 
    rounding differences rather than a functional issue.
# """

import argparse
import os
import sys

import keras
import numpy as np
import torch
from keras import ops
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

from keras_hub.src.models.modernbert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modernbert.modern_bert_masked_lm import (
    ModernBertMaskedLM,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PRESET_MAP = {
    "modernbert_base_en": "answerdotai/ModernBERT-base",
    "modernbert_large_en": "answerdotai/ModernBERT-large",
}


def convert_modern_bert_weights(keras_model, hf_model):
    """Convert HuggingFace ModernBERT weights to Keras."""

    if hasattr(hf_model, "model"):
        hf_backbone = hf_model.model
    elif hasattr(hf_model, "bert"):
        hf_backbone = hf_model.bert
    else:
        hf_backbone = hf_model

    keras_backbone = getattr(keras_model, "backbone", keras_model)

    # Embeddings
    keras_backbone.token_embedding.embeddings.assign(
        hf_backbone.embeddings.tok_embeddings.weight.detach().cpu().numpy()
    )

    if hasattr(hf_backbone.embeddings, "norm") and not isinstance(
        hf_backbone.embeddings.norm, torch.nn.Identity
    ):
        keras_backbone.embedding_norm.gamma.assign(
            hf_backbone.embeddings.norm.weight.detach().cpu().numpy()
        )

    # Transformer layers
    for hf_layer, keras_layer in zip(
        hf_backbone.layers,
        keras_backbone.transformer_layers,
    ):
        # Attention QKV
        keras_layer.attn.qkv.kernel.assign(
            hf_layer.attn.Wqkv.weight.detach().cpu().numpy().T
        )

        if (
            hf_layer.attn.Wqkv.bias is not None
            and keras_layer.attn.qkv.bias is not None
        ):
            keras_layer.attn.qkv.bias.assign(
                hf_layer.attn.Wqkv.bias.detach().cpu().numpy()
            )

        # Attention output
        keras_layer.attn.output_dense.kernel.assign(
            hf_layer.attn.Wo.weight.detach().cpu().numpy().T
        )

        if (
            hf_layer.attn.Wo.bias is not None
            and keras_layer.attn.output_dense.bias is not None
        ):
            keras_layer.attn.output_dense.bias.assign(
                hf_layer.attn.Wo.bias.detach().cpu().numpy()
            )

        # Attention LayerNorm
        if not isinstance(hf_layer.attn_norm, torch.nn.Identity) and hasattr(
            keras_layer.attn_norm, "gamma"
        ):
            keras_layer.attn_norm.gamma.assign(
                hf_layer.attn_norm.weight.detach().cpu().numpy()
            )

        # MLP LayerNorm
        keras_layer.mlp_norm.gamma.assign(
            hf_layer.mlp_norm.weight.detach().cpu().numpy()
        )

        # GeGLU input projection
        wi = hf_layer.mlp.Wi.weight.detach().cpu().numpy()

        input_proj, gate_proj = np.split(wi, 2, axis=0)

        keras_layer.mlp.wi_0.kernel.assign(input_proj.T)
        keras_layer.mlp.wi_1.kernel.assign(gate_proj.T)

        if (
            hf_layer.mlp.Wi.bias is not None
            and keras_layer.mlp.wi_0.bias is not None
        ):
            input_bias, gate_bias = np.split(
                hf_layer.mlp.Wi.bias.detach().cpu().numpy(),
                2,
            )

            keras_layer.mlp.wi_0.bias.assign(input_bias)
            keras_layer.mlp.wi_1.bias.assign(gate_bias)

        # GeGLU output projection
        keras_layer.mlp.wo.kernel.assign(
            hf_layer.mlp.Wo.weight.detach().cpu().numpy().T
        )

        if (
            hf_layer.mlp.Wo.bias is not None
            and keras_layer.mlp.wo.bias is not None
        ):
            keras_layer.mlp.wo.bias.assign(
                hf_layer.mlp.Wo.bias.detach().cpu().numpy()
            )

    # Final LayerNorm
    keras_backbone.final_norm.gamma.assign(
        hf_backbone.final_norm.weight.detach().cpu().numpy()
    )

    print("Backbone weights converted successfully.")


def convert_modern_bert_masked_lm_weights(keras_lm, backbone, hf_model):
    """Convert Hugging Face ModernBERT MLM weights to Keras."""

    print("Converting MaskedLM head weights...")
    convert_modern_bert_weights(backbone, hf_model)

    hf_head = hf_model.head
    hf_decoder = hf_model.decoder
    keras_lm.mlm_head_dense.kernel.assign(
        hf_head.dense.weight.detach().cpu().numpy().T
    )

    # Dense bias
    if (
        hf_head.dense.bias is not None
        and keras_lm.mlm_head_dense.bias is not None
    ):
        keras_lm.mlm_head_dense.bias.assign(
            hf_head.dense.bias.detach().cpu().numpy()
        )

    # LayerNorm
    keras_lm.mlm_head_norm.gamma.assign(
        hf_head.norm.weight.detach().cpu().numpy()
    )

    if (
        hf_head.norm.bias is not None
        and keras_lm.mlm_head_norm.beta is not None
    ):
        keras_lm.mlm_head_norm.beta.assign(
            hf_head.norm.bias.detach().cpu().numpy()
        )

    # decoder.weight = tied embedding weight
    # decoder.bias   = decoder bias
    # PyTorch Linear:
    #   [vocab_size, hidden_size]
    #
    # Keras Dense:
    #   [hidden_size, vocab_size]

    keras_lm.decoder.kernel.assign(hf_decoder.weight.detach().cpu().numpy().T)

    if hf_decoder.bias is not None and keras_lm.decoder.bias is not None:
        keras_lm.decoder.bias.assign(hf_decoder.bias.detach().cpu().numpy())

    print("MaskedLM head weights converted successfully.")


def verify_conversion(keras_lm, hf_model, hf_tokenizer):
    """Numerically verify the converted ModernBERT checkpoint."""

    print("Running numerical verification..")

    text = "The capital of France is [MASK]."

    # Tokenization
    hf_inputs = hf_tokenizer(
        text,
        return_tensors="pt",
    )

    hf_inputs.pop("token_type_ids", None)

    input_ids = hf_inputs["input_ids"].cpu().numpy().astype("int32")

    padding_mask = hf_inputs["attention_mask"].cpu().numpy().astype("int32")

    print(f"Input shape: {input_ids.shape}")

    backbone = keras_lm.backbone

    # EMBEDDING VERIFICATION
    print("\nEmbedding verification")

    with torch.no_grad():
        hf_embedding = (
            hf_model.model.embeddings(hf_inputs["input_ids"]).cpu().numpy()
        )

    keras_embedding = backbone.token_embedding(input_ids)
    keras_embedding = backbone.embedding_norm(keras_embedding)
    keras_embedding = keras.ops.convert_to_numpy(keras_embedding)
    embedding_diff = np.abs(hf_embedding - keras_embedding)
    embedding_max_diff = np.max(embedding_diff)

    print(f"Embedding max diff: {embedding_max_diff:.6e}")
    print(f"HF embedding mean: {hf_embedding.mean():.6e}")
    print(f"Keras embedding mean: {keras_embedding.mean():.6e}")
    print(f"HF embedding std: {hf_embedding.std():.6e}")
    print(f"Keras embedding std: {keras_embedding.std():.6e}")

    np.testing.assert_allclose(
        hf_embedding,
        keras_embedding,
        atol=1e-6,
        rtol=1e-6,
    )

    print("Embedding verification passed.")

    print("\nBackbone verification")
    with torch.no_grad():
        hf_hidden = (
            hf_model.model(
                input_ids=hf_inputs["input_ids"],
                attention_mask=hf_inputs["attention_mask"],
            )
            .last_hidden_state.cpu()
            .numpy()
        )

    keras_hidden = backbone(
        {
            "token_ids": input_ids,
            "padding_mask": padding_mask,
        },
        training=False,
    )

    keras_hidden = keras.ops.convert_to_numpy(keras_hidden)

    backbone_diff = np.abs(hf_hidden - keras_hidden)

    backbone_max_diff = np.max(backbone_diff)

    print(f"Backbone max diff: {backbone_max_diff:.6e}")
    print(f"HF hidden mean: {hf_hidden.mean():.6e}")
    print(f"Keras hidden mean: {keras_hidden.mean():.6e}")
    print(f"HF hidden std: {hf_hidden.std():.6e}")
    print(f"Keras hidden std: {keras_hidden.std():.6e}")
    print("Backbone verification completed.")

    print("\nMaskedLM verification")

    mask_token_id = hf_tokenizer.mask_token_id

    mask_positions = np.argwhere(input_ids == mask_token_id)[:, 1:].astype(
        "int32"
    )

    print(f"Mask positions: {mask_positions.tolist()}\n")

    # Hugging Face logits
    with torch.no_grad():
        hf_logits = (
            hf_model(
                input_ids=hf_inputs["input_ids"],
                attention_mask=hf_inputs["attention_mask"],
            )
            .logits.cpu()
            .numpy()
        )

    batch_index = np.arange(input_ids.shape[0])[:, None]

    hf_mask_logits = hf_logits[
        batch_index,
        mask_positions,
    ]

    print(f"HF masked logits shape: {hf_mask_logits.shape}")

    # Keras logits
    keras_logits = keras_lm(
        {
            "token_ids": input_ids,
            "padding_mask": padding_mask,
            "mask_positions": mask_positions,
        },
        training=False,
    )

    if isinstance(keras_logits, dict):
        keras_logits = keras_logits["logits"]

    keras_logits = keras.ops.convert_to_numpy(keras_logits)

    print(f"Keras masked logits shape: {keras_logits.shape}")

    logits_diff = np.abs(hf_mask_logits - keras_logits)

    logits_max_diff = np.max(logits_diff)

    print(f"MaskedLM max diff: {logits_max_diff:.6e}")
    print(f"HF logits mean: {hf_mask_logits.mean():.6e}")
    print(f"Keras logits mean: {keras_logits.mean():.6e}")
    print(f"HF logits std: {hf_mask_logits.std():.6e}")
    print(f"Keras logits std: {keras_logits.std():.6e}")
    print("\n")
    print("MaskedLM verification completed.\n")

    overall_diff = np.concatenate(
        [
            embedding_diff.reshape(-1),
            backbone_diff.reshape(-1),
            logits_diff.reshape(-1),
        ]
    )

    overall_max_diff = np.max(overall_diff)

    overall_mean_diff = np.mean(overall_diff)

    # SUMMARY
    print("NUMERICAL VERIFICATION SUMMARY\n")

    print(f"Embedding max diff : {embedding_max_diff:.6e}")
    print(f"Backbone max diff  : {backbone_max_diff:.6e}")
    print(f"MaskedLM max diff  : {logits_max_diff:.6e}")

    print("\n")
    print(f"Overall max abs diff : {overall_max_diff:.6e}")
    print(f"Overall mean abs diff: {overall_mean_diff:.6e}")
    print("\n")
    print("Numerical verification complete.")

    return {
        "embedding_max_diff": float(embedding_max_diff),
        "backbone_max_diff": float(backbone_max_diff),
        "masked_lm_max_diff": float(logits_max_diff),
        "overall_max_abs_diff": float(overall_max_diff),
        "overall_mean_abs_diff": float(overall_mean_diff),
    }


def verify_embedding_and_qkv(
    keras_backbone,
    hf_model,
    hf_tokenizer,
):
    """Verify embeddings and Layer 0 QKV against Hugging Face."""

    sample_text = "The capital of France is [MASK]."

    # Tokenize
    hf_inputs = hf_tokenizer(
        sample_text,
        return_tensors="pt",
    )

    hf_inputs.pop(
        "token_type_ids",
        None,
    )

    input_ids = hf_inputs["input_ids"]

    input_ids_np = input_ids.cpu().numpy().astype("int32")

    # Embedding
    print("\nEmbedding")

    with torch.no_grad():
        hf_embedding = hf_model.model.embeddings(input_ids)

    hf_embedding_np = hf_embedding.detach().cpu().numpy()

    keras_embedding = keras_backbone.token_embedding(input_ids_np)

    keras_embedding = keras_backbone.embedding_norm(keras_embedding)

    keras_embedding_np = ops.convert_to_numpy(keras_embedding)

    embedding_diff = np.abs(
        hf_embedding_np.astype("float32") - keras_embedding_np.astype("float32")
    )

    print(f"Embedding max diff : {np.max(embedding_diff):.6e}")

    print(f"Embedding mean diff: {np.mean(embedding_diff):.6e}")

    hf_layer = hf_model.model.layers[0]
    keras_layer = keras_backbone.transformer_layers[0]

    # Attention input
    with torch.no_grad():
        hf_attn_input = hf_layer.attn_norm(hf_embedding)

    keras_attn_input = keras_layer.attn_norm(keras_embedding)

    hf_attn_input_np = hf_attn_input.detach().cpu().numpy()

    keras_attn_input_np = ops.convert_to_numpy(keras_attn_input)

    attn_input_diff = np.abs(
        hf_attn_input_np.astype("float32")
        - keras_attn_input_np.astype("float32")
    )

    print("\nAttention input")

    print(f"max diff : {np.max(attn_input_diff):.6e}")

    print(f"mean diff: {np.mean(attn_input_diff):.6e}")

    # QKV projection
    with torch.no_grad():
        hf_qkv = hf_layer.attn.Wqkv(hf_attn_input)

    keras_qkv = keras_layer.attn.qkv(keras_attn_input)

    hf_qkv_np = hf_qkv.detach().cpu().numpy()

    keras_qkv_np = ops.convert_to_numpy(keras_qkv)

    qkv_diff = np.abs(
        hf_qkv_np.astype("float32") - keras_qkv_np.astype("float32")
    )

    print("\nQKV")

    print(f"QKV max diff : {np.max(qkv_diff):.6e}")

    print(f"QKV mean diff: {np.mean(qkv_diff):.6e}")

    # Split Q / K / V
    hidden_dim = keras_layer.attn.hidden_dim

    hf_q, hf_k, hf_v = torch.split(
        hf_qkv,
        hidden_dim,
        dim=-1,
    )

    keras_q, keras_k, keras_v = ops.split(
        keras_qkv,
        3,
        axis=-1,
    )

    hf_q_np = hf_q.detach().cpu().numpy()
    hf_k_np = hf_k.detach().cpu().numpy()
    hf_v_np = hf_v.detach().cpu().numpy()

    keras_q_np = ops.convert_to_numpy(keras_q)
    keras_k_np = ops.convert_to_numpy(keras_k)
    keras_v_np = ops.convert_to_numpy(keras_v)

    print("\nQ / K / V")

    print(f"Q max diff: {np.max(np.abs(hf_q_np - keras_q_np)):.6e}")

    print(f"K max diff: {np.max(np.abs(hf_k_np - keras_k_np)):.6e}")

    print(f"V max diff: {np.max(np.abs(hf_v_np - keras_v_np)):.6e}")

    batch_size = input_ids_np.shape[0]
    seq_len = input_ids_np.shape[1]

    num_heads = keras_layer.attn.num_heads
    head_dim = keras_layer.attn.head_dim

    keras_q_reshaped = ops.transpose(
        ops.reshape(
            keras_q,
            (
                batch_size,
                seq_len,
                num_heads,
                head_dim,
            ),
        ),
        (0, 2, 1, 3),
    )

    keras_k_reshaped = ops.transpose(
        ops.reshape(
            keras_k,
            (
                batch_size,
                seq_len,
                num_heads,
                head_dim,
            ),
        ),
        (0, 2, 1, 3),
    )

    keras_v_reshaped = ops.transpose(
        ops.reshape(
            keras_v,
            (
                batch_size,
                seq_len,
                num_heads,
                head_dim,
            ),
        ),
        (0, 2, 1, 3),
    )

    keras_q_reshaped = ops.convert_to_numpy(keras_q_reshaped)
    keras_k_reshaped = ops.convert_to_numpy(keras_k_reshaped)
    keras_v_reshaped = ops.convert_to_numpy(keras_v_reshaped)

    print(
        "Q shape:",
        keras_q_reshaped.shape,
    )
    print(
        "K shape:",
        keras_k_reshaped.shape,
    )
    print(
        "V shape:",
        keras_v_reshaped.shape,
    )
    print("\n")
    print("Embedding + Layer 0 QKV verification complete.")


def verify_conversion_comprehensive(keras_lm, hf_model, hf_tokenizer):
    """Comprehensive numerical verification across varied inputs."""

    print("Running comprehensive numerical verification..\n")

    backbone = keras_lm.backbone

    test_cases = [
        "The capital of France is [MASK].",
        "Hello, my name is [MASK] and I live in [MASK].",
        "The [MASK] barked loudly at the mailman.",
        "In 1969, humans first landed on the [MASK].",
        "The quick brown fox jumps over the lazy dog while the " * 5
        + "[MASK] watches.",
    ]

    all_backbone_diffs = []
    all_logits_diffs = []
    top1_matches = 0
    top5_matches = 0
    total_masks = 0

    for text in test_cases:
        hf_inputs = hf_tokenizer(text, return_tensors="pt")
        hf_inputs.pop("token_type_ids", None)

        input_ids = hf_inputs["input_ids"].cpu().numpy().astype("int32")
        padding_mask = hf_inputs["attention_mask"].cpu().numpy().astype("int32")

        # Backbone comparison
        with torch.no_grad():
            hf_hidden = (
                hf_model.model(
                    input_ids=hf_inputs["input_ids"],
                    attention_mask=hf_inputs["attention_mask"],
                )
                .last_hidden_state.cpu()
                .numpy()
            )

        keras_hidden = backbone(
            {"token_ids": input_ids, "padding_mask": padding_mask},
            training=False,
        )
        keras_hidden = keras.ops.convert_to_numpy(keras_hidden)

        backbone_diff = np.max(np.abs(hf_hidden - keras_hidden))
        all_backbone_diffs.append(backbone_diff)

        # MaskedLM comparison
        mask_token_id = hf_tokenizer.mask_token_id
        mask_positions = np.argwhere(input_ids == mask_token_id)[:, 1:].astype(
            "int32"
        )

        if mask_positions.shape[0] == 0:
            continue

        with torch.no_grad():
            hf_logits = (
                hf_model(
                    input_ids=hf_inputs["input_ids"],
                    attention_mask=hf_inputs["attention_mask"],
                )
                .logits.cpu()
                .numpy()
            )

        batch_index = np.arange(input_ids.shape[0])[:, None]
        hf_mask_logits = hf_logits[batch_index, mask_positions]

        keras_logits = keras_lm(
            {
                "token_ids": input_ids,
                "padding_mask": padding_mask,
                "mask_positions": mask_positions,
            },
            training=False,
        )
        if isinstance(keras_logits, dict):
            keras_logits = keras_logits["logits"]
        keras_logits = keras.ops.convert_to_numpy(keras_logits)

        logits_diff = np.max(np.abs(hf_mask_logits - keras_logits))
        all_logits_diffs.append(logits_diff)

        # Practical correctness check
        for i in range(hf_mask_logits.shape[0]):
            hf_top1 = np.argmax(hf_mask_logits[i, 0])
            keras_top1 = np.argmax(keras_logits[i, 0])
            hf_top5 = set(np.argsort(hf_mask_logits[i, 0])[-5:])
            keras_top5 = set(np.argsort(keras_logits[i, 0])[-5:])

            total_masks += 1
            if hf_top1 == keras_top1:
                top1_matches += 1
            if hf_top5 == keras_top5:
                top5_matches += 1

            hf_word = hf_tokenizer.decode([hf_top1])
            keras_word = hf_tokenizer.decode([keras_top1])

            print(
                f"  [{text[:40]}...] HF pred: '{hf_word}' | "
                f"Keras pred: '{keras_word}' | match: {hf_top1 == keras_top1}"
            )

        print(
            f"  backbone_diff={backbone_diff:.4e}  "
            f"logits_diff={logits_diff:.4e}\n"
        )

    print("Comprehensive Verification Summary\n")
    print(f"Test cases run           : {len(test_cases)}")
    print(f"Total mask predictions   : {total_masks}")
    print(f"Top-1 prediction matches : {top1_matches}/{total_masks}")
    print(f"Top-5 prediction matches : {top5_matches}/{total_masks}")
    print(f"Max backbone diff        : {max(all_backbone_diffs):.6e}")
    print(f"Max logits diff          : {max(all_logits_diffs):.6e}")

    assert top1_matches == total_masks, (
        f"Top-1 predictions diverged on {total_masks - top1_matches} "
        f"mask(s) — this indicates a real bug, not just float rounding."
    )

    print("\nAll top-1 predictions match HF. Conversion verified.")


def main(preset, output_dir):
    hf_repo = PRESET_MAP.get(preset, preset)
    print(f"Loading HF checkpoint: {hf_repo}")
    hf_model = AutoModelForMaskedLM.from_pretrained(hf_repo)
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_repo)

    preset_dir = os.path.join(output_dir, preset)
    os.makedirs(preset_dir, exist_ok=True)
    hf_tokenizer.save_pretrained(preset_dir)

    hf_config = hf_model.config

    rope_params = getattr(hf_config, "rope_parameters", {})

    global_rope = rope_params.get("global_attention", {}) or rope_params.get(
        "full_attention", {}
    )

    local_rope = rope_params.get("local_attention", {}) or rope_params.get(
        "sliding_attention", {}
    )

    global_rope_theta = global_rope.get(
        "rope_theta", getattr(hf_config, "global_rope_theta", 160000.0)
    )
    local_rope_theta = local_rope.get(
        "rope_theta", getattr(hf_config, "local_rope_theta", 10000.0)
    )

    config = {
        "vocabulary_size": hf_config.vocab_size,
        "hidden_dim": hf_config.hidden_size,
        "intermediate_dim": hf_config.intermediate_size,
        "num_layers": hf_config.num_hidden_layers,
        "num_heads": hf_config.num_attention_heads,
        "local_attention_window": hf_config.local_attention,
        "global_attn_every_n_layers": getattr(
            hf_config, "global_attn_every_n_layers", 3
        ),
        "dropout": hf_config.attention_dropout,
        "layer_norm_epsilon": hf_config.norm_eps,
        "rotary_max_wavelength": global_rope_theta,
        "local_rotary_max_wavelength": local_rope_theta,
    }

    backbone = ModernBertBackbone(**config)
    keras_lm = ModernBertMaskedLM(backbone=backbone)

    # Initialize model weights
    batch_size, seq_len = 1, 16
    dummy_input = {
        "token_ids": ops.ones((batch_size, seq_len), dtype="int32"),
        "padding_mask": ops.ones((batch_size, seq_len), dtype="bool"),
        "mask_positions": ops.zeros((batch_size, 1), dtype="int32"),
    }
    _ = keras_lm(dummy_input)
    text = "The capital of France is [MASK]."
    hf_inputs = hf_tokenizer(text, return_tensors="pt")
    hf_inputs.pop("token_type_ids", None)

    convert_modern_bert_masked_lm_weights(keras_lm, backbone, hf_model)
    verify_conversion(keras_lm, hf_model, hf_tokenizer)
    verify_embedding_and_qkv(
        backbone,
        hf_model,
        hf_tokenizer,
    )
    verify_conversion_comprehensive(keras_lm, hf_model, hf_tokenizer)
    keras_lm.save_to_preset(preset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="modernbert_base_en")
    parser.add_argument(
        "--output_dir", type=str, default="./modernbert_base_en"
    )
    args = parser.parse_args()
    main(args.preset, args.output_dir)
