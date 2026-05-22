"""
Convert BLIP-2 Flan-T5 checkpoints to KerasHub format.

Supported HuggingFace model IDs
--------------------------------
  Salesforce/blip2-flan-t5-xl
  
  Salesforce/blip2-flan-t5-xxl

Usage:
```shell
python convert_blip2_t5_flan_checkpoints \
    --model_id Salesforce/blip2-flan-t5-xl \
    --output_dir blip2_flan_t5_xl_converted
```

Outputs
-------
  <output_dir>/
      model.weights.h5   – BLIP2Backbone weights (vision + qformer + T5 LM)
      spiece.model       – SentencePiece proto for BLIP2FlanT5Tokenizer

Architecture notes
------------------
  Vision encoder : EVA-CLIP ViT (frozen in BLIP-2 training)
  Q-Former       : Lightweight cross-attention bridge (32 query tokens)
  Language model : Flan-T5 encoder-decoder
      The Q-Former output is projected from ``qformer_hidden_dim`` to
      ``t5_hidden_dim`` and prepended to the T5 encoder input embeddings
      as a visual soft-prompt.  The T5 decoder generates text.

  Language projection weights are stored inside BLIP2FlanT5 (not the
  backbone directly) to keep the architecture self-contained.
"""

import gc
import os
import shutil

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from blip2_conversion_utils import print_header  # noqa: E402
from blip2_conversion_utils import report_diff  # noqa: E402
from blip2_conversion_utils import set_seed  # noqa: E402
from blip2_conversion_utils import to_np  # noqa: E402
from blip2_conversion_utils import transfer_projection_weights  # noqa: E402
from blip2_conversion_utils import transfer_qformer_weights  # noqa: E402
from blip2_conversion_utils import transfer_vision_weights  # noqa: E402
from blip2_conversion_utils import validate_projection  # noqa: E402
from blip2_conversion_utils import validate_qformer  # noqa: E402
from blip2_conversion_utils import validate_vision_encoder  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import Blip2ForConditionalGeneration  # noqa: E402
from transformers import Blip2Processor  # noqa: E402

from keras_hub.src.models.blip2.blip2_backbone import (
    BLIP2Backbone,  # noqa: E402
)
from keras_hub.src.models.blip2.blip2_causal_lm import (
    BLIP2CausalLM,  # noqa: E402
)
from keras_hub.src.models.blip2.blip2_causal_lm_preprocessor import (  # noqa: E402
    BLIP2CausalLMPreprocessor,
)
from keras_hub.src.models.blip2.blip2_flan_t5_lm import (
    BLIP2FlanT5,  # noqa: E402
)
from keras_hub.src.models.blip2.blip2_flan_t5_tokenizer import (
    BLIP2FlanT5Tokenizer,  # noqa: E402
)
from keras_hub.src.models.blip2.blip2_image_converter import (  # noqa: E402
    BLIP2ImageConverter,
)
from keras_hub.src.models.blip2.blip2_qformer import BLIP2QFormer  # noqa: E402
from keras_hub.src.models.blip2.blip2_vision_encoder import (  # noqa: E402
    BLIP2VisionEncoder,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_id",
    "Salesforce/blip2-flan-t5-xl",
    "HuggingFace model ID.  Supported: blip2-flan-t5-xl, blip2-flan-t5-xxl.",
)
flags.DEFINE_string(
    "output_dir",
    "blip2_flan_t5_xl_converted",
    "Output directory for converted weights and tokenizer assets.",
)
flags.DEFINE_bool(
    "skip_generate",
    False,
    "Skip the end-to-end generation validation (slow).",
)

_VALIDATION_PROMPT = "Question: What is in this picture? Answer:"


# ── T5 weight transfer ────────────────────────────────────────────────────────


def transfer_t5_weights(keras_flan_t5, hf_t5_model) -> None:
    """Transfer weights from HuggingFace T5ForConditionalGeneration.

    ``hf_t5_model`` should be ``hf_model.language_model``.

    HF weight note
    --------------
    All projection kernels in HF T5 are stored as ``(out_dim, in_dim)``
    (PyTorch convention) so we ``.transpose(1, 0)`` before assigning to
    Keras Dense layers that expect ``(in_dim, out_dim)``.
    T5LayerNorm has only a ``weight`` (gamma); no bias.
    """
    print("Transferring T5 (Flan-T5) weights...")
    pt_state = hf_t5_model.state_dict()
    t5 = keras_flan_t5.t5

    # Shared token embedding
    t5.token_embedding.embeddings.assign(to_np(pt_state["shared.weight"]))

    num_layers = t5.num_layers

    # ── Encoder layers ───────────────────────────────────────────────────────
    for i in range(num_layers):
        enc = t5.encoder_transformer_layers[i]
        ep = f"encoder.block.{i}.layer"

        # Self-attention Q/K/V/O
        enc.self_attention.query_projector.kernel.assign(
            to_np(pt_state[f"{ep}.0.SelfAttention.q.weight"]).T
        )
        enc.self_attention.key_projector.kernel.assign(
            to_np(pt_state[f"{ep}.0.SelfAttention.k.weight"]).T
        )
        enc.self_attention.value_projector.kernel.assign(
            to_np(pt_state[f"{ep}.0.SelfAttention.v.weight"]).T
        )
        enc.self_attention.output_projector.kernel.assign(
            to_np(pt_state[f"{ep}.0.SelfAttention.o.weight"]).T
        )
        if enc.self_attention.use_relative_attention_bias:
            enc.self_attention.relative_attention_bias.assign(
                to_np(
                    pt_state[
                        f"{ep}.0.SelfAttention.relative_attention_bias.weight"
                    ]
                )
            )
        enc.self_attention_layer_norm.weight.assign(
            to_np(pt_state[f"{ep}.0.layer_norm.weight"])
        )

        # FFN: gated-GELU (wi_0 → input_projector, wi_1 → gate_projector)
        enc.input_projector.weights[0].assign(
            to_np(pt_state[f"{ep}.1.DenseReluDense.wi_0.weight"]).T
        )
        enc.gate_projector.weights[0].assign(
            to_np(pt_state[f"{ep}.1.DenseReluDense.wi_1.weight"]).T
        )
        enc.output_projector.weights[0].assign(
            to_np(pt_state[f"{ep}.1.DenseReluDense.wo.weight"]).T
        )
        enc.layer_norm.weight.assign(
            to_np(pt_state[f"{ep}.1.layer_norm.weight"])
        )

    t5.encoder_layer_norm.weight.assign(
        to_np(pt_state["encoder.final_layer_norm.weight"])
    )

    # ── Decoder layers ───────────────────────────────────────────────────────
    for i in range(num_layers):
        dec = t5.decoder_transformer_layers[i]
        dp = f"decoder.block.{i}.layer"

        # Self-attention Q/K/V/O
        dec.self_attention.query_projector.kernel.assign(
            to_np(pt_state[f"{dp}.0.SelfAttention.q.weight"]).T
        )
        dec.self_attention.key_projector.kernel.assign(
            to_np(pt_state[f"{dp}.0.SelfAttention.k.weight"]).T
        )
        dec.self_attention.value_projector.kernel.assign(
            to_np(pt_state[f"{dp}.0.SelfAttention.v.weight"]).T
        )
        dec.self_attention.output_projector.kernel.assign(
            to_np(pt_state[f"{dp}.0.SelfAttention.o.weight"]).T
        )
        if dec.self_attention.use_relative_attention_bias:
            dec.self_attention.relative_attention_bias.assign(
                to_np(
                    pt_state[
                        f"{dp}.0.SelfAttention.relative_attention_bias.weight"
                    ]
                )
            )
        dec.self_attention_layer_norm.weight.assign(
            to_np(pt_state[f"{dp}.0.layer_norm.weight"])
        )

        # Cross-attention Q/K/V/O (layer index 1 for decoder)
        dec.cross_attention.query_projector.kernel.assign(
            to_np(pt_state[f"{dp}.1.EncDecAttention.q.weight"]).T
        )
        dec.cross_attention.key_projector.kernel.assign(
            to_np(pt_state[f"{dp}.1.EncDecAttention.k.weight"]).T
        )
        dec.cross_attention.value_projector.kernel.assign(
            to_np(pt_state[f"{dp}.1.EncDecAttention.v.weight"]).T
        )
        dec.cross_attention.output_projector.kernel.assign(
            to_np(pt_state[f"{dp}.1.EncDecAttention.o.weight"]).T
        )
        dec.cross_attention_layer_norm.weight.assign(
            to_np(pt_state[f"{dp}.1.layer_norm.weight"])
        )

        # FFN (layer index 2 for decoder)
        dec.input_projector.weights[0].assign(
            to_np(pt_state[f"{dp}.2.DenseReluDense.wi_0.weight"]).T
        )
        dec.gate_projector.weights[0].assign(
            to_np(pt_state[f"{dp}.2.DenseReluDense.wi_1.weight"]).T
        )
        dec.output_projector.weights[0].assign(
            to_np(pt_state[f"{dp}.2.DenseReluDense.wo.weight"]).T
        )
        dec.layer_norm.weight.assign(
            to_np(pt_state[f"{dp}.2.layer_norm.weight"])
        )

    t5.decoder_layer_norm.weight.assign(
        to_np(pt_state["decoder.final_layer_norm.weight"])
    )
    print("✓ T5 weights transferred")


# ── T5 validation ─────────────────────────────────────────────────────────────


def validate_t5_encoder(keras_flan_t5, hf_model) -> bool:
    """Validate T5 encoder output on a short dummy sequence.

    Runs the encoder with token IDs [1, 2, 3, 4, 5] (no visual prefix)
    and compares last hidden states between Keras and HF.
    """
    print_header("T5 ENCODER VALIDATION")

    dummy_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    dummy_mask = np.ones_like(dummy_ids, dtype=np.int32)

    hf_t5 = hf_model.language_model
    with torch.no_grad():
        pt_out = to_np(
            hf_t5.encoder(
                input_ids=torch.tensor(dummy_ids),
                attention_mask=torch.tensor(dummy_mask),
            ).last_hidden_state
        )

    # Run Keras T5 encoder manually (no visual prefix for isolation)
    t5 = keras_flan_t5.t5
    x = t5.token_embedding(dummy_ids)
    attn_mask = dummy_mask[:, None, :]
    position_bias = None
    for layer in t5.encoder_transformer_layers:
        out = layer(
            x,
            attention_mask=attn_mask,
            position_bias=position_bias,
            use_causal_mask=False,
            training=False,
        )
        if isinstance(out, tuple):
            x, position_bias = out
    keras_out = to_np(t5.encoder_layer_norm(x))

    print(f"   -> HF    shape : {pt_out.shape}")
    print(f"   -> Keras shape : {keras_out.shape}")
    return report_diff("T5 encoder hidden states", keras_out, pt_out)


def validate_t5_decoder(keras_flan_t5, hf_model) -> bool:
    """Validate T5 encoder+decoder output on short dummy sequences."""
    print_header("T5 DECODER VALIDATION")

    enc_ids = np.array([[1, 2, 3]], dtype=np.int32)
    enc_mask = np.ones_like(enc_ids, dtype=np.int32)
    dec_ids = np.array([[0, 1, 2]], dtype=np.int32)
    dec_mask = np.ones_like(dec_ids, dtype=np.int32)

    hf_t5 = hf_model.language_model
    with torch.no_grad():
        hf_out = hf_t5(
            input_ids=torch.tensor(enc_ids),
            attention_mask=torch.tensor(enc_mask),
            decoder_input_ids=torch.tensor(dec_ids),
            decoder_attention_mask=torch.tensor(dec_mask),
            output_hidden_states=True,
        )
        # decoder_hidden_states[-1] is the last layer output (after final norm)
        pt_out = to_np(hf_out.decoder_hidden_states[-1])

    # Run Keras encoder then decoder manually
    t5 = keras_flan_t5.t5
    # Encoder
    x = t5.token_embedding(enc_ids)
    enc_attn = enc_mask[:, None, :]
    pos_bias = None
    for layer in t5.encoder_transformer_layers:
        out = layer(
            x,
            attention_mask=enc_attn,
            position_bias=pos_bias,
            use_causal_mask=False,
            training=False,
        )
        if isinstance(out, tuple):
            x, pos_bias = out
    x = t5.encoder_layer_norm(x)
    encoder_out = x
    # Decoder
    x = t5.token_embedding(dec_ids)
    dec_attn = dec_mask[:, None, :]
    pos_bias = None
    for layer in t5.decoder_transformer_layers:
        out = layer(
            x,
            attention_mask=dec_attn,
            position_bias=pos_bias,
            encoder_hidden_states=encoder_out,
            encoder_attention_mask=enc_attn,
            use_causal_mask=True,
            training=False,
        )
        if isinstance(out, tuple):
            x, pos_bias = out
    keras_out = to_np(t5.decoder_layer_norm(x))

    print(f"   -> HF    shape : {pt_out.shape}")
    print(f"   -> Keras shape : {keras_out.shape}")
    return report_diff("T5 decoder hidden states", keras_out, pt_out)


def validate_t5_lm(
    keras_flan_t5, hf_model, qformer_out_np: np.ndarray, hf_processor
) -> bool:
    """Compare decoder logits for the full BLIP-2 T5 forward pass.

    Feeds the projected Q-Former output + tokenised prompt to the T5
    encoder (matching what BLIP2FlanT5.call does) and a single decoder
    step, then compares logits between HF and Keras.
    """
    print_header("T5 LM LOGITS VALIDATION")

    # ── Parameter count ───────────────────────────────────────────────────────
    # Keras: BLIP2FlanT5 holds T5 + language_projection (token_embedding is
    #        shared so counted once).
    # HF: language_model (T5ForConditionalGeneration) + language_projection,
    #     minus the three tied/duplicated embedding tables that HF counts twice:
    #     lm_head, encoder.embed_tokens, decoder.embed_tokens.
    keras_params = keras_flan_t5.count_params()
    hf_t5_params = sum(p.numel() for p in hf_model.language_model.parameters())
    hf_proj_params = sum(
        p.numel() for p in hf_model.language_projection.parameters()
    )
    embed_params = hf_model.language_model.shared.weight.numel()
    # lm_head + encoder.embed_tokens share the same tensor → subtract 2 copies
    hf_params = hf_t5_params + hf_proj_params - 2 * embed_params
    print(f"   -> Keras params : {keras_params:,}")
    print(f"   -> HF params    : {hf_params:,}")
    if keras_params == hf_params:
        print("   -> ✅ Parameter counts match")
    else:
        diff = abs(keras_params - hf_params)
        print(f"   -> ⚠️  Mismatch by {diff:,} parameters")

    hf_inputs = hf_processor(
        images=Image.new("RGB", (224, 224), color=(114, 114, 114)),
        text=_VALIDATION_PROMPT,
        return_tensors="pt",
        padding=False,
    )
    text_ids_pt = hf_inputs["input_ids"]
    text_ids_np = text_ids_pt.numpy()
    text_mask_np = np.ones_like(text_ids_np, dtype=np.int32)

    # Decoder input: single pad/BOS token (T5 uses 0)
    dec_ids_pt = torch.zeros((1, 1), dtype=torch.long)
    dec_ids_np = np.zeros((1, 1), dtype=np.int32)
    dec_mask_np = np.ones((1, 1), dtype=np.int32)

    hf_t5 = hf_model.language_model
    with torch.no_grad():
        proj_pt = hf_model.language_projection(torch.tensor(qformer_out_np))
        text_emb_pt = hf_t5.encoder.embed_tokens(text_ids_pt)
        enc_embeds_pt = torch.cat([proj_pt, text_emb_pt], dim=1)
        enc_mask_pt = torch.ones(1, enc_embeds_pt.shape[1], dtype=torch.long)
        hf_out = hf_t5(
            inputs_embeds=enc_embeds_pt,
            attention_mask=enc_mask_pt,
            decoder_input_ids=dec_ids_pt,
        )
    hf_logits = hf_out.logits.float().numpy()  # (1, 1, vocab_size)

    keras_hidden = keras_flan_t5(
        {
            "qformer_features": qformer_out_np,
            "token_ids": text_ids_np,
            "padding_mask": text_mask_np,
            "decoder_token_ids": dec_ids_np,
            "decoder_padding_mask": dec_mask_np,
        },
        training=False,
    )
    keras_logits = to_np(
        keras_flan_t5.token_embedding(keras_hidden, reverse=True)
    )  # (1, 1, vocab_size)

    hf_pred = int(np.argmax(hf_logits[0, 0]))
    keras_pred = int(np.argmax(keras_logits[0, 0]))
    print(f"   -> HF    logits shape : {hf_logits.shape}")
    print(f"   -> Keras logits shape : {keras_logits.shape}")
    print(f"   -> HF    top-1 token  : {hf_pred}")
    print(f"   -> Keras top-1 token  : {keras_pred}")
    print(
        f"   -> Top-1 {'✅ match' if hf_pred == keras_pred else '⚠️  mismatch'}"
    )

    return report_diff("T5 LM logits", keras_logits, hf_logits)


def validate_generate(keras_flan_t5, hf_model, hf_processor) -> None:
    """Greedy-decode up to 20 tokens and compare HF vs manual Keras output.

    HF generation uses ``hf_model.generate()``.  Keras generation is a
    manual greedy loop: the encoder runs once (with the visual prefix),
    then the decoder is called step-by-step, argmax-selecting the next
    token at each step.
    """
    print_header("GENERATION VALIDATION (greedy, 20 tokens)")

    max_new_tokens = 20
    image_pil = Image.new("RGB", (224, 224), color=(114, 114, 114))

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
    hf_text = hf_processor.batch_decode(
        hf_generated, skip_special_tokens=True
    )[0]

    # ── Manual Keras greedy decode ────────────────────────────────────────
    text_ids_np = hf_inputs["input_ids"].numpy()
    text_mask_np = np.ones_like(text_ids_np, dtype=np.int32)

    # Dummy image -> vision -> qformer to get visual features for Keras
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    image_pt = torch.tensor(image_np[None]).permute(0, 3, 1, 2)
    with torch.no_grad():
        vis_np = to_np(
            hf_model.vision_model(pixel_values=image_pt).last_hidden_state
        )
        qt_pt = hf_model.query_tokens.expand(1, -1, -1)
        qf_np = to_np(
            hf_model.qformer(
                query_embeds=qt_pt,
                encoder_hidden_states=torch.tensor(vis_np),
            ).last_hidden_state
        )

    # Build encoder output once (projected qformer + text prompt)
    t5 = keras_flan_t5.t5
    proj_enc = to_np(keras_flan_t5.language_projection(qf_np))
    text_emb = to_np(t5.token_embedding(text_ids_np))
    enc_emb = np.concatenate([proj_enc, text_emb], axis=1)
    vis_m = np.ones((1, qf_np.shape[1]), dtype=np.int32)
    enc_mask = np.concatenate([vis_m, text_mask_np], axis=1)
    enc_attn = enc_mask[:, None, :]

    x = enc_emb
    pos_bias = None
    for layer in t5.encoder_transformer_layers:
        out = layer(
            x,
            attention_mask=enc_attn,
            position_bias=pos_bias,
            use_causal_mask=False,
            training=False,
        )
        if isinstance(out, tuple):
            x, pos_bias = out
    x = t5.encoder_layer_norm(x)
    encoder_out = x  # fixed for all decoder steps

    # Greedy decode loop
    pad_token_id = 0
    eos_token_id = 1  # </s> in T5 vocabulary
    dec_ids = np.array([[pad_token_id]], dtype=np.int32)
    generated = []

    for _ in range(max_new_tokens):
        dec_mask = np.ones_like(dec_ids, dtype=np.int32)
        dec_attn = dec_mask[:, None, :]
        x = t5.token_embedding(dec_ids)
        d_pos_bias = None
        for layer in t5.decoder_transformer_layers:
            out = layer(
                x,
                attention_mask=dec_attn,
                position_bias=d_pos_bias,
                encoder_hidden_states=encoder_out,
                encoder_attention_mask=enc_attn,
                use_causal_mask=True,
                training=False,
            )
            if isinstance(out, tuple):
                x, d_pos_bias = out
        x = t5.decoder_layer_norm(x)
        scale = keras_flan_t5.hidden_dim**-0.5
        logits = to_np(
            keras_flan_t5.token_embedding(x * scale, reverse=True)
        )  # (1, dec_len, vocab)
        next_token = int(np.argmax(logits[0, -1]))
        if next_token == eos_token_id:
            break
        generated.append(next_token)
        dec_ids = np.concatenate(
            [dec_ids, np.array([[next_token]], dtype=np.int32)], axis=1
        )

    keras_text = hf_processor.batch_decode(
        [generated], skip_special_tokens=True
    )[0]

    print(f"\n   -> HuggingFace : {hf_text!r}")
    print(f"   -> Keras       : {keras_text!r}")
    match = keras_text.strip() == hf_text.strip()
    print(f"   -> {'✅ Match' if match else '⚠️  No match'}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(_):
    set_seed(42)
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    print_header(f"Loading HF model: {FLAGS.model_id}")
    hf_model = Blip2ForConditionalGeneration.from_pretrained(
        FLAGS.model_id, torch_dtype=torch.float32
    )
    hf_processor = Blip2Processor.from_pretrained(FLAGS.model_id)
    hf_model.eval()

    # Verify all parameters are float32
    bad = [
        (n, p.dtype)
        for n, p in hf_model.named_parameters()
        if p.dtype != torch.float32
    ]
    if bad:
        print("⚠️  Non-fp32 parameters found — forcing float32:")
        for name, dtype in bad:
            print(f"   {name}: {dtype}")
        hf_model = hf_model.float()
    else:
        print("✅ All HF parameters confirmed float32")

    hf_config = hf_model.config
    hf_vision_config = hf_model.vision_model.config
    hf_qformer_config = hf_model.qformer.config
    hf_t5_config = hf_model.language_model.config

    num_query_tokens = getattr(hf_config, "num_query_tokens", 32)

    # ── Vision encoder ───────────────────────────────────────────────────────
    print_header("Building Vision Encoder")
    vision_config = {
        "image_size": hf_vision_config.image_size,
        "patch_size": hf_vision_config.patch_size,
        "num_layers": hf_vision_config.num_hidden_layers,
        "num_heads": hf_vision_config.num_attention_heads,
        "hidden_dim": hf_vision_config.hidden_size,
        "intermediate_dim": hf_vision_config.intermediate_size,
        "use_patch_bias": True,
        "use_class_token": True,
        "use_mha_bias": True,
        "use_mlp_bias": True,
        "dropout_rate": 0.0,
        "layer_norm_epsilon": hf_vision_config.layer_norm_eps,
        "initializer_range": hf_vision_config.initializer_range,
    }
    v_enc = BLIP2VisionEncoder(**vision_config)
    v_enc.build((None, 224, 224, 3))
    transfer_vision_weights(v_enc, hf_model.vision_model)

    # ── Q-Former ─────────────────────────────────────────────────────────────
    print_header("Building Q-Former")
    qf = BLIP2QFormer(
        num_query_tokens=num_query_tokens,
        num_layers=hf_qformer_config.num_hidden_layers,
        num_heads=hf_qformer_config.num_attention_heads,
        hidden_dim=hf_qformer_config.hidden_size,
        intermediate_dim=hf_qformer_config.intermediate_size,
        vision_dim=hf_vision_config.hidden_size,
        cross_attention_frequency=hf_qformer_config.cross_attention_frequency,
        dropout=hf_qformer_config.hidden_dropout_prob,
        layer_norm_epsilon=hf_qformer_config.layer_norm_eps,
    )
    qf.build(
        (None, v_enc.num_vision_tokens_per_image, hf_vision_config.hidden_size)
    )
    transfer_qformer_weights(qf, hf_model)

    # ── Flan-T5 language model ────────────────────────────────────────────────
    print_header("Building BLIP2FlanT5")
    t5_config = {
        "vocabulary_size": hf_t5_config.vocab_size,
        "num_layers": hf_t5_config.num_layers,
        "num_heads": hf_t5_config.num_heads,
        "hidden_dim": hf_t5_config.d_model,
        "intermediate_dim": hf_t5_config.d_ff,
        "key_value_dim": hf_t5_config.d_kv,
        "num_query_tokens": num_query_tokens,
        "qformer_hidden_dim": hf_qformer_config.hidden_size,
        "dropout": hf_t5_config.dropout_rate,
        "layer_norm_epsilon": hf_t5_config.layer_norm_epsilon,
    }
    flan_t5 = BLIP2FlanT5(**t5_config)

    # Build with dummy inputs to initialise all weights
    dummy_t5_inputs = {
        "token_ids": np.zeros((1, 8), dtype=np.int32),
        "padding_mask": np.ones((1, 8), dtype=np.int32),
        "decoder_token_ids": np.zeros((1, 4), dtype=np.int32),
        "decoder_padding_mask": np.ones((1, 4), dtype=np.int32),
        "qformer_features": np.zeros(
            (1, num_query_tokens, hf_qformer_config.hidden_size),
            dtype=np.float32,
        ),
    }
    _ = flan_t5(dummy_t5_inputs, training=False)

    transfer_t5_weights(flan_t5, hf_model.language_model)
    transfer_projection_weights(
        flan_t5.language_projection, hf_model.language_projection
    )

    # ── BLIP2Backbone ─────────────────────────────────────────────────────────
    print_header("Building BLIP2Backbone")
    backbone = BLIP2Backbone(
        vision_encoder=v_enc,
        qformer=qf,
        language_model=flan_t5,
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print_header("Saving tokenizer assets")
    try:
        spiece_src = hf_hub_download(
            repo_id=FLAGS.model_id, filename="spiece.model"
        )
    except Exception:
        spiece_src = hf_hub_download(
            repo_id=FLAGS.model_id, filename="tokenizer.model"
        )
    spiece_dst = os.path.join(FLAGS.output_dir, "spiece.model")
    shutil.copy(spiece_src, spiece_dst)
    print(f"✓ spiece.model saved to {spiece_dst}")

    tokenizer = BLIP2FlanT5Tokenizer(proto=spiece_dst)
    preprocessor = BLIP2CausalLMPreprocessor(
        tokenizer=tokenizer,
        image_converter=BLIP2ImageConverter(
            image_size=(
                hf_vision_config.image_size,
                hf_vision_config.image_size,
            )
        ),
    )
    causal_lm = BLIP2CausalLM(backbone=backbone, preprocessor=preprocessor)

    # ── Validation ────────────────────────────────────────────────────────────
    image_pil = Image.new(
        "RGB",
        (hf_vision_config.image_size, hf_vision_config.image_size),
        color=(114, 114, 114),
    )
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

    proj_ok = validate_projection(
        flan_t5.language_projection, hf_model, qformer_out_np
    )
    enc_ok = validate_t5_encoder(flan_t5, hf_model)
    dec_ok = validate_t5_decoder(flan_t5, hf_model)
    lm_ok = validate_t5_lm(flan_t5, hf_model, qformer_out_np, hf_processor)

    if not FLAGS.skip_generate:
        validate_generate(flan_t5, hf_model, hf_processor)
    else:
        print("\n⏭️  Skipping generation validation (--skip_generate=True)")

    print_header("SUBMODULE VALIDATION SUMMARY")
    for name, ok in [
        ("Vision Encoder", vision_ok),
        ("Q-Former", qformer_ok),
        ("Projection", proj_ok),
        ("T5 Encoder", enc_ok),
        ("T5 Decoder", dec_ok),
        ("T5 LM Logits", lm_ok),
    ]:
        status = "✅ PASS" if ok else "⚠️  FAIL"
        print(f"   {status}  {name}")

    # ── Save weights ──────────────────────────────────────────────────────────
    print_header("SAVING WEIGHTS")
    weights_path = os.path.join(FLAGS.output_dir, "model.weights.h5")
    backbone.save_weights(weights_path)
    print(f"✓ Weights saved to {weights_path}")

    del hf_model, causal_lm, backbone
    gc.collect()

    print("\n✓ Conversion complete.")


if __name__ == "__main__":
    app.run(main)
