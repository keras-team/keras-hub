import gc
import os
import random
import shutil

import huggingface_hub
import keras
import numpy as np
import requests
import torch
import transformers
from absl import app
from absl import flags
from checkpoint_conversion_utils import get_md5_checksum
from PIL import Image

import keras_hub
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.t5gemma2.t5gemma2_seq_2_seq_lm import (
    T5Gemma2Seq2SeqLM,
)
from keras_hub.src.models.t5gemma2.t5gemma2_seq_2_seq_lm_preprocessor import (
    T5Gemma2Seq2SeqLMPreprocessor,
)

random.seed(123)
torch.manual_seed(123)
device = torch.device("cpu")
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)

PRESET_MAP = {
    "t5gemma2_270m_270m": "google/t5gemma-2-270m-270m",
    "t5gemma2_1b_1b": "google/t5gemma-2-1b-1b",
    "t5gemma2_4b_4b": "google/t5gemma-2-4b-4b",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
)


def convert_checkpoints(hf_model):
    """Convert HuggingFace T5Gemma2 weights to KerasHub format."""
    print("\n-> Convert original weights to KerasHub format.")
    print("\n-> Load KerasHub model.")

    encoder_config = hf_model.config.encoder
    enc_text_config = encoder_config.text_config
    decoder_config = hf_model.config.decoder
    if decoder_config.hidden_activation == "gelu_pytorch_tanh":
        decoder_config.hidden_activation = "gelu_approximate"
    if enc_text_config.hidden_activation == "gelu_pytorch_tanh":
        enc_text_config.hidden_activation = "gelu_approximate"

    # Vision encoder (optional — only present in multimodal models).
    vision_encoder = None
    has_vision = hasattr(encoder_config, "vision_config") and (
        encoder_config.vision_config is not None
    )
    if has_vision:
        from keras_hub.src.models.gemma3.gemma3_vision_encoder import (
            Gemma3VisionEncoder,
        )

        vc = encoder_config.vision_config
        mm_tokens = getattr(encoder_config, "mm_tokens_per_image", 256)
        pool_size = int(vc.image_size // vc.patch_size // int(mm_tokens**0.5))
        vision_encoder = Gemma3VisionEncoder(
            image_size=vc.image_size,
            patch_size=vc.patch_size,
            num_heads=vc.num_attention_heads,
            hidden_dim=vc.hidden_size,
            num_layers=vc.num_hidden_layers,
            intermediate_dim=vc.intermediate_size,
            output_dim=enc_text_config.hidden_size,
            pool_size=pool_size,
            layer_norm_epsilon=getattr(vc, "layer_norm_eps", 1e-6),
        )
        print(
            f"  Vision encoder created: {vc.image_size}x{vc.image_size}, "
            f"pool_size={pool_size}, mm_tokens={mm_tokens}"
        )

    keras.config.set_floatx("float32")
    keras_hub_model = keras_hub.models.T5Gemma2Backbone(
        vocabulary_size=decoder_config.vocab_size,
        encoder_hidden_dim=enc_text_config.hidden_size,
        encoder_intermediate_dim=enc_text_config.intermediate_size,
        encoder_num_layers=enc_text_config.num_hidden_layers,
        encoder_num_attention_heads=(enc_text_config.num_attention_heads),
        encoder_num_key_value_heads=(enc_text_config.num_key_value_heads),
        encoder_head_dim=enc_text_config.head_dim,
        encoder_layer_types=enc_text_config.layer_types,
        decoder_hidden_dim=decoder_config.hidden_size,
        decoder_intermediate_dim=decoder_config.intermediate_size,
        decoder_num_layers=decoder_config.num_hidden_layers,
        decoder_num_attention_heads=decoder_config.num_attention_heads,
        decoder_num_key_value_heads=(decoder_config.num_key_value_heads),
        decoder_head_dim=decoder_config.head_dim,
        decoder_layer_types=decoder_config.layer_types,
        dropout_rate=decoder_config.dropout_rate,
        rms_norm_eps=decoder_config.rms_norm_eps,
        query_pre_attn_scalar=decoder_config.query_pre_attn_scalar,
        tie_word_embeddings=getattr(
            hf_model.config, "tie_word_embeddings", True
        ),
        attention_bias=decoder_config.attention_bias,
        hidden_activation=decoder_config.hidden_activation,
        initializer_range=decoder_config.initializer_range,
        attention_dropout=decoder_config.attention_dropout,
        sliding_window=decoder_config.sliding_window,
        cross_attention_hidden_size=enc_text_config.hidden_size,
        attn_logit_softcapping=(decoder_config.attn_logit_softcapping),
        final_logit_softcapping=(decoder_config.final_logit_softcapping),
        rope_max_wavelength=(
            decoder_config.rope_parameters["sliding_attention"]["rope_theta"]
        ),
        global_rope_scaling_factor=(
            decoder_config.rope_parameters["full_attention"]["factor"]
        ),
        encoder_rope_max_wavelength=(
            enc_text_config.rope_parameters["sliding_attention"]["rope_theta"]
        ),
        encoder_global_rope_scaling_factor=(
            enc_text_config.rope_parameters["full_attention"]["factor"]
        ),
        use_query_key_norm=any(
            "q_norm" in k for k in hf_model.state_dict().keys()
        ),
        vision_encoder=vision_encoder,
        eoi_token_index=hf_model.config.eoi_token_index,
        dtype="float32",
    )

    hf_wts = hf_model.state_dict()
    # Cast all weights to float32 (HF uses bfloat16).
    hf_wts = {k: v.float() for k, v in hf_wts.items()}

    # Token embeddings.
    # Encoder embeds are under encoder.text_model.embed_tokens.*
    keras_hub_model.get_layer("encoder_token_embedding").embeddings.assign(
        hf_wts["encoder.text_model.embed_tokens.weight"].numpy()
    )
    keras_hub_model.get_layer("decoder_token_embedding").embeddings.assign(
        hf_wts["decoder.embed_tokens.weight"].numpy()
    )

    # Vision encoder weights.
    if has_vision:
        ve = keras_hub_model.vision_encoder
        ie = ve.get_layer("image_encoder")
        ie.vision_embeddings.patch_embedding.kernel.assign(
            hf_wts[
                "encoder.vision_tower.vision_model."
                "embeddings.patch_embedding.weight"
            ]
            .permute(2, 3, 1, 0)
            .numpy()
        )
        ie.vision_embeddings.patch_embedding.bias.assign(
            hf_wts[
                "encoder.vision_tower.vision_model."
                "embeddings.patch_embedding.bias"
            ].numpy()
        )
        ie.vision_embeddings.position_embedding.embeddings.assign(
            hf_wts[
                "encoder.vision_tower.vision_model."
                "embeddings.position_embedding.weight"
            ].numpy()
        )
        for vi in range(ie.num_layers):
            vp = f"encoder.vision_tower.vision_model.encoder.layers.{vi}"
            rb = ie.resblocks[vi]
            rb.layer_norm_1.gamma.assign(
                hf_wts[f"{vp}.layer_norm1.weight"].numpy()
            )
            rb.layer_norm_1.beta.assign(
                hf_wts[f"{vp}.layer_norm1.bias"].numpy()
            )
            rb.attn.query_proj.kernel.assign(
                hf_wts[f"{vp}.self_attn.q_proj.weight"].T.numpy()
            )
            rb.attn.query_proj.bias.assign(
                hf_wts[f"{vp}.self_attn.q_proj.bias"].numpy()
            )
            rb.attn.key_proj.kernel.assign(
                hf_wts[f"{vp}.self_attn.k_proj.weight"].T.numpy()
            )
            rb.attn.key_proj.bias.assign(
                hf_wts[f"{vp}.self_attn.k_proj.bias"].numpy()
            )
            rb.attn.value_proj.kernel.assign(
                hf_wts[f"{vp}.self_attn.v_proj.weight"].T.numpy()
            )
            rb.attn.value_proj.bias.assign(
                hf_wts[f"{vp}.self_attn.v_proj.bias"].numpy()
            )
            rb.attn.out_proj.kernel.assign(
                hf_wts[f"{vp}.self_attn.out_proj.weight"].T.numpy()
            )
            rb.attn.out_proj.bias.assign(
                hf_wts[f"{vp}.self_attn.out_proj.bias"].numpy()
            )
            rb.layer_norm_2.gamma.assign(
                hf_wts[f"{vp}.layer_norm2.weight"].numpy()
            )
            rb.layer_norm_2.beta.assign(
                hf_wts[f"{vp}.layer_norm2.bias"].numpy()
            )
            rb.mlp_dense_1.kernel.assign(
                hf_wts[f"{vp}.mlp.fc1.weight"].T.numpy()
            )
            rb.mlp_dense_1.bias.assign(hf_wts[f"{vp}.mlp.fc1.bias"].numpy())
            rb.mlp_dense_2.kernel.assign(
                hf_wts[f"{vp}.mlp.fc2.weight"].T.numpy()
            )
            rb.mlp_dense_2.bias.assign(hf_wts[f"{vp}.mlp.fc2.bias"].numpy())
        ie.encoder_layer_norm.gamma.assign(
            hf_wts[
                "encoder.vision_tower.vision_model.post_layernorm.weight"
            ].numpy()
        )
        ie.encoder_layer_norm.beta.assign(
            hf_wts[
                "encoder.vision_tower.vision_model.post_layernorm.bias"
            ].numpy()
        )
        # Multi-modal projector.
        vo = ve.get_layer("vision_output_encoder")
        vo.vision_soft_embedding_norm.scale.assign(
            hf_wts[
                "encoder.multi_modal_projector.mm_soft_emb_norm.weight"
            ].numpy()
        )
        vo.vision_input_projection.kernel.assign(
            hf_wts[
                "encoder.multi_modal_projector.mm_input_projection_weight"
            ].numpy()
        )
        # EOI embeddings.
        keras_hub_model.encoder_eoi_embedding.assign(
            hf_wts["encoder.text_model.embed_tokens.eoi_embedding"].numpy()
        )
        keras_hub_model.decoder_eoi_embedding.assign(
            hf_wts["decoder.embed_tokens.eoi_embedding"].numpy()
        )
        print("  Vision encoder weights loaded.")

    # Encoder (weights under encoder.text_model.*).
    enc_hdim = keras_hub_model.encoder_hidden_dim
    enc_heads = keras_hub_model.encoder_num_attention_heads
    enc_kv_heads = keras_hub_model.encoder_num_key_value_heads
    enc_head_dim = keras_hub_model.encoder_head_dim
    keras_hub_model.encoder_norm.scale.assign(
        hf_wts["encoder.text_model.norm.weight"]
    )

    for i in range(keras_hub_model.encoder_num_layers):
        layer = keras_hub_model.get_layer(f"encoder_layer_{i}")
        pfx = f"encoder.text_model.layers.{i}"

        # Self-attention Q/K/V/O.
        layer.self_attn.query_dense.kernel.assign(
            hf_wts[f"{pfx}.self_attn.q_proj.weight"]
            .T.reshape(enc_hdim, enc_heads, enc_head_dim)
            .numpy()
        )
        layer.self_attn.key_dense.kernel.assign(
            hf_wts[f"{pfx}.self_attn.k_proj.weight"]
            .T.reshape(enc_hdim, enc_kv_heads, enc_head_dim)
            .numpy()
        )
        layer.self_attn.value_dense.kernel.assign(
            hf_wts[f"{pfx}.self_attn.v_proj.weight"]
            .T.reshape(enc_hdim, enc_kv_heads, enc_head_dim)
            .numpy()
        )
        layer.self_attn.output_dense.kernel.assign(
            hf_wts[f"{pfx}.self_attn.o_proj.weight"]
            .T.reshape(enc_heads, enc_head_dim, enc_hdim)
            .numpy()
        )

        # Q/K normalization.
        layer.self_attn.query_norm.scale.assign(
            hf_wts[f"{pfx}.self_attn.q_norm.weight"]
        )
        layer.self_attn.key_norm.scale.assign(
            hf_wts[f"{pfx}.self_attn.k_norm.weight"]
        )

        # MLP.
        layer.mlp.gate_proj.kernel.assign(
            hf_wts[f"{pfx}.mlp.gate_proj.weight"].T.numpy()
        )
        layer.mlp.up_proj.kernel.assign(
            hf_wts[f"{pfx}.mlp.up_proj.weight"].T.numpy()
        )
        layer.mlp.down_proj.kernel.assign(
            hf_wts[f"{pfx}.mlp.down_proj.weight"].T.numpy()
        )

        # Layer norms.
        layer.pre_self_attn_layernorm.scale.assign(
            hf_wts[f"{pfx}.pre_self_attn_layernorm.weight"]
        )
        layer.post_self_attn_layernorm.scale.assign(
            hf_wts[f"{pfx}.post_self_attn_layernorm.weight"]
        )
        layer.pre_feedforward_layernorm.scale.assign(
            hf_wts[f"{pfx}.pre_feedforward_layernorm.weight"]
        )
        layer.post_feedforward_layernorm.scale.assign(
            hf_wts[f"{pfx}.post_feedforward_layernorm.weight"]
        )

    # Decoder.
    dec_hdim = keras_hub_model.decoder_hidden_dim
    dec_heads = keras_hub_model.decoder_num_attention_heads
    dec_kv_heads = keras_hub_model.decoder_num_key_value_heads
    dec_head_dim = keras_hub_model.decoder_head_dim
    keras_hub_model.decoder_norm.scale.assign(hf_wts["decoder.norm.weight"])

    for i in range(keras_hub_model.decoder_num_layers):
        layer = keras_hub_model.get_layer(f"decoder_layer_{i}")
        pfx = f"decoder.layers.{i}"

        # Merged attention (self+cross uses single self_attn in HF).
        layer.merged_attn.query_dense.kernel.assign(
            hf_wts[f"{pfx}.self_attn.q_proj.weight"]
            .T.reshape(dec_hdim, dec_heads, dec_head_dim)
            .numpy()
        )
        layer.merged_attn.key_dense.kernel.assign(
            hf_wts[f"{pfx}.self_attn.k_proj.weight"]
            .T.reshape(dec_hdim, dec_kv_heads, dec_head_dim)
            .numpy()
        )
        layer.merged_attn.value_dense.kernel.assign(
            hf_wts[f"{pfx}.self_attn.v_proj.weight"]
            .T.reshape(dec_hdim, dec_kv_heads, dec_head_dim)
            .numpy()
        )
        layer.merged_attn.output_dense.kernel.assign(
            hf_wts[f"{pfx}.self_attn.o_proj.weight"]
            .T.reshape(dec_heads, dec_head_dim, dec_hdim)
            .numpy()
        )

        # Q/K normalization.
        layer.merged_attn.query_norm.scale.assign(
            hf_wts[f"{pfx}.self_attn.q_norm.weight"]
        )
        layer.merged_attn.key_norm.scale.assign(
            hf_wts[f"{pfx}.self_attn.k_norm.weight"]
        )

        # MLP.
        layer.mlp.gate_proj.kernel.assign(
            hf_wts[f"{pfx}.mlp.gate_proj.weight"].T.numpy()
        )
        layer.mlp.up_proj.kernel.assign(
            hf_wts[f"{pfx}.mlp.up_proj.weight"].T.numpy()
        )
        layer.mlp.down_proj.kernel.assign(
            hf_wts[f"{pfx}.mlp.down_proj.weight"].T.numpy()
        )

        # Layer norms (no cross-attn norms — merged into self_attn).
        layer.pre_self_attn_layernorm.scale.assign(
            hf_wts[f"{pfx}.pre_self_attn_layernorm.weight"]
        )
        layer.post_self_attn_layernorm.scale.assign(
            hf_wts[f"{pfx}.post_self_attn_layernorm.weight"]
        )
        layer.pre_feedforward_layernorm.scale.assign(
            hf_wts[f"{pfx}.pre_feedforward_layernorm.weight"]
        )
        layer.post_feedforward_layernorm.scale.assign(
            hf_wts[f"{pfx}.post_feedforward_layernorm.weight"]
        )

    return keras_hub_model


def extract_vocab(hf_model_dir):
    """Extract vocabulary from the downloaded HF model directory."""
    vocabulary_path = os.path.join(FLAGS.preset, "tokenizer.model")
    print(f"\n-> Save KerasHub vocab to `{vocabulary_path}`.")

    # T5Gemma2 HF repos only have tokenizer.json (no tokenizer.model).
    # The SentencePiece proto is the same as Gemma3's vocabulary.
    source_path = os.path.join(hf_model_dir, "tokenizer.model")
    if os.path.exists(source_path):
        shutil.copyfile(source_path, vocabulary_path)
    else:
        # Download tokenizer.model from Gemma3 (same vocab).
        print(
            "  tokenizer.model not found in HF repo. "
            "Downloading from google/gemma-3-1b-pt..."
        )
        gemma_dir = huggingface_hub.snapshot_download(
            repo_id="google/gemma-3-1b-pt",
            allow_patterns=["tokenizer.model"],
        )
        gemma_proto = os.path.join(gemma_dir, "tokenizer.model")
        shutil.copyfile(gemma_proto, vocabulary_path)

    keras_hub_tokenizer = keras_hub.models.T5Gemma2Tokenizer(
        proto=vocabulary_path
    )

    print("-> Print MD5 checksum of the vocab file.")
    print(
        f"`{vocabulary_path}` md5sum: ",
        get_md5_checksum(vocabulary_path),
    )

    return keras_hub_tokenizer


def check_text_output(
    keras_hub_tokenizer,
    keras_hub_model,
    hf_tokenizer,
    hf_model,
    preprocessor,
):
    """Check outputs of KerasHub and HuggingFace models match."""
    # Note: KerasHub counts encoder + decoder embeddings as separate
    # weight matrices. HF shares a single nn.Embedding across
    # encoder/decoder/lm_head, so counts it once.
    print("\n--- Model Verification starts ---")
    print("\n")
    print("\n-> Verify parameter counts.")
    keras_hub_params = keras_hub_model.count_params()
    hf_params = hf_model.num_parameters()
    print(f"KerasHub params: {keras_hub_params:,}")
    print(f"HF params:       {hf_params:,}")
    if keras_hub_params == hf_params:
        print("-> Parameter counts match!")
    else:
        diff = keras_hub_params - hf_params
        print(
            f"-> Parameter count difference: {diff:,} "
            f"(expected — KerasHub has separate encoder/decoder "
            f"embeddings; HF shares a single nn.Embedding)"
        )

    # Output comparison.
    print("\n-> ---- Text-only verification. ----\n")
    enc_sample_text = [
        "cricket is awesome, easily the best sport in the world!"
    ]
    dec_sample_text = [
        "football is good too, but nowhere near as good as cricket."
    ]

    # KerasHub — build unpadded inputs (match HF's natural lengths).
    keras_hub_enc_token_ids = hf_tokenizer(
        enc_sample_text, return_tensors="np"
    )["input_ids"]
    keras_hub_dec_token_ids = hf_tokenizer(
        dec_sample_text, return_tensors="np"
    )["input_ids"]
    keras_hub_dec_token_ids = np.concatenate(
        [
            np.array([[keras_hub_tokenizer.start_token_id]]),
            keras_hub_dec_token_ids,
        ],
        axis=-1,
    )
    keras_hub_inputs = {
        "encoder_token_ids": keras_hub_enc_token_ids,
        "encoder_padding_mask": np.ones_like(keras_hub_enc_token_ids),
        "decoder_token_ids": keras_hub_dec_token_ids,
        "decoder_padding_mask": np.ones_like(keras_hub_dec_token_ids),
    }
    # For multimodal backbones, use preprocessor to add dummy
    # image/vision_indices (the single source of truth).
    keras_hub_inputs = preprocessor._add_vision_inputs(
        keras_hub_inputs, batch_size=1
    )
    keras_hub_output = keras_hub_model.predict(keras_hub_inputs)

    # HF.
    hf_enc_inputs = hf_tokenizer(enc_sample_text, return_tensors="pt")
    hf_dec_inputs = hf_tokenizer(dec_sample_text, return_tensors="pt")
    hf_decoder_input_ids = torch.cat(
        [
            torch.tensor([[hf_tokenizer.bos_token_id]]),
            hf_dec_inputs["input_ids"],
        ],
        dim=-1,
    )
    hf_decoder_attention_mask = torch.cat(
        [
            torch.ones(1, 1, dtype=torch.long),
            hf_dec_inputs["attention_mask"],
        ],
        dim=-1,
    )

    hf_output = hf_model(
        **hf_enc_inputs,
        decoder_input_ids=hf_decoder_input_ids,
        decoder_attention_mask=hf_decoder_attention_mask,
    )

    # Encoder output comparison.
    keras_enc_out = keras_hub_output["encoder_sequence_output"]
    hf_enc_out = hf_output.encoder_last_hidden_state.detach().float().numpy()
    enc_abs_diff = np.abs(keras_enc_out - hf_enc_out)
    print()
    print("Encoder Outputs:")
    print("KerasHub output:", keras_enc_out[0, 0, :10])
    print("HF output:", hf_enc_out[0, 0, :10])
    print(f"Mean absolute diff: {enc_abs_diff.mean():.6f}")
    try:
        np.testing.assert_allclose(
            keras_enc_out, hf_enc_out, rtol=1e-4, atol=1e-4
        )
        print("-> Encoder outputs match! (rtol=1e-4, atol=1e-4)")
    except AssertionError:
        mismatch = np.sum(
            ~np.isclose(keras_enc_out, hf_enc_out, rtol=1e-4, atol=1e-4)
        )
        total = keras_enc_out.size
        print(
            f"-> Encoder outputs differ slightly beyond rtol=1e-4 "
            f"(mismatched: {mismatch}/{total}, "
            f"{mismatch / total * 100:.2f}%)"
        )

    # Decoder output comparison.
    keras_dec_out = keras_hub_output["decoder_sequence_output"]
    hf_dec_out = hf_output.last_hidden_state.detach().float().numpy()
    dec_abs_diff = np.abs(keras_dec_out - hf_dec_out)
    print()
    print("Decoder Outputs:")
    print("KerasHub output:", keras_dec_out[0, 0, :10])
    print("HF output:", hf_dec_out[0, 0, :10])
    print(f"Mean absolute diff: {dec_abs_diff.mean():.6f}")
    try:
        np.testing.assert_allclose(
            keras_dec_out, hf_dec_out, rtol=1e-4, atol=1e-4
        )
        print("-> Decoder outputs match! (rtol=1e-4, atol=1e-4)")
    except AssertionError:
        mismatch = np.sum(
            ~np.isclose(keras_dec_out, hf_dec_out, rtol=1e-4, atol=1e-4)
        )
        total = keras_dec_out.size
        print(
            f"-> Decoder outputs differ slightly beyond rtol=1e-4 "
            f"(mismatched: {mismatch}/{total}, "
            f"{mismatch / total * 100:.2f}%)"
        )


def check_multimodal_output(
    keras_hub_model,
    hf_model,
    hf_model_dir,
    hf_tokenizer,
):
    """Check multimodal (text+image) outputs match between KerasHub and HF."""
    if keras_hub_model.vision_encoder is None:
        print("\n-> Skipping multimodal check (text-only model).")
        return

    print("\n-> ---- Multimodal (text+image) verification. ----\n")

    # Download a test image.
    image_url = (
        "https://huggingface.co/datasets/huggingface/"
        "documentation-images/resolve/main/bee.jpg"
    )
    print(f"  Downloading test image: {image_url}")
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    # HF side: use AutoProcessor for proper multimodal preprocessing.
    hf_processor = transformers.AutoProcessor.from_pretrained(hf_model_dir)
    enc_prompt = "<start_of_image> Describe this image"
    dec_prompt = "This image shows"

    # HF encoder inputs (with image).
    hf_enc_inputs = hf_processor(
        text=enc_prompt, images=image, return_tensors="pt"
    )
    # HF decoder inputs.
    hf_dec_inputs = hf_tokenizer(dec_prompt, return_tensors="pt")
    hf_decoder_input_ids = torch.cat(
        [
            torch.tensor([[hf_tokenizer.bos_token_id]]),
            hf_dec_inputs["input_ids"],
        ],
        dim=-1,
    )
    hf_decoder_attention_mask = torch.cat(
        [
            torch.ones(1, 1, dtype=torch.long),
            hf_dec_inputs["attention_mask"],
        ],
        dim=-1,
    )

    with torch.no_grad():
        hf_output = hf_model(
            input_ids=hf_enc_inputs["input_ids"],
            attention_mask=hf_enc_inputs["attention_mask"],
            pixel_values=hf_enc_inputs["pixel_values"],
            decoder_input_ids=hf_decoder_input_ids,
            decoder_attention_mask=hf_decoder_attention_mask,
        )

    # Build KerasHub inputs from HF token_ids (same tokenizer).
    keras_enc_token_ids = hf_enc_inputs["input_ids"].numpy()
    keras_enc_padding_mask = hf_enc_inputs["attention_mask"].numpy()
    keras_dec_token_ids = hf_decoder_input_ids.numpy()
    keras_dec_padding_mask = hf_decoder_attention_mask.numpy()

    # Transpose HF pixel_values (B,C,H,W) to KerasHub (B,1,H,W,C).
    pixel_values = hf_enc_inputs["pixel_values"].numpy()
    if pixel_values.ndim == 5:
        pixel_values = np.transpose(pixel_values, (0, 1, 3, 4, 2))
    elif pixel_values.ndim == 4:
        pixel_values = np.transpose(pixel_values, (0, 2, 3, 1))
        pixel_values = np.expand_dims(pixel_values, axis=1)

    # Find positions of image placeholder tokens for vision_indices.
    image_token_id = hf_processor.tokenizer.convert_tokens_to_ids(
        "<image_soft_token>"
    )
    num_vision_tokens = (
        keras_hub_model.vision_encoder.num_vision_tokens_per_image
    )
    # Find indices of image placeholder tokens.
    token_ids_flat = keras_enc_token_ids[0]
    vision_idx_list = np.where(token_ids_flat == image_token_id)[0].tolist()

    # Pad or truncate to num_vision_tokens.
    if len(vision_idx_list) < num_vision_tokens:
        vision_idx_list = vision_idx_list + [0] * (
            num_vision_tokens - len(vision_idx_list)
        )
    vision_indices = np.array(
        [vision_idx_list[:num_vision_tokens]], dtype="int32"
    )

    keras_hub_inputs = {
        "encoder_token_ids": keras_enc_token_ids,
        "encoder_padding_mask": keras_enc_padding_mask,
        "decoder_token_ids": keras_dec_token_ids,
        "decoder_padding_mask": keras_dec_padding_mask,
        "images": pixel_values.astype("float32"),
        "vision_indices": vision_indices,
    }

    print("\n--- Multimodal Verification ---")
    keras_hub_output = keras_hub_model.predict(keras_hub_inputs)

    # Encoder output comparison.
    keras_enc_out = keras_hub_output["encoder_sequence_output"]
    hf_enc_out = hf_output.encoder_last_hidden_state.detach().float().numpy()
    enc_abs_diff = np.abs(keras_enc_out - hf_enc_out)
    print()
    print("Encoder Outputs (multimodal):")
    print("KerasHub output:", keras_enc_out[0, 0, :10])
    print("HF output:", hf_enc_out[0, 0, :10])
    print(f"Mean absolute diff: {enc_abs_diff.mean():.6f}")
    try:
        np.testing.assert_allclose(
            keras_enc_out, hf_enc_out, rtol=1e-4, atol=1e-4
        )
        print("-> Encoder outputs match! (rtol=1e-4, atol=1e-4)")
    except AssertionError:
        mismatch = np.sum(
            ~np.isclose(keras_enc_out, hf_enc_out, rtol=1e-4, atol=1e-4)
        )
        total = keras_enc_out.size
        print(
            f"-> Encoder outputs differ slightly beyond rtol=1e-4 "
            f"(mismatched: {mismatch}/{total}, "
            f"{mismatch / total * 100:.2f}%)"
        )

    # Decoder output comparison.
    keras_dec_out = keras_hub_output["decoder_sequence_output"]
    hf_dec_out = hf_output.last_hidden_state.detach().float().numpy()
    dec_abs_diff = np.abs(keras_dec_out - hf_dec_out)
    print()
    print("Decoder Outputs (multimodal):")
    print("KerasHub output:", keras_dec_out[0, 0, :10])
    print("HF output:", hf_dec_out[0, 0, :10])
    print(f"Mean absolute diff: {dec_abs_diff.mean():.6f}")
    try:
        np.testing.assert_allclose(
            keras_dec_out, hf_dec_out, rtol=1e-4, atol=1e-4
        )
        print("-> Decoder outputs match! (rtol=1e-4, atol=1e-4)")
    except AssertionError:
        mismatch = np.sum(
            ~np.isclose(keras_dec_out, hf_dec_out, rtol=1e-4, atol=1e-4)
        )
        total = keras_dec_out.size
        print(
            f"-> Decoder outputs differ slightly beyond rtol=1e-4 "
            f"(mismatched: {mismatch}/{total}, "
            f"{mismatch / total * 100:.2f}%)"
        )


def main(_):
    os.makedirs(FLAGS.preset, exist_ok=True)

    hf_model_name = PRESET_MAP[FLAGS.preset]

    print("\n-> Download HF model files.")
    hf_model_dir = huggingface_hub.snapshot_download(
        repo_id=hf_model_name,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "tokenizer.model",
        ],
    )

    print("\n-> Load HF model and HF tokenizer.")
    hf_model = transformers.AutoModel.from_pretrained(hf_model_dir)
    hf_model.float()  # Convert all params/buffers to float32.
    # Re-create embed_scale with true f32 precision (bf16-init artifact).
    enc_hdim = hf_model.config.encoder.text_config.hidden_size
    dec_hdim = hf_model.config.decoder.hidden_size
    hf_model.encoder.text_model.embed_tokens.embed_scale = torch.tensor(
        enc_hdim**0.5, dtype=torch.float32
    )
    hf_model.decoder.embed_tokens.embed_scale = torch.tensor(
        dec_hdim**0.5, dtype=torch.float32
    )
    hf_model.eval()
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_dir)

    keras_hub_model = convert_checkpoints(hf_model)
    print("\n-> Load KerasHub tokenizer.")
    keras_hub_tokenizer = extract_vocab(hf_model_dir)

    # Create preprocessor to check_text_output can use it.
    preprocessor_kwargs = {}
    if keras_hub_model.vision_encoder is not None:
        preprocessor_kwargs.update(
            {
                "image_size": keras_hub_model.vision_encoder.image_size,
                "num_vision_tokens_per_image": (
                    keras_hub_model.vision_encoder.num_vision_tokens_per_image
                ),
            }
        )
    preprocessor = T5Gemma2Seq2SeqLMPreprocessor(
        tokenizer=keras_hub_tokenizer,
        encoder_sequence_length=512,
        decoder_sequence_length=512,
        **preprocessor_kwargs,
    )

    check_text_output(
        keras_hub_tokenizer,
        keras_hub_model,
        hf_tokenizer,
        hf_model,
        preprocessor,
    )

    check_multimodal_output(
        keras_hub_model,
        hf_model,
        hf_model_dir,
        hf_tokenizer,
    )

    print("\n-> Releasing HF backbone from memory.")
    del hf_model
    gc.collect()

    keras_lm = T5Gemma2Seq2SeqLM(
        backbone=keras_hub_model,
        preprocessor=preprocessor,
        dtype=keras_hub_model.dtype,
    )

    print(f"\n-> Saving T5Gemma2Seq2SeqLM preset to `{FLAGS.preset}`.")
    keras_lm.save_to_preset(FLAGS.preset)
    print("-> Preset saved successfully.")

    print("\n-> Testing preset loading.")
    keras_lm = Seq2SeqLM.from_preset(FLAGS.preset)
    print("-> Preset loading verified successfully.")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
