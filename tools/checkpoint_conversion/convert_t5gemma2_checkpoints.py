import gc
import os
import random
import shutil

import huggingface_hub
import keras
import numpy as np
import tensorflow as tf
import torch
import transformers
from absl import app
from absl import flags
from checkpoint_conversion_utils import get_md5_checksum

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
    decoder_config = hf_model.config.decoder
    if decoder_config.hidden_activation == "gelu_pytorch_tanh":
        decoder_config.hidden_activation = "gelu_approximate"
    if encoder_config.hidden_activation == "gelu_pytorch_tanh":
        encoder_config.hidden_activation = "gelu_approximate"

    keras.config.set_floatx("float32")
    keras_hub_model = keras_hub.models.T5Gemma2Backbone(
        vocabulary_size=decoder_config.vocab_size,
        encoder_hidden_dim=encoder_config.hidden_size,
        encoder_intermediate_dim=encoder_config.intermediate_size,
        encoder_num_layers=encoder_config.num_hidden_layers,
        encoder_num_attention_heads=encoder_config.num_attention_heads,
        encoder_num_key_value_heads=(encoder_config.num_key_value_heads),
        encoder_head_dim=encoder_config.head_dim,
        encoder_layer_types=encoder_config.layer_types,
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
        cross_attention_hidden_size=encoder_config.hidden_size,
        attn_logit_softcapping=decoder_config.attn_logit_softcapping,
        final_logit_softcapping=decoder_config.final_logit_softcapping,
        rope_max_wavelength=decoder_config.rope_theta,
        use_query_key_norm=True,
        dtype="float32",
    )

    hf_wts = hf_model.state_dict()

    # Token embeddings.
    keras_hub_model.get_layer("encoder_token_embedding").embeddings.assign(
        hf_wts["encoder.embed_tokens.weight"]
    )
    keras_hub_model.get_layer("decoder_token_embedding").embeddings.assign(
        hf_wts["decoder.embed_tokens.weight"]
    )

    # Encoder.
    enc_hdim = keras_hub_model.encoder_hidden_dim
    enc_heads = keras_hub_model.encoder_num_attention_heads
    enc_kv_heads = keras_hub_model.encoder_num_key_value_heads
    enc_head_dim = keras_hub_model.encoder_head_dim
    keras_hub_model.encoder_norm.scale.assign(hf_wts["encoder.norm.weight"])

    for i in range(keras_hub_model.encoder_num_layers):
        layer = keras_hub_model.get_layer(f"encoder_layer_{i}")
        pfx = f"encoder.layers.{i}"

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
    source_path = os.path.join(hf_model_dir, "tokenizer.model")
    vocabulary_path = os.path.join(FLAGS.preset, "tokenizer.model")
    print(f"\n-> Save KerasHub vocab to `{vocabulary_path}`.")

    shutil.copyfile(source_path, vocabulary_path)

    keras_hub_tokenizer = keras_hub.models.T5Gemma2Tokenizer(
        proto=vocabulary_path
    )

    print("-> Print MD5 checksum of the vocab file.")
    print(
        f"`{vocabulary_path}` md5sum: ",
        get_md5_checksum(vocabulary_path),
    )

    return keras_hub_tokenizer


def check_output(
    keras_hub_tokenizer,
    keras_hub_model,
    hf_tokenizer,
    hf_model,
):
    """Check outputs of KerasHub and HuggingFace models match."""
    print("\n-> Check the outputs.")
    enc_sample_text = [
        "cricket is awesome, easily the best sport in the world!"
    ]
    dec_sample_text = [
        "football is good too, but nowhere near as good as cricket."
    ]

    # KerasHub.
    keras_hub_enc_token_ids = hf_tokenizer(
        enc_sample_text, return_tensors="tf"
    )["input_ids"]
    keras_hub_dec_token_ids = hf_tokenizer(
        dec_sample_text, return_tensors="tf"
    )["input_ids"]
    keras_hub_dec_token_ids = tf.concat(
        [
            tf.constant([[keras_hub_tokenizer.start_token_id]]),
            keras_hub_dec_token_ids,
        ],
        axis=-1,
    )
    keras_hub_inputs = {
        "encoder_token_ids": keras_hub_enc_token_ids,
        "encoder_padding_mask": tf.ones_like(keras_hub_enc_token_ids),
        "decoder_token_ids": keras_hub_dec_token_ids,
        "decoder_padding_mask": tf.ones_like(keras_hub_dec_token_ids),
    }
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

    print("Encoder Outputs:")
    print(
        "KerasHub output:",
        keras_hub_output["encoder_sequence_output"][0, 0, :10],
    )
    print(
        "HF output:",
        hf_output.encoder_last_hidden_state[0, 0, :10],
    )
    print(
        "Difference:",
        np.mean(
            keras_hub_output["encoder_sequence_output"]
            - hf_output.encoder_last_hidden_state.detach().numpy()
        ),
    )

    print("Decoder Outputs:")
    print(
        "KerasHub output:",
        keras_hub_output["decoder_sequence_output"][0, 0, :10],
    )
    print("HF output:", hf_output.last_hidden_state[0, 0, :10])
    print(
        "Difference:",
        np.mean(
            keras_hub_output["decoder_sequence_output"]
            - hf_output.last_hidden_state.detach().numpy()
        ),
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
    hf_model.eval()
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_dir)

    keras_hub_model = convert_checkpoints(hf_model)
    print("\n-> Load KerasHub tokenizer.")
    keras_hub_tokenizer = extract_vocab(hf_model_dir)

    check_output(
        keras_hub_tokenizer,
        keras_hub_model,
        hf_tokenizer,
        hf_model,
    )
    print("\n-> Releasing HF backbone from memory.")
    del hf_model
    gc.collect()

    preprocessor = T5Gemma2Seq2SeqLMPreprocessor(
        tokenizer=keras_hub_tokenizer,
        encoder_sequence_length=512,
        decoder_sequence_length=512,
    )
    keras_lm = T5Gemma2Seq2SeqLM(
        backbone=keras_hub_model,
        preprocessor=preprocessor,
        dtype=keras_hub_model.dtype,
    )
    keras_lm.compile(sampler="greedy")

    print(f"\n-> Saving T5Gemma2Seq2SeqLM preset to `{FLAGS.preset}`.")
    keras_lm.save_to_preset(FLAGS.preset)
    print("-> Preset saved successfully.")

    print("\n-> Testing preset loading.")
    keras_lm = Seq2SeqLM.from_preset(FLAGS.preset)
    print("-> Preset loading verified successfully.")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
