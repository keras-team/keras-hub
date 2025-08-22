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
from keras_hub.src.models.t5gemma.t5gemma_seq_2_seq_lm import T5GemmaSeq2SeqLM
from keras_hub.src.models.t5gemma.t5gemma_seq_2_seq_lm_preprocessor import (
    T5GemmaSeq2SeqLMPreprocessor,
)

random.seed(123)
torch.manual_seed(123)
device = torch.device("cpu")
# Force PyTorch to use CPU
torch.set_default_device(device)

PRESET_MAP = {
    "t5gemma_s_s_ul2": "google/t5gemma-s-s-ul2",
    "t5gemma_s_s_prefixlm": "google/t5gemma-s-s-prefixlm",
    "t5gemma_s_s_ul2_it": "google/t5gemma-s-s-ul2-it",
    "t5gemma_s_s_prefixlm_it": "google/t5gemma-s-s-prefixlm-it",
    "t5gemma_b_b_ul2": "google/t5gemma-b-b-ul2",
    "t5gemma_b_b_prefixlm": "google/t5gemma-b-b-prefixlm",
    "t5gemma_b_b_ul2_it": "google/t5gemma-b-b-ul2-it",
    "t5gemma_b_b_prefixlm_it": "google/t5gemma-b-b-prefixlm-it",
    "t5gemma_l_l_ul2": "google/t5gemma-l-l-ul2",
    "t5gemma_l_l_prefixlm": "google/t5gemma-l-l-prefixlm",
    "t5gemma_l_l_ul2_it": "google/t5gemma-l-l-ul2-it",
    "t5gemma_l_l_prefixlm_it": "google/t5gemma-l-l-prefixlm-it",
    "t5gemma_ml_ml_ul2": "google/t5gemma-ml-ml-ul2",
    "t5gemma_ml_ml_prefixlm": "google/t5gemma-ml-ml-prefixlm",
    "t5gemma_ml_ml_ul2_it": "google/t5gemma-ml-ml-ul2-it",
    "t5gemma_ml_ml_prefixlm_it": "google/t5gemma-ml-ml-prefixlm-it",
    "t5gemma_xl_xl_ul2": "google/t5gemma-xl-xl-ul2",
    "t5gemma_xl_xl_prefixlm": "google/t5gemma-xl-xl-prefixlm",
    "t5gemma_xl_xl_ul2_it": "google/t5gemma-xl-xl-ul2-it",
    "t5gemma_xl_xl_prefixlm_it": "google/t5gemma-xl-xl-prefixlm-it",
    "t5gemma_2b_2b_ul2": "google/t5gemma-2b-2b-ul2",
    "t5gemma_2b_2b_prefixlm": "google/t5gemma-2b-2b-prefixlm",
    "t5gemma_2b_2b_ul2_it": "google/t5gemma-2b-2b-ul2-it",
    "t5gemma_2b_2b_prefixlm_it": "google/t5gemma-2b-2b-prefixlm-it",
    "t5gemma_9b_2b_ul2": "google/t5gemma-9b-2b-ul2",
    "t5gemma_9b_2b_prefixlm": "google/t5gemma-9b-2b-prefixlm",
    "t5gemma_9b_2b_ul2_it": "google/t5gemma-9b-2b-ul2-it",
    "t5gemma_9b_2b_prefixlm_it": "google/t5gemma-9b-2b-prefixlm-it",
    "t5gemma_9b_9b_ul2": "google/t5gemma-9b-9b-ul2",
    "t5gemma_9b_9b_prefixlm": "google/t5gemma-9b-9b-prefixlm",
    "t5gemma_9b_9b_ul2_it": "google/t5gemma-9b-9b-ul2-it",
    "t5gemma_9b_9b_prefixlm_it": "google/t5gemma-9b-9b-prefixlm-it",
}


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def convert_checkpoints(hf_model):
    """Convert Hugging Face weights to Keras Hub format."""
    print("\n-> Convert original weights to KerasHub format.")

    print("\n-> Load KerasHub model.")
    encoder_config = hf_model.config.encoder
    decoder_config = hf_model.config.decoder
    if decoder_config.hidden_activation == "gelu_pytorch_tanh":
        decoder_config.hidden_activation = "gelu_approximate"
    if encoder_config.hidden_activation == "gelu_pytorch_tanh":
        encoder_config.hidden_activation = "gelu_approximate"
    keras.config.set_floatx("float32")
    keras_hub_model = keras_hub.models.T5GemmaBackbone(
        vocabulary_size=decoder_config.vocab_size,
        encoder_hidden_dim=encoder_config.hidden_size,
        encoder_intermediate_dim=encoder_config.intermediate_size,
        encoder_num_layers=encoder_config.num_hidden_layers,
        encoder_num_attention_heads=encoder_config.num_attention_heads,
        encoder_num_key_value_heads=encoder_config.num_key_value_heads,
        encoder_head_dim=encoder_config.head_dim,
        encoder_layer_types=encoder_config.layer_types,
        decoder_hidden_dim=decoder_config.hidden_size,
        decoder_intermediate_dim=decoder_config.intermediate_size,
        decoder_num_layers=decoder_config.num_hidden_layers,
        decoder_num_attention_heads=decoder_config.num_attention_heads,
        decoder_num_key_value_heads=decoder_config.num_key_value_heads,
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
        dtype="float32",
    )

    hf_wts = hf_model.state_dict()
    # Token embedding.
    keras_hub_model.get_layer("encoder_token_embedding").embeddings.assign(
        hf_wts["encoder.embed_tokens.weight"]
    )
    keras_hub_model.get_layer("decoder_token_embedding").embeddings.assign(
        hf_wts["decoder.embed_tokens.weight"]
    )

    # Encoder.
    encoder_hidden_dim = keras_hub_model.encoder_hidden_dim
    encoder_num_attention_heads = keras_hub_model.encoder_num_attention_heads
    encoder_num_key_value_heads = keras_hub_model.encoder_num_key_value_heads
    encoder_head_dim = keras_hub_model.encoder_head_dim
    keras_hub_model.encoder_norm.scale.assign(hf_wts["encoder.norm.weight"])

    for i in range(keras_hub_model.encoder_num_layers):
        encoder_layer = keras_hub_model.get_layer(f"encoder_layer_{i}")
        hf_prefix = f"encoder.layers.{i}"

        # Self-attention.
        q_w = hf_wts[f"{hf_prefix}.self_attn.q_proj.weight"]
        k_w = hf_wts[f"{hf_prefix}.self_attn.k_proj.weight"]
        v_w = hf_wts[f"{hf_prefix}.self_attn.v_proj.weight"]
        o_w = hf_wts[f"{hf_prefix}.self_attn.o_proj.weight"]

        encoder_layer.self_attn.query_dense.kernel.assign(
            q_w.T.reshape(
                encoder_hidden_dim,
                encoder_num_attention_heads,
                encoder_head_dim,
            ).numpy()
        )
        encoder_layer.self_attn.key_dense.kernel.assign(
            k_w.T.reshape(
                encoder_hidden_dim,
                encoder_num_key_value_heads,
                encoder_head_dim,
            ).numpy()
        )
        encoder_layer.self_attn.value_dense.kernel.assign(
            v_w.T.reshape(
                encoder_hidden_dim,
                encoder_num_key_value_heads,
                encoder_head_dim,
            ).numpy()
        )
        encoder_layer.self_attn.output_dense.kernel.assign(
            o_w.T.reshape(
                encoder_num_attention_heads,
                encoder_head_dim,
                encoder_hidden_dim,
            ).numpy()
        )

        # MLP.
        encoder_layer.mlp.gate_proj.kernel.assign(
            hf_wts[f"{hf_prefix}.mlp.gate_proj.weight"].T.numpy()
        )
        encoder_layer.mlp.up_proj.kernel.assign(
            hf_wts[f"{hf_prefix}.mlp.up_proj.weight"].T.numpy()
        )
        encoder_layer.mlp.down_proj.kernel.assign(
            hf_wts[f"{hf_prefix}.mlp.down_proj.weight"].T.numpy()
        )

        # Layer norm.
        encoder_layer.pre_self_attn_layernorm.scale.assign(
            hf_wts[f"{hf_prefix}.pre_self_attn_layernorm.weight"]
        )
        encoder_layer.post_self_attn_layernorm.scale.assign(
            hf_wts[f"{hf_prefix}.post_self_attn_layernorm.weight"]
        )
        encoder_layer.pre_feedforward_layernorm.scale.assign(
            hf_wts[f"{hf_prefix}.pre_feedforward_layernorm.weight"]
        )
        encoder_layer.post_feedforward_layernorm.scale.assign(
            hf_wts[f"{hf_prefix}.post_feedforward_layernorm.weight"]
        )

    # Decoder.
    decoder_hidden_dim = keras_hub_model.decoder_hidden_dim
    decoder_num_attention_heads = keras_hub_model.decoder_num_attention_heads
    decoder_num_key_value_heads = keras_hub_model.decoder_num_key_value_heads
    decoder_head_dim = keras_hub_model.decoder_head_dim
    cross_attention_hidden_size = keras_hub_model.cross_attention_hidden_size
    keras_hub_model.decoder_norm.scale.assign(hf_wts["decoder.norm.weight"])

    for i in range(keras_hub_model.decoder_num_layers):
        decoder_layer = keras_hub_model.get_layer(f"decoder_layer_{i}")
        hf_prefix = f"decoder.layers.{i}"

        # Self-attention.
        q_w = hf_wts[f"{hf_prefix}.self_attn.q_proj.weight"]
        k_w = hf_wts[f"{hf_prefix}.self_attn.k_proj.weight"]
        v_w = hf_wts[f"{hf_prefix}.self_attn.v_proj.weight"]
        o_w = hf_wts[f"{hf_prefix}.self_attn.o_proj.weight"]
        decoder_layer.self_attn.query_dense.kernel.assign(
            q_w.T.reshape(
                decoder_hidden_dim,
                decoder_num_attention_heads,
                decoder_head_dim,
            ).numpy()
        )
        decoder_layer.self_attn.key_dense.kernel.assign(
            k_w.T.reshape(
                decoder_hidden_dim,
                decoder_num_key_value_heads,
                decoder_head_dim,
            ).numpy()
        )
        decoder_layer.self_attn.value_dense.kernel.assign(
            v_w.T.reshape(
                decoder_hidden_dim,
                decoder_num_key_value_heads,
                decoder_head_dim,
            ).numpy()
        )
        decoder_layer.self_attn.output_dense.kernel.assign(
            o_w.T.reshape(
                decoder_num_attention_heads,
                decoder_head_dim,
                decoder_hidden_dim,
            ).numpy()
        )

        # Cross-attention.
        q_w = hf_wts[f"{hf_prefix}.cross_attn.q_proj.weight"]
        k_w = hf_wts[f"{hf_prefix}.cross_attn.k_proj.weight"]
        v_w = hf_wts[f"{hf_prefix}.cross_attn.v_proj.weight"]
        o_w = hf_wts[f"{hf_prefix}.cross_attn.o_proj.weight"]
        decoder_layer.cross_attn.query_dense.kernel.assign(
            q_w.T.reshape(
                decoder_hidden_dim,
                decoder_num_attention_heads,
                decoder_head_dim,
            ).numpy()
        )
        decoder_layer.cross_attn.key_dense.kernel.assign(
            k_w.T.reshape(
                cross_attention_hidden_size,
                decoder_num_key_value_heads,
                decoder_head_dim,
            ).numpy()
        )
        decoder_layer.cross_attn.value_dense.kernel.assign(
            v_w.T.reshape(
                cross_attention_hidden_size,
                decoder_num_key_value_heads,
                decoder_head_dim,
            ).numpy()
        )
        decoder_layer.cross_attn.output_dense.kernel.assign(
            o_w.T.reshape(
                decoder_num_attention_heads,
                decoder_head_dim,
                decoder_hidden_dim,
            ).numpy()
        )

        # MLP.
        decoder_layer.mlp.gate_proj.kernel.assign(
            hf_wts[f"{hf_prefix}.mlp.gate_proj.weight"].T.numpy()
        )
        decoder_layer.mlp.up_proj.kernel.assign(
            hf_wts[f"{hf_prefix}.mlp.up_proj.weight"].T.numpy()
        )
        decoder_layer.mlp.down_proj.kernel.assign(
            hf_wts[f"{hf_prefix}.mlp.down_proj.weight"].T.numpy()
        )

        # Layer norm.
        decoder_layer.pre_self_attn_layernorm.scale.assign(
            hf_wts[f"{hf_prefix}.pre_self_attn_layernorm.weight"]
        )
        decoder_layer.post_self_attn_layernorm.scale.assign(
            hf_wts[f"{hf_prefix}.post_self_attn_layernorm.weight"]
        )
        decoder_layer.pre_cross_attn_layernorm.scale.assign(
            hf_wts[f"{hf_prefix}.pre_cross_attn_layernorm.weight"]
        )
        decoder_layer.post_cross_attn_layernorm.scale.assign(
            hf_wts[f"{hf_prefix}.post_cross_attn_layernorm.weight"]
        )
        decoder_layer.pre_feedforward_layernorm.scale.assign(
            hf_wts[f"{hf_prefix}.pre_feedforward_layernorm.weight"]
        )
        decoder_layer.post_feedforward_layernorm.scale.assign(
            hf_wts[f"{hf_prefix}.post_feedforward_layernorm.weight"]
        )

    return keras_hub_model


def extract_vocab(hf_model_dir):
    """Extract vocabulary from the downloaded Hugging Face model directory."""
    source_path = os.path.join(hf_model_dir, "tokenizer.model")
    vocabulary_path = os.path.join(FLAGS.preset, "tokenizer.model")
    print(f"\n-> Save KerasHub vocab to `{vocabulary_path}`.")

    shutil.copyfile(source_path, vocabulary_path)

    keras_hub_tokenizer = keras_hub.models.T5GemmaTokenizer(
        proto=vocabulary_path
    )

    print("-> Print MD5 checksum of the vocab file.")
    print(f"`{vocabulary_path}` md5sum: ", get_md5_checksum(vocabulary_path))

    return keras_hub_tokenizer


def check_output(
    keras_hub_tokenizer,
    keras_hub_model,
    hf_tokenizer,
    hf_model,
):
    """Check the outputs of the Keras Hub and Hugging Face models."""
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
        [torch.ones(1, 1, dtype=torch.long), hf_dec_inputs["attention_mask"]],
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
    print("HF output:", hf_output.encoder_last_hidden_state[0, 0, :10])
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
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
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
    preprocessor = T5GemmaSeq2SeqLMPreprocessor(
        tokenizer=keras_hub_tokenizer,
        encoder_sequence_length=512,
        decoder_sequence_length=512,
    )
    keras_lm = T5GemmaSeq2SeqLM(
        backbone=keras_hub_model,
        preprocessor=preprocessor,
        dtype=keras_hub_model.dtype,
    )
    keras_lm.compile(sampler="greedy")

    print(f"\n-> Saving T5GemmaSeq2SeqLM preset to `{FLAGS.preset}`.")
    keras_lm.save_to_preset(FLAGS.preset)
    print("-> Preset saved successfully.")

    print("\n-> Testing preset loading.")
    keras_lm = Seq2SeqLM.from_preset("t5gemma_b_b_prefixlm_it")
    print("-> Preset loading verified successfully.")

    # Show the MD5 checksum of the model weights after saving.
    print("\n-> Print MD5 checksum of the model weights.")
    weights_path = os.path.join(FLAGS.preset, "model.weights.h5")
    print(f"`{weights_path}` md5sum: ", get_md5_checksum(weights_path))


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
