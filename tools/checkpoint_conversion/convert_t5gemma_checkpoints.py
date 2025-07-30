"""
T5Gemma weight conversion script.

This script converts checkpoints from a Hugging Face T5Gemma model to a
KerasHub T5Gemma model.

To run, first install the dependencies:
```
pip install keras-core keras-nlp tensorflow-text
pip install transformers huggingface-hub sentencepiece absl-py torch
```

Then, log in to Hugging Face:
```
huggingface-cli login
```

Finally, run the script to convert the weights:
```
python convert_t5gemma_checkpoints.py --preset t5gemma_b_b_prefixlm_it
```
"""

import os

import absl
import huggingface_hub
import numpy as np
import torch
import transformers

from keras_hub.src.models.t5gemma.t5gemma_causal_lm import T5GemmaCausalLM
from keras_hub.src.models.t5gemma.t5gemma_causal_lm_preprocessor import (
    T5GemmaCausalLMPreprocessor,
)
from keras_hub.src.models.t5gemma.t5gemma_tokenizer import T5GemmaTokenizer

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
    "t5gemma_9b_9b_ul2": "google/t5gemma-9b-9b-ul2",
    "t5gemma_9b_9b_prefixlm": "google/t5gemma-9b-9b-prefixlm",
    "t5gemma_9b_9b_ul2_it": "google/t5gemma-9b-9b-ul2-it",
    "t5gemma_9b_9b_prefixlm_it": "google/t5gemma-9b-9b-prefixlm-it",
}
EXTRACT_DIR = "./model_t5gemma"
FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string(
    "preset",
    "t5gemma_b_b_prefixlm_it",
    f"Must be one of {','.join(PRESET_MAP.keys())}.",
)


def download_hf_model(hf_model_name):
    print(f"⬇️ Downloading Hugging Face model '{hf_model_name}'...")
    hf_model_dir = huggingface_hub.snapshot_download(
        repo_id=hf_model_name,
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
        local_dir=EXTRACT_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"✅ Model downloaded to: {hf_model_dir}")
    return hf_model_dir


def convert_model(hf_model, preprocessor):
    encoder_config = hf_model.config.encoder
    decoder_config = hf_model.config.decoder
    if decoder_config.hidden_activation == "gelu_pytorch_tanh":
        decoder_config.hidden_activation = "gelu_approximate"
    if encoder_config.hidden_activation == "gelu_pytorch_tanh":
        encoder_config.hidden_activation = "gelu_approximate"
    keras_backbone = T5GemmaCausalLM.backbone_cls(
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
    )
    keras_model = T5GemmaCausalLM(
        backbone=keras_backbone, preprocessor=preprocessor
    )
    print("✅ Keras model instantiated.")
    return keras_model


def convert_tokenizer(hf_model_dir):
    print("🗣️ Converting tokenizer...")
    tokenizer_path = os.path.join(hf_model_dir, "tokenizer.model")
    keras_tokenizer = T5GemmaTokenizer(proto=tokenizer_path)
    print("✅ Tokenizer converted.")
    return keras_tokenizer


def convert_weights(keras_model, hf_model):
    print("🏋️ Converting weights...")
    hf_wts = hf_model.state_dict()
    keras_backbone = keras_model.backbone
    # Token Embeddings.
    keras_backbone.token_embedding.embeddings.assign(
        hf_wts["encoder.embed_tokens.weight"]
    )
    keras_backbone.decoder_token_embedding.embeddings.assign(
        hf_wts["decoder.embed_tokens.weight"]
    )

    # Encoder.
    encoder_hidden_dim = keras_backbone.encoder_hidden_dim
    encoder_num_attention_heads = keras_backbone.encoder_num_attention_heads
    encoder_num_key_value_heads = keras_backbone.encoder_num_key_value_heads
    encoder_head_dim = keras_backbone.encoder_head_dim
    keras_backbone.encoder_norm.scale.assign(hf_wts["encoder.norm.weight"])
    for i in range(keras_backbone.encoder_num_layers):
        encoder_layer = keras_backbone.get_layer(f"encoder_layer_{i}")
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
    decoder_hidden_dim = keras_backbone.decoder_hidden_dim
    decoder_num_attention_heads = keras_backbone.decoder_num_attention_heads
    decoder_num_key_value_heads = keras_backbone.decoder_num_key_value_heads
    decoder_head_dim = keras_backbone.decoder_head_dim
    cross_attention_hidden_size = keras_backbone.cross_attention_hidden_size
    keras_backbone.decoder_norm.scale.assign(hf_wts["decoder.norm.weight"])
    for i in range(keras_backbone.decoder_num_layers):
        decoder_layer = keras_backbone.get_layer(f"decoder_layer_{i}")
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
    print("✅ Weights converted.")


def validate_output(hf_model, keras_model, hf_tokenizer, keras_tokenizer):
    hf_model.eval()
    print("🔎 Validating tokenizer outputs...")
    # Example sentence.
    test_sentence = "What is the fastest land animal?"
    hf_tokens = hf_tokenizer(test_sentence, return_tensors="pt")["input_ids"][
        0
    ].tolist()
    keras_tokens = keras_tokenizer.tokenize(test_sentence).numpy().tolist()
    print(f"🔶 Test Sentence: '{test_sentence}'")
    print(f"🔶 Hugging Face Tokens: {hf_tokens}")
    print(f"🔶 Keras Tokens:        {keras_tokens}")
    assert hf_tokens == keras_tokens, "Tokenizer outputs do not match!"
    print("✅ Tokenizer outputs are consistent.")
    print("🔎 Validating numeric outputs...")
    input_ids_np = np.ones((1, 10), dtype="int32")
    attention_mask_np = np.ones((1, 10), dtype="int32")
    keras_inputs = {
        "token_ids": input_ids_np,
        "padding_mask": attention_mask_np,
    }
    hf_input_ids = torch.from_numpy(input_ids_np)
    hf_attention_mask = torch.from_numpy(attention_mask_np)
    hf_decoder_input_ids = hf_input_ids.clone()
    hf_outputs = hf_model(
        input_ids=hf_input_ids,
        attention_mask=hf_attention_mask,
        decoder_input_ids=hf_decoder_input_ids,
    )
    hf_final_hidden_states = hf_outputs.last_hidden_state.detach().numpy()
    print("\n🔎 Validating final hidden states...")
    keras_final_hidden_states = keras_model.backbone.predict(keras_inputs)
    final_difference = np.mean(
        np.abs(hf_final_hidden_states - keras_final_hidden_states)
    )
    print(f"🔶 Keras final output shape: {keras_final_hidden_states.shape}")
    print(f"🔶 HF final output shape:    {hf_final_hidden_states.shape}")
    print(f"🔶 Mean absolute difference: {final_difference:.6e}")
    assert final_difference < 1e-4, "Final output difference is too high!"
    print("✅ Final hidden states are consistent.")


def main(_):
    preset = FLAGS.preset
    print(f"🚀 Starting conversion for preset: {preset}")

    hf_model_name = PRESET_MAP[preset]
    hf_model_dir = download_hf_model(hf_model_name)

    print("🧩 Loading Hugging Face model and tokenizer...")
    hf_model = transformers.T5GemmaModel.from_pretrained(hf_model_dir)
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_dir)
    print("✅ Hugging Face model and tokenizer loaded.")

    keras_tokenizer = convert_tokenizer(hf_model_dir)

    keras_preprocessor = T5GemmaCausalLMPreprocessor(
        tokenizer=keras_tokenizer,
    )
    keras_model = convert_model(hf_model, keras_preprocessor)
    convert_weights(keras_model, hf_model)
    validate_output(hf_model, keras_model, hf_tokenizer, keras_tokenizer)

    print(f"💾 Saving Keras model and tokenizer to preset '{preset}'...")
    keras_model.save_to_preset(preset)
    keras_tokenizer.save_to_preset(preset)
    print("✅ Preset saved successfully.")
    print("🎉 Conversion complete!")


if __name__ == "__main__":
    absl.app.run(main)
