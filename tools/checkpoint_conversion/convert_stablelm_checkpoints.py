import json
import os
import traceback

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
import transformers
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from keras import ops
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from keras_hub.models import StableLMBackbone
from keras_hub.models import StableLMCausalLM
from keras_hub.models import StableLMCausalLMPreprocessor
from keras_hub.models import StableLMTokenizer

PRESET_MAP = {
    "stablelm_3b_4e1t_en": "stabilityai/stablelm-3b-4e1t",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def convert_checkpoints(keras_hub_model, hf_model):
    """Manually transfer weights from Hugging Face to KerasHub."""
    config = hf_model.config
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads

    # 1. Token Embeddings
    keras_hub_model.token_embedding.embeddings.assign(
        hf_model.model.embed_tokens.weight.detach().cpu().float().numpy()
    )

    # 2. Transformer Layers
    for i in range(keras_hub_model.num_layers):
        hf_layer = hf_model.model.layers[i]
        keras_layer = keras_hub_model.transformer_layers[i]

        # Query, Key, Value Projections
        for hf_proj, keras_dense, heads in [
            (
                hf_layer.self_attn.q_proj,
                keras_layer.self_attention_layer.query_dense,
                num_attention_heads,
            ),
            (
                hf_layer.self_attn.k_proj,
                keras_layer.self_attention_layer.key_dense,
                num_key_value_heads,
            ),
            (
                hf_layer.self_attn.v_proj,
                keras_layer.self_attention_layer.value_dense,
                num_key_value_heads,
            ),
        ]:
            weight = hf_proj.weight.detach().cpu().float().numpy().T
            weight = weight.reshape(hidden_size, heads, head_dim)
            weights = [weight]
            if getattr(hf_proj, "bias", None) is not None:
                weights.append(hf_proj.bias.detach().cpu().float().numpy())
            keras_dense.set_weights(weights)

        # Attention Output Projection
        o_weight = (
            hf_layer.self_attn.o_proj.weight.detach().cpu().float().numpy().T
        )
        o_weight = o_weight.reshape(num_attention_heads, head_dim, hidden_size)
        weights = [o_weight]
        if getattr(hf_layer.self_attn.o_proj, "bias", None) is not None:
            weights.append(
                hf_layer.self_attn.o_proj.bias.detach().cpu().float().numpy()
            )
        keras_layer.self_attention_layer.output_dense.set_weights(weights)

        # Pre-attention LayerNorm
        keras_layer.self_attention_layernorm.set_weights(
            [
                hf_layer.input_layernorm.weight.detach().cpu().float().numpy(),
                hf_layer.input_layernorm.bias.detach().cpu().float().numpy(),
            ]
        )

        # Feedforward Network
        for hf_proj, keras_dense in [
            (hf_layer.mlp.gate_proj, keras_layer.feedforward_gate_dense),
            (hf_layer.mlp.up_proj, keras_layer.feedforward_intermediate_dense),
            (hf_layer.mlp.down_proj, keras_layer.feedforward_output_dense),
        ]:
            weights = [hf_proj.weight.detach().cpu().float().numpy().T]
            if getattr(hf_proj, "bias", None) is not None:
                weights.append(hf_proj.bias.detach().cpu().float().numpy())
            keras_dense.set_weights(weights)

        # Post-attention LayerNorm
        keras_layer.feedforward_layernorm.set_weights(
            [
                hf_layer.post_attention_layernorm.weight.detach()
                .cpu()
                .float()
                .numpy(),
                hf_layer.post_attention_layernorm.bias.detach()
                .cpu()
                .float()
                .numpy(),
            ]
        )

    # 3. Final LayerNorm & LM Head
    keras_hub_model.layer_norm.set_weights(
        [
            hf_model.model.norm.weight.detach().cpu().float().numpy(),
            hf_model.model.norm.bias.detach().cpu().float().numpy(),
        ]
    )
    keras_hub_model.token_embedding.reverse_embeddings.assign(
        hf_model.lm_head.weight.T.detach().cpu().float().numpy()
    )


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    """Verify that tokenizer outputs align."""
    hf_output = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()

    keras_hub_preprocessor = StableLMCausalLMPreprocessor(keras_hub_tokenizer)
    keras_hub_output = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=6
    )
    keras_hub_output = ops.convert_to_numpy(keras_hub_output[0]["token_ids"])

    np.testing.assert_equal(keras_hub_output, hf_output)
    print("\n✅ Tokenizer outputs match perfectly!")


def test_model(
    keras_hub_model, keras_hub_tokenizer, hf_model, hf_model_tokenizer
):
    """Verify parameter counts and logit alignment."""
    keras_hub_params = keras_hub_model.count_params()
    hf_params = hf_model.num_parameters()
    assert keras_hub_params == hf_params

    hf_inputs = hf_model_tokenizer(["What is Keras?"], return_tensors="pt")
    hf_outputs = hf_model(**hf_inputs.to(hf_model.device))
    hf_output_logits = hf_outputs.logits.detach().cpu().float().numpy()

    keras_hub_preprocessor = StableLMCausalLMPreprocessor(keras_hub_tokenizer)
    keras_hub_inputs = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=6
    )[0]

    keras_hub_output = keras_hub_model(keras_hub_inputs)
    keras_hub_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits)

    try:
        np.testing.assert_allclose(
            keras_hub_logits, hf_output_logits, atol=1e-3
        )
        print("✅ Model outputs (Logits) match perfectly!")
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")


def validate_output(keras_model, hf_model, hf_tokenizer):
    """Verify end-to-end text generation parity."""
    input_str = "Table tennis is the best sport"
    length = 50

    # 1. Hugging Face Generation
    hf_inputs = hf_tokenizer([input_str], return_tensors="pt")
    hf_outputs = hf_model.generate(
        **hf_inputs.to(hf_model.device),
        max_length=length,
        do_sample=False,  # Greedy
        pad_token_id=hf_tokenizer.eos_token_id,
    )
    hf_generated_text = hf_tokenizer.batch_decode(
        hf_outputs, skip_special_tokens=True
    )[0]
    print(f"🔶 Hugging Face output: {hf_generated_text}")

    # 2. KerasHub Generation
    keras_model.compile(sampler="greedy")
    # Temporarily remove preprocessor to manually pad inputs for a strict 1:1
    # check
    original_preprocessor = keras_model.preprocessor
    keras_model.preprocessor = None

    prompt_length = hf_inputs["input_ids"].shape[1]
    padded_token_ids = (
        np.ones((1, length), dtype=np.int32) * hf_tokenizer.eos_token_id
    )
    padded_token_ids[0, :prompt_length] = (
        hf_inputs["input_ids"].detach().cpu().numpy()
    )

    padding_mask = np.zeros((1, length), dtype=bool)
    padding_mask[0, :prompt_length] = True

    keras_inputs = {
        "token_ids": ops.convert_to_tensor(padded_token_ids),
        "padding_mask": ops.convert_to_tensor(padding_mask),
    }

    keras_outputs = keras_model.generate(
        keras_inputs, stop_token_ids=[hf_tokenizer.eos_token_id]
    )
    keras_generated_ids = ops.convert_to_numpy(keras_outputs["token_ids"][0])
    keras_generated_text = hf_tokenizer.decode(
        keras_generated_ids, skip_special_tokens=True
    )
    print(f"🔶 KerasHub output:    {keras_generated_text}")

    keras_model.preprocessor = original_preprocessor

    if hf_generated_text == keras_generated_text:
        print("✅ End-to-End Generation matches perfectly!")
    else:
        print("❌ Generation mismatch!")


def main(_):
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    print("\n-> Loading Hugging Face models...")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_config = transformers.AutoConfig.from_pretrained(hf_preset)

    # Patch config dynamically for robust loading
    if not hasattr(hf_config, "pad_token_id") or hf_config.pad_token_id is None:
        hf_config.pad_token_id = hf_tokenizer.eos_token_id

    # Load in float32 and eager attention to ensure 1:1 parity with Keras
    # defaults
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_preset, config=hf_config, attn_implementation="eager"
    ).to(torch.float32)
    hf_model.eval()
    print("\n-> Hugging Face model and tokenizer loaded.")

    print("\n-> Creating KerasHub backbone and tokenizer...")
    keras_hub_backbone = StableLMBackbone(
        vocabulary_size=hf_model.config.vocab_size,
        hidden_dim=hf_model.config.hidden_size,
        num_layers=hf_model.config.num_hidden_layers,
        num_query_heads=hf_model.config.num_attention_heads,
        num_key_value_heads=hf_model.config.num_key_value_heads,
        intermediate_dim=hf_model.config.intermediate_size,
        layer_norm_epsilon=hf_model.config.layer_norm_eps,
        rope_max_wavelength=getattr(hf_model.config, "rope_theta", 10000),
        partial_rotary_factor=getattr(
            hf_model.config, "partial_rotary_factor", 0.25
        ),
        tie_weights=False,
    )

    tokenizer_path = hf_hub_download(hf_preset, "tokenizer.json")
    with open(tokenizer_path, "r") as f:
        tokenizer_content = json.load(f)
    keras_hub_tokenizer = StableLMTokenizer(
        hf_tokenizer.vocab, tokenizer_content["model"]["merges"]
    )
    keras_hub_preprocessor = StableLMCausalLMPreprocessor(keras_hub_tokenizer)
    stablelm_lm = StableLMCausalLM(keras_hub_backbone, keras_hub_preprocessor)

    print("\n-> Transferring weights...")
    convert_checkpoints(keras_hub_backbone, hf_model)

    print("\n-> Verifying Tokenizer and Model Parity...")
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
    test_model(keras_hub_backbone, keras_hub_tokenizer, hf_model, hf_tokenizer)

    print("\n-> Verifying Text Generation Alignment...")
    validate_output(stablelm_lm, hf_model, hf_tokenizer)

    print(f"\n-> Saving KerasHub preset to ./{preset}")
    stablelm_lm.save_to_preset(f"./{preset}")
    print("\n✅ Preset Saved Successfully!")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
