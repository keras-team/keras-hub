import os
import traceback

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide any CUDA devices

import numpy as np
import torch
from absl import app
from absl import flags

import keras_hub

device = torch.device("cpu")
# Force PyTorch to use CPU
torch.set_default_device(device)

from keras import ops  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers.models.gpt_oss.configuration_gpt_oss import (
    GptOssConfig,  # noqa: E402
)

# noqa: E402
from keras_hub.models.gpt_oss.gpt_oss_backbone import (
    GptOssBackbone,  # For type hinting
)

# Hypothetical preset map for GPT-OSS models.
# Replace with actual Hugging Face paths if available.
PRESET_MAP = {
    "gpt_oss_7b_en": "HF/gpt-oss-7b",
    "gpt_oss_instruct_7b_en": "HF/gpt-oss-7b-instruct",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def convert_backbone_config(hf_config: GptOssConfig):
    """Converts Hugging Face GPT-OSS config to KerasHub GptOssBackbone config.

    Args:
        hf_config: The Hugging Face GptOssConfig object.

    Returns:
        A dictionary containing the KerasHub GptOssBackbone configuration.
    """
    keras_config = {
        "vocabulary_size": hf_config.vocab_size,
        "num_layers": hf_config.num_hidden_layers,
        "num_query_heads": hf_config.num_attention_heads,
        "hidden_dim": hf_config.hidden_size,
        "intermediate_dim": hf_config.intermediate_size,
        "num_key_value_heads": hf_config.num_key_value_heads,
        "num_experts": hf_config.num_local_experts,
        "top_k": hf_config.num_experts_per_tok,
        "rope_max_wavelength": hf_config.rope_theta,
        "layer_norm_epsilon": hf_config.rms_norm_eps,
        "sliding_window": hf_config.sliding_window,
        "dropout": hf_config.attention_dropout,
        "use_bias": hf_config.attention_bias,
    }
    # Handle rope_scaling if present in HF config
    if (
        hasattr(hf_config, "rope_scaling")
        and hf_config.rope_scaling is not None
    ):
        if hf_config.rope_scaling["type"] == "linear":
            keras_config["rope_scaling_factor"] = hf_config.rope_scaling[
                "factor"
            ]
        else:
            raise ValueError(
                f"Unsupported RoPE scaling type:{hf_config.rope_scaling['type']}"
            )
    return keras_config


def convert_weights(
    hf_model: AutoModelForCausalLM, keras_hub_backbone: GptOssBackbone
):
    """Converts Hugging Face GPT-OSS model weights to KerasHub GptOssBackbone.

    Args:
        hf_model: The Hugging Face GPT-OSS model.
        keras_hub_backbone: The KerasHub GptOssBackbone model.
    """
    print("Converting weights...")

    # Embedding layer
    keras_hub_backbone.token_embedding.embeddings.assign(
        hf_model.model.embed_tokens.weight.detach().cpu().numpy()
    )

    # Final Layer Norm
    keras_hub_backbone.transformer_layers[-1].layer_norm.gamma.assign(
        hf_model.model.norm.weight.detach().cpu().numpy()
    )

    # Loop through transformer layers
    for i, hf_layer in enumerate(hf_model.model.layers):
        keras_layer = keras_hub_backbone.transformer_layers[i]

        # Input Layer Norm
        keras_layer.pre_attention_norm.gamma.assign(
            hf_layer.input_layernorm.weight.detach().cpu().numpy()
        )

        # Attention
        # Q, K, V, O projections
        keras_layer.attention.query_dense.kernel.assign(
            hf_layer.self_attn.q_proj.weight.T.detach().cpu().numpy()
        )
        if hf_layer.self_attn.q_proj.bias is not None:
            keras_layer.attention.query_dense.bias.assign(
                hf_layer.self_attn.q_proj.bias.detach().cpu().numpy()
            )

        keras_layer.attention.key_dense.kernel.assign(
            hf_layer.self_attn.k_proj.weight.T.detach().cpu().numpy()
        )
        if hf_layer.self_attn.k_proj.bias is not None:
            keras_layer.attention.key_dense.bias.assign(
                hf_layer.self_attn.k_proj.bias.detach().cpu().numpy()
            )

        keras_layer.attention.value_dense.kernel.assign(
            hf_layer.self_attn.v_proj.weight.T.detach().cpu().numpy()
        )
        if hf_layer.self_attn.v_proj.bias is not None:
            keras_layer.attention.value_dense.bias.assign(
                hf_layer.self_attn.v_proj.bias.detach().cpu().numpy()
            )

        keras_layer.attention.output_dense.kernel.assign(
            hf_layer.self_attn.o_proj.weight.T.detach().cpu().numpy()
        )
        if hf_layer.self_attn.o_proj.bias is not None:
            keras_layer.attention.output_dense.bias.assign(
                hf_layer.self_attn.o_proj.bias.detach().cpu().numpy()
            )

        # Sinks
        keras_layer.attention.sinks.assign(
            hf_layer.self_attn.sinks.detach().cpu().numpy()
        )

        # Post-Attention Layer Norm
        keras_layer.pre_mlp_norm.gamma.assign(
            hf_layer.post_attention_layernorm.weight.detach().cpu().numpy()
        )

        # MoE MLP
        # Router
        keras_layer.moe_mlp.router.kernel.assign(
            hf_layer.mlp.router.weight.T.detach().cpu().numpy()
        )
        keras_layer.moe_mlp.router.bias.assign(
            hf_layer.mlp.router.bias.detach().cpu().numpy()
        )

        # Experts
        num_experts = hf_model.config.num_local_experts
        for j in range(num_experts):
            hf_expert_gate_up_proj = hf_layer.mlp.experts.gate_up_proj[
                j
            ]  # (hidden_size, 2 * expert_dim)
            hf_expert_gate_up_proj_bias = (
                hf_layer.mlp.experts.gate_up_proj_bias[j]
            )  # (2 * expert_dim)

            # Split gate_up_proj into gate and up based on
            # PyTorch forward logic (::2, 1::2)
            hf_gate_proj_weight = hf_expert_gate_up_proj[
                :, ::2
            ]  # (hidden_size, expert_dim)
            hf_up_proj_weight = hf_expert_gate_up_proj[
                :, 1::2
            ]  # (hidden_size, expert_dim)

            hf_gate_proj_bias = hf_expert_gate_up_proj_bias[::2]  # (expert_dim)
            hf_up_proj_bias = hf_expert_gate_up_proj_bias[1::2]  # (expert_dim)

            keras_layer.moe_mlp.experts[j].gate_dense.kernel.assign(
                hf_gate_proj_weight.T.detach().cpu().numpy()
            )
            keras_layer.moe_mlp.experts[j].gate_dense.bias.assign(
                hf_gate_proj_bias.detach().cpu().numpy()
            )

            keras_layer.moe_mlp.experts[j].up_dense.kernel.assign(
                hf_up_proj_weight.T.detach().cpu().numpy()
            )
            keras_layer.moe_mlp.experts[j].up_dense.bias.assign(
                hf_up_proj_bias.detach().cpu().numpy()
            )

            keras_layer.moe_mlp.experts[j].down_dense.kernel.assign(
                hf_layer.mlp.experts.down_proj[j].T.detach().cpu().numpy()
            )
            keras_layer.moe_mlp.experts[j].down_dense.bias.assign(
                hf_layer.mlp.experts.down_proj_bias[j].detach().cpu().numpy()
            )
    print("Weights converted successfully.")


def convert_tokenizer(hf_tokenizer: AutoTokenizer, preset: str):
    """Converts Hugging Face GPT-OSS tokenizer to KerasHub GptOssTokenizer.

    Args:
        hf_tokenizer: The Hugging Face GPT-OSS tokenizer.
        preset: The preset string used to load the tokenizer.

    Returns:
        A KerasHub GptOssTokenizer instance.
    """
    print("Converting tokenizer...")
    # The GptOssTokenizer is a SentencePieceTokenizer,
    # so it can load from the HF model path directly.
    # The `from_preset` method of KerasHub tokenizers handles this.
    keras_hub_tokenizer = keras_hub.models.GptOssTokenizer.from_preset(
        f"hf://{preset}"
    )
    print("Tokenizer converted successfully.")
    return keras_hub_tokenizer


def compute_hf_output(hf_model, hf_model_tokenizer):
    """Computes logits from the Hugging Face model."""
    hf_inputs = hf_model_tokenizer(["What is Keras?"], return_tensors="pt").to(
        device
    )
    hf_outputs = hf_model(**hf_inputs)
    hf_output_logits = hf_outputs.logits.detach().cpu().float().numpy()

    return hf_output_logits


def compute_keras_output(keras_hub_model, keras_hub_tokenizer):
    """Computes logits from the KerasHub model."""
    keras_hub_preprocessor = keras_hub.models.GptOssCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_inputs = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=6
    )[0]
    keras_hub_inputs = {k: v.to(device) for k, v in keras_hub_inputs.items()}

    keras_hub_output = keras_hub_model(keras_hub_inputs)
    keras_hub_output_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_output_logits = ops.convert_to_numpy(keras_hub_output_logits)
    return keras_hub_output_logits


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    """Tests if the KerasHub tokenizer produces
    the same output as the HF tokenizer."""
    hf_output = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()
    keras_hub_preprocessor = keras_hub.models.GptOssCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_output = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=6
    )
    keras_hub_output = ops.convert_to_numpy(keras_hub_output[0]["token_ids"])

    np.testing.assert_equal(keras_hub_output, hf_output)


def main(_):
    # === Get the preset name ===
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    # === Load the Huggingface model ===
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_preset,
        device_map=device,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset, return_tensors="pt")
    hf_model.eval()
    print("\n-> Huggingface model and tokenizer loaded")

    # === Load KerasHub tokenizer and test ===
    keras_hub_tokenizer = keras_hub.models.GptOssTokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    print("\n-> Keras tokenizer loaded")
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
    print("\n -> Keras tokenizer test successful")

    # === Compute HF outputs ===
    hf_params = hf_model.num_parameters()
    hf_output_logits = compute_hf_output(hf_model, hf_tokenizer)
    print("\n -> Computed HF outputs successfully")

    # === Load KerasHub backbone and test ===
    # Free up memory before loading Keras model
    del hf_model, hf_tokenizer
    keras_hub_backbone = keras_hub.models.GptOssBackbone.from_preset(
        f"hf://{hf_preset}"
    )
    print("\n-> Keras model loaded")

    keras_hub_params = keras_hub_backbone.count_params()
    assert keras_hub_params == hf_params, (
        f"Keras model has {keras_hub_params} parameters, "
        f"but HF model has {hf_params} parameters."
    )

    keras_hub_output_logits = compute_keras_output(
        keras_hub_backbone, keras_hub_tokenizer
    )

    try:
        np.testing.assert_allclose(
            keras_hub_output_logits, hf_output_logits, atol=1e-4
        )
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")
        raise  # Re-raise the error to indicate failure

    print("\n-> Tests passed!")

    # === Save KerasHub model to preset ===
    preprocessor = keras_hub.models.GptOssCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_model = keras_hub.models.GptOssCausalLM(
        keras_hub_backbone, preprocessor
    )

    keras_hub_model.save_to_preset(f"./{preset}")
    print(f"\n-> KerasHub model saved to ./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
