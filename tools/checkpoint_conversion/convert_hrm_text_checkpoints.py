"""Convert and validate the official HRM-Text checkpoint.

Run with:

    python tools/checkpoint_conversion/convert_hrm_text_checkpoints.py \
        --output_dir /tmp/hrm_text_1b
"""

import json
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
from absl import app
from absl import flags
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from keras_hub.src.models.hrm_text.hrm_text_backbone import HrmTextBackbone
from keras_hub.src.models.hrm_text.hrm_text_causal_lm import HrmTextCausalLM
from keras_hub.src.models.hrm_text.hrm_text_causal_lm_preprocessor import (
    HrmTextCausalLMPreprocessor,
)
from keras_hub.src.models.hrm_text.hrm_text_tokenizer import HrmTextTokenizer

FLAGS = flags.FLAGS
flags.DEFINE_string("output_dir", None, "Directory for the converted preset.")
flags.DEFINE_string(
    "source",
    "sapientinc/HRM-Text-1B",
    "Hugging Face model ID or a local snapshot directory.",
)


def create_backbone(config):
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    rope_theta = getattr(
        config, "rope_theta", rope_parameters.get("rope_theta", 10000.0)
    )
    num_layers_per_stack = (
        getattr(config, "num_layers_per_stack", None)
        or config.num_hidden_layers
    )
    return HrmTextBackbone(
        vocabulary_size=config.vocab_size,
        hidden_dim=config.hidden_size,
        intermediate_dim=config.intermediate_size,
        num_layers_per_stack=num_layers_per_stack,
        num_attention_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        h_cycles=config.H_cycles,
        l_cycles=config.L_cycles,
        max_sequence_length=config.max_position_embeddings,
        rope_theta=rope_theta,
        rms_norm_epsilon=getattr(config, "rms_norm_eps", 1e-6),
        embedding_scale=getattr(config, "embedding_scale", 1.0),
        tie_word_embeddings=config.tie_word_embeddings,
        dtype="float32",
    )


def convert_weights(backbone, hf_model):
    """Assigns all official model tensors, rejecting incomplete mappings."""
    state = hf_model.state_dict()
    assigned = set()

    def assign(variable, name, transpose=False):
        value = state[name].detach().cpu().numpy()
        if transpose:
            value = value.T
        if tuple(variable.shape) != value.shape:
            raise ValueError(
                f"Shape mismatch for {name}: {variable.shape} vs {value.shape}"
            )
        variable.assign(value)
        assigned.add(name)

    assign(backbone.token_embedding.embeddings, "model.embed_tokens.weight")
    assign(backbone.initial_state.z_L_init, "model.z_L_init")
    for stack_name in ("L_module", "H_module"):
        stack = getattr(backbone, stack_name)
        for index, layer in enumerate(stack.layers):
            prefix = f"model.{stack_name}.layers.{index}"
            for projection in (
                "q_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "o_proj",
            ):
                assign(
                    getattr(layer.self_attn, projection).kernel,
                    f"{prefix}.self_attn.{projection}.weight",
                    transpose=True,
                )
            for projection in ("gate_proj", "up_proj", "down_proj"):
                assign(
                    getattr(layer.mlp, projection).kernel,
                    f"{prefix}.mlp.{projection}.weight",
                    transpose=True,
                )
    assign(
        backbone.token_embedding.reverse_embeddings,
        "lm_head.weight",
        transpose=True,
    )
    unused = sorted(set(state) - assigned)
    if unused:
        raise ValueError(f"Unmapped Hugging Face tensors: {unused}")


def convert_tokenizer(hf_tokenizer):
    """Converts the official Qwen2-style BPE tokenizer assets."""
    tokenizer_json = json.loads(hf_tokenizer.backend_tokenizer.to_str())
    model = tokenizer_json["model"]
    merges = [" ".join(merge) for merge in model["merges"]]
    return HrmTextTokenizer(vocabulary=model["vocab"], merges=merges)


def main(_):
    if not FLAGS.output_dir:
        raise ValueError("--output_dir is required.")
    keras.config.set_dtype_policy("float32")
    hf_model = AutoModelForCausalLM.from_pretrained(FLAGS.source)
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(FLAGS.source)
    backbone = create_backbone(hf_model.config)
    convert_weights(backbone, hf_model)

    tokenizer = convert_tokenizer(hf_tokenizer)
    preprocessor = HrmTextCausalLMPreprocessor(tokenizer)
    model = HrmTextCausalLM(backbone, preprocessor=preprocessor)
    inputs = hf_tokenizer("HRM-Text", return_tensors="pt")
    keras_inputs = {
        "token_ids": inputs.input_ids.numpy(),
        "padding_mask": inputs.attention_mask.numpy(),
        "token_type_ids": np.zeros_like(inputs.input_ids.numpy()),
    }
    hf_logits = hf_model(**inputs).logits.detach().cpu().float().numpy()
    keras_logits = keras.ops.convert_to_numpy(model(keras_inputs))
    np.testing.assert_allclose(keras_logits, hf_logits, atol=2e-4, rtol=2e-4)
    model.save_to_preset(FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
