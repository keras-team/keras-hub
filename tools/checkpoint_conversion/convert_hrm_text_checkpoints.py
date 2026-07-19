"""Convert and validate the official HRM-Text-1B checkpoint.

The script downloads the Apache-2.0 ``sapientinc/HRM-Text-1B`` checkpoint,
maps every Hugging Face tensor to KerasHub, verifies logits against
Transformers with ``atol=rtol=2e-4``, and writes a loadable local Keras preset.
It also verifies the exact ``1,182,795,264`` parameter count, tiny PrefixLM
gradient routing, and a saved-preset round-trip. The converted preset includes
the backbone, causal-LM task, and tokenizer.

Run with:

    uv run python tools/checkpoint_conversion/convert_hrm_text_checkpoints.py \
        --output_dir /tmp/hrm_text_1b

Use ``--source`` to point at a pinned local Hugging Face snapshot. The source
revision used for the initial port is
``9f082d68b8cd0ebc56e33f1c88c45609174c272c``.
"""

import gc
import json
import math
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import HrmTextConfig
from transformers import HrmTextForCausalLM

from keras_hub.src.models.hrm_text.hrm_text_backbone import HrmTextBackbone
from keras_hub.src.models.hrm_text.hrm_text_causal_lm import HrmTextCausalLM
from keras_hub.src.models.hrm_text.hrm_text_causal_lm_preprocessor import (
    HrmTextCausalLMPreprocessor,
)
from keras_hub.src.models.hrm_text.hrm_text_tokenizer import HrmTextTokenizer

FLAGS = flags.FLAGS
SOURCE_REVISION = "9f082d68b8cd0ebc56e33f1c88c45609174c272c"
EXPECTED_PARAMETER_COUNT = 1_182_795_264
flags.DEFINE_string("output_dir", None, "Directory for the converted preset.")
flags.DEFINE_string(
    "source",
    "sapientinc/HRM-Text-1B",
    "Hugging Face model ID or a local snapshot directory.",
)
flags.DEFINE_string(
    "revision",
    SOURCE_REVISION,
    "Pinned Hugging Face revision. Ignored for a local source directory.",
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
        l_bp_cycles=getattr(config, "L_bp_cycles", None),
        initializer_range=getattr(config, "initializer_range", 0.02),
        embedding_scale=getattr(config, "embedding_scale", 1.0),
        tie_word_embeddings=config.tie_word_embeddings,
        dtype="float32",
    )


def iter_weight_mappings(backbone):
    """Yields every Keras variable and its expanded HF state-dict name."""
    yield (
        backbone.token_embedding.embeddings,
        "model.embed_tokens.weight",
        False,
    )
    yield backbone.initial_state.z_L_init, "model.z_L_init", False
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
                yield (
                    getattr(layer.self_attn, projection).kernel,
                    f"{prefix}.self_attn.{projection}.weight",
                    True,
                )
            for projection in ("gate_proj", "up_proj", "down_proj"):
                yield (
                    getattr(layer.mlp, projection).kernel,
                    f"{prefix}.mlp.{projection}.weight",
                    True,
                )
    yield (
        backbone.token_embedding.reverse_embeddings,
        "lm_head.weight",
        True,
    )


def convert_weights(
    backbone, hf_model, expected_parameter_count=EXPECTED_PARAMETER_COUNT
):
    """Assigns all official model tensors, rejecting incomplete mappings."""
    state = hf_model.state_dict()
    assigned = set()
    assigned_variables = set()

    for variable, name, transpose in iter_weight_mappings(backbone):
        if variable.path in assigned_variables:
            raise ValueError(f"Keras variable assigned twice: {variable.path}")
        value = state[name].detach().cpu().float().numpy()
        if transpose:
            value = value.T
        if tuple(variable.shape) != value.shape:
            raise ValueError(
                f"Shape mismatch for {name}: {variable.shape} vs {value.shape}"
            )
        variable.assign(value)
        assigned.add(name)
        assigned_variables.add(variable.path)

    unused = sorted(set(state) - assigned)
    if unused:
        raise ValueError(f"Unmapped Hugging Face tensors: {unused}")
    unassigned_variables = sorted(
        variable.path
        for variable in backbone.weights
        if variable.path not in assigned_variables
    )
    if unassigned_variables:
        raise ValueError(f"Unassigned Keras variables: {unassigned_variables}")

    source_parameter_count = sum(
        math.prod(tensor.shape) for tensor in state.values()
    )
    keras_parameter_count = backbone.count_params()
    if source_parameter_count != keras_parameter_count:
        raise ValueError(
            "Source and Keras parameter counts differ: "
            f"{source_parameter_count:,} != {keras_parameter_count:,}"
        )
    if (
        expected_parameter_count is not None
        and keras_parameter_count != expected_parameter_count
    ):
        raise ValueError(
            "Unexpected Keras parameter count: "
            f"{keras_parameter_count:,} != {expected_parameter_count:,}"
        )
    return {
        "mapped_tensors": len(assigned),
        "source_parameter_count": source_parameter_count,
        "keras_parameter_count": keras_parameter_count,
    }


def validate_tiny_gradient_parity():
    """Checks PrefixLM logits and routed gradients against Transformers."""
    import torch

    torch.manual_seed(2026)
    config = HrmTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=4,
        H_cycles=2,
        L_cycles=2,
        L_bp_cycles=[0, 2],
        max_position_embeddings=8,
        embedding_scale=1.0,
        prefix_lm=True,
        tie_word_embeddings=False,
    )
    hf_model = HrmTextForCausalLM(config).float().eval()
    backbone = create_backbone(hf_model.config)
    device = backbone.weights[0].value.device
    hf_model.to(device)
    conversion = convert_weights(
        backbone, hf_model, expected_parameter_count=None
    )
    model = HrmTextCausalLM(backbone)

    token_ids = np.array([[1, 2, 3, 4]], dtype="int32")
    token_type_ids = np.array([[1, 1, 0, 0]], dtype="int32")
    padding_mask = np.ones_like(token_ids)
    hf_logits = hf_model(
        input_ids=torch.as_tensor(token_ids, device=device),
        attention_mask=torch.as_tensor(padding_mask, device=device),
        token_type_ids=torch.as_tensor(token_type_ids, device=device),
        use_cache=False,
    ).logits
    keras_logits = model(
        {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "token_type_ids": token_type_ids,
        }
    )
    logit_max_abs = float(
        torch.max(torch.abs(hf_logits - keras_logits)).detach().cpu()
    )
    hf_logits.square().mean().backward()
    keras_logits.square().mean().backward()

    hf_parameters = dict(hf_model.named_parameters())
    gradient_errors = []
    for variable, name, transpose in iter_weight_mappings(backbone):
        keras_gradient = variable.value.grad
        hf_gradient = hf_parameters[name].grad
        if keras_gradient is None and hf_gradient is None:
            continue
        if keras_gradient is None or hf_gradient is None:
            raise ValueError(f"Gradient presence differs for {name}.")
        keras_gradient = keras_gradient.detach().cpu().numpy()
        hf_gradient = hf_gradient.detach().cpu().numpy()
        if transpose:
            hf_gradient = hf_gradient.T
        gradient_errors.append(
            float(np.max(np.abs(keras_gradient - hf_gradient)))
        )
    gradient_max_abs = max(gradient_errors)
    if logit_max_abs >= 1e-5 or gradient_max_abs >= 1e-5:
        raise ValueError(
            "Tiny gradient parity failed: "
            f"logits={logit_max_abs:.3e}, gradients={gradient_max_abs:.3e}"
        )
    del hf_model, model, backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    keras.backend.clear_session()
    return {
        **conversion,
        "logit_max_abs": logit_max_abs,
        "gradient_max_abs": gradient_max_abs,
    }


def convert_tokenizer(hf_tokenizer):
    """Converts the official Qwen2-style BPE tokenizer assets."""
    tokenizer_json = json.loads(hf_tokenizer.backend_tokenizer.to_str())
    model = tokenizer_json["model"]
    merges = [" ".join(merge) for merge in model["merges"]]
    return HrmTextTokenizer(vocabulary=model["vocab"], merges=merges)


def main(_):
    if not FLAGS.output_dir:
        raise ValueError("--output_dir is required.")
    tf.config.set_visible_devices([], "GPU")
    keras.config.set_dtype_policy("float32")
    tiny_gradient_parity = validate_tiny_gradient_parity()
    revision_kwargs = (
        {} if os.path.isdir(FLAGS.source) else {"revision": FLAGS.revision}
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        FLAGS.source, dtype="float32", **revision_kwargs
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(
        FLAGS.source, **revision_kwargs
    )
    backbone = create_backbone(hf_model.config)
    conversion = convert_weights(backbone, hf_model)

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
    conversion["logit_max_abs"] = float(
        np.max(np.abs(keras_logits - hf_logits))
    )
    model.save_to_preset(FLAGS.output_dir)

    expected_special_token_ids = {
        "start": tokenizer.start_token_id,
        "end": tokenizer.end_token_id,
        "pad": tokenizer.pad_token_id,
    }
    del hf_model, model, backbone
    gc.collect()
    keras.backend.clear_session()

    restored = HrmTextCausalLM.from_preset(FLAGS.output_dir)
    restored_logits = keras.ops.convert_to_numpy(restored(keras_inputs))
    np.testing.assert_allclose(restored_logits, hf_logits, atol=2e-4, rtol=2e-4)
    conversion["restored_logit_max_abs"] = float(
        np.max(np.abs(restored_logits - hf_logits))
    )
    restored_tokenizer = restored.preprocessor.tokenizer
    actual_special_token_ids = {
        "start": restored_tokenizer.start_token_id,
        "end": restored_tokenizer.end_token_id,
        "pad": restored_tokenizer.pad_token_id,
    }
    if actual_special_token_ids != expected_special_token_ids:
        raise ValueError(
            "Tokenizer special-token IDs changed during preset round-trip: "
            f"{actual_special_token_ids} != {expected_special_token_ids}"
        )
    conversion["special_token_ids"] = actual_special_token_ids
    conversion["source_revision"] = (
        "local" if os.path.isdir(FLAGS.source) else FLAGS.revision
    )
    print(
        json.dumps(
            {
                "official_conversion": conversion,
                "tiny_gradient_parity": tiny_gradient_parity,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app.run(main)
