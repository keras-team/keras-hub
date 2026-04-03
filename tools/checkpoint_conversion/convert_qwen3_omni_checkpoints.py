import gc
import math
import os
import traceback

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide any CUDA devices

import numpy as np
import torch
from absl import app
from absl import flags

device = torch.device("cpu")
# Force PyTorch to use CPU
torch.set_default_device(device)

from keras import ops  # noqa: E402
from transformers import AutoModelForMultimodalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import keras_hub  # noqa: E402

PRESET_MAP = {
    "qwen3_omni_30b_a3b_en": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "qwen3_omni_30b_a3b_captioner_en": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    "qwen3_omni_30b_a3b_thinking_en": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
}

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_string(
    "validate_dtype",
    "bfloat16",
    "Dtype to use while validating HF and Keras numerics.",
)
flags.DEFINE_string(
    "save_dtype",
    "bfloat16",
    "Dtype to use when saving the converted Keras preset.",
)
flags.DEFINE_bool(
    "run_generate_check",
    True,
    "Whether to compare generated text after conversion.",
)


def compute_hf_references(hf_model, hf_tokenizer, run_generate_check):
    input_str = "What is Keras?"
    length = 32
    hf_inputs = hf_tokenizer([input_str], return_tensors="pt").to(device)
    hf_outputs = hf_model(**hf_inputs)
    references = {
        "params": hf_model.num_parameters(),
        "token_ids": hf_inputs["input_ids"].detach().cpu().numpy(),
        "logits": hf_outputs.logits.detach().cpu().float().numpy(),
    }

    if run_generate_check:
        outputs = hf_model.generate(
            **hf_inputs,
            max_length=length,
            do_sample=False,
            pad_token_id=hf_tokenizer.pad_token_id,
        )
        references["generated_text"] = hf_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )[0]

    return references


def _count_weight_params(weights):
    total = 0
    seen = set()
    for weight in weights:
        weight_id = id(weight)
        if weight_id in seen:
            continue
        seen.add(weight_id)
        total += math.prod(int(dim) for dim in weight.shape)
    return total


def count_keras_hub_thinker_params(keras_hub_model):
    backbone_weights = list(keras_hub_model.weights)
    extra_weights = []
    if keras_hub_model.audio_encoder is not None:
        extra_weights.extend(keras_hub_model.audio_encoder.weights)
    if keras_hub_model.vision_encoder is not None:
        extra_weights.extend(keras_hub_model.vision_encoder.weights)
    return _count_weight_params(backbone_weights + extra_weights)


def test_model(keras_hub_model, keras_hub_tokenizer, hf_references):
    # First, test that the number of parameters match
    keras_hub_params = count_keras_hub_thinker_params(keras_hub_model)
    hf_params = hf_references["params"]
    print(f"KerasHub thinker params: {keras_hub_params:,}")
    print(f"HuggingFace thinker params: {hf_params:,}")
    assert keras_hub_params == hf_params

    # Test the outputs of both the models
    keras_hub_preprocessor = keras_hub.models.Qwen3OmniCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_inputs = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=5
    )[0]

    keras_hub_output = keras_hub_model(keras_hub_inputs)
    keras_hub_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits)

    # High tolerance since bfloat16 is used as the default dtype for Qwen

    try:
        np.testing.assert_allclose(
            keras_hub_logits, hf_references["logits"], atol=1e-3
        )
        print("All numerics match with tolerance limit 1e-3")
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")
        raise


def test_tokenizer(keras_hub_tokenizer, hf_token_ids):
    keras_hub_preprocessor = keras_hub.models.Qwen3OmniCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_output = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=5
    )
    keras_hub_output = ops.convert_to_numpy(keras_hub_output[0]["token_ids"])

    np.testing.assert_equal(keras_hub_output, hf_token_ids)


def validate_output(qwen3_omni_lm, hf_generated_text):
    input_str = "What is Keras?"
    length = 32

    keras_output = qwen3_omni_lm.generate([input_str], max_length=length)
    keras_output = keras_output[0]
    print("🔶 KerasHub output:", keras_output)
    print("🔶 Huggingface output:", hf_generated_text)


def main(_):
    # === Get the preset name ===
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )

    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]
    validate_dtype = FLAGS.validate_dtype
    validate_torch_dtype = TORCH_DTYPE_MAP.get(validate_dtype)
    if validate_torch_dtype is None:
        raise ValueError(
            "Invalid validate_dtype. Must be one of "
            f"{','.join(TORCH_DTYPE_MAP.keys())}"
        )
    save_dtype = FLAGS.save_dtype
    if save_dtype not in TORCH_DTYPE_MAP:
        raise ValueError(
            "Invalid save_dtype. Must be one of "
            f"{','.join(TORCH_DTYPE_MAP.keys())}"
        )
    run_generate_check = FLAGS.run_generate_check

    # === Load the Huggingface model ===
    hf_full_model = AutoModelForMultimodalLM.from_pretrained(
        hf_preset,
        device_map=device,
        torch_dtype=validate_torch_dtype,
        trust_remote_code=True,
    )

    # Use full Thinker model (includes audio/vision encoders)
    hf_model = hf_full_model.thinker
    del hf_full_model
    hf_tokenizer = AutoTokenizer.from_pretrained(
        hf_preset,
        return_tensors="pt",
        trust_remote_code=True,
    )
    hf_model.eval()
    print("\n-> Huggingface model and tokenizer loaded")
    hf_references = compute_hf_references(
        hf_model, hf_tokenizer, run_generate_check
    )
    print("\n-> Huggingface references computed")

    del hf_model
    del hf_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === Check that the models and tokenizers outputs match ===
    keras_hub_model = keras_hub.models.Qwen3OmniBackbone.from_preset(
        f"hf://{hf_preset}", dtype=validate_dtype
    )
    keras_hub_tokenizer = keras_hub.tokenizers.Qwen3OmniTokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    print("\n-> Keras model and tokenizer loaded")

    test_tokenizer(keras_hub_tokenizer, hf_references["token_ids"])
    test_model(keras_hub_model, keras_hub_tokenizer, hf_references)

    preprocessor = keras_hub.models.Qwen3OmniCausalLMPreprocessor(
        keras_hub_tokenizer
    )
    qwen3_omni_lm = keras_hub.models.Qwen3OmniCausalLM(
        backbone=keras_hub_model, preprocessor=preprocessor, sampler="greedy"
    )
    # == Validate model.generate output ==
    if run_generate_check:
        validate_output(qwen3_omni_lm, hf_references["generated_text"])
    print("\n-> Tests passed!")
    if save_dtype == validate_dtype:
        qwen3_omni_lm.save_to_preset(f"./{preset}")
    else:
        del qwen3_omni_lm
        del keras_hub_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        keras_hub_model_save = keras_hub.models.Qwen3OmniBackbone.from_preset(
            f"hf://{hf_preset}", dtype=save_dtype
        )
        qwen3_omni_lm_save = keras_hub.models.Qwen3OmniCausalLM(
            backbone=keras_hub_model_save,
            preprocessor=preprocessor,
            sampler="greedy",
        )
        qwen3_omni_lm_save.save_to_preset(f"./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
