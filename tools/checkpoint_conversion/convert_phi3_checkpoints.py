# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gc
import json
import os
import re
import sys

os.environ["KERAS_BACKEND"] = "torch"

import huggingface_hub  # noqa: E402
import keras  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from keras_nlp import upload_preset  # noqa: E402
from keras_nlp.src.models import Phi3Backbone  # noqa: E402
from keras_nlp.src.models import Phi3Preprocessor  # noqa: E402
from keras_nlp.src.models import Phi3Tokenizer  # noqa: E402

PRESET_MAP = {
    "phi3_mini_4k_instruct_en": "microsoft/Phi-3-mini-4k-instruct",
    "phi3_mini_128k_instruct_en": "microsoft/Phi-3-mini-128k-instruct",
}


def download_hf_model(hf_model_name, extract_dir):
    hf_model_dir = huggingface_hub.snapshot_download(
        repo_id=hf_model_name,
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.model"],
        ignore_patterns=["*/*"],
        local_dir=extract_dir,
    )

    return hf_model_dir


def convert_tokenizer(hf_model_dir):
    # We can't import `sentencepiece_model_pb2` from `sentencepiece` because of
    # this protobuffer lib error
    # `TypeError: Couldn't build proto file into descriptor pool: duplicate
    # file name sentencepiece_model.proto`
    # transformers library build `sentencepiece_model.proto` prot once. we can't
    # import again becaase it will try to build the proto again into the
    # descriptor pool and protobuffer forbids that.
    sp_pb2_module = sys.modules.get(
        "transformers.utils.sentencepiece_model_pb2_new", None
    )
    if sp_pb2_module is None:
        sp_pb2_module = sys.modules.get(
            "transformers.utils.sentencepiece_model_pb2", None
        )
    if sp_pb2_module is None:
        from sentencepiece import sentencepiece_model_pb2 as sp_pb2_module
    model_path = os.path.join(hf_model_dir, "tokenizer.model")
    added_tokens_path = os.path.join(hf_model_dir, "added_tokens.json")
    with open(model_path, "rb") as sp_model_file:
        model_proto = sp_pb2_module.ModelProto()
        model_proto.ParseFromString(sp_model_file.read())
    with open(added_tokens_path, "rb") as added_tokens_file:
        added_tokens = json.load(added_tokens_file)
    # Add the new tokens to the model as user defined pieces.
    for token in added_tokens.keys():
        new_token = sp_pb2_module.ModelProto().SentencePiece()
        new_token.piece = token
        new_token.score = 0.0
        new_token.type = 4  # user defined symbols.
        model_proto.pieces.append(new_token)
    tokenizer = Phi3Tokenizer(model_proto.SerializeToString())
    for key, value in added_tokens.items():
        assert key == tokenizer.id_to_token(
            value
        ), f"{key} token have different id in the tokenizer"

    return tokenizer


def convert_model(hf_model, device, dtype):
    hf_config = hf_model.config.to_dict()

    kwargs = {}
    kwargs["vocabulary_size"] = hf_config["vocab_size"]
    kwargs["num_layers"] = hf_config["num_hidden_layers"]
    kwargs["num_query_heads"] = hf_config["num_attention_heads"]
    kwargs["num_key_value_heads"] = hf_config["num_key_value_heads"]
    kwargs["hidden_dim"] = hf_config["hidden_size"]
    kwargs["intermediate_dim"] = hf_config["intermediate_size"]
    kwargs["dropout"] = hf_config["attention_dropout"]
    kwargs["layer_norm_epsilon"] = hf_config["rms_norm_eps"]
    kwargs["max_sequence_length"] = hf_config["max_position_embeddings"]
    kwargs["original_max_sequence_length"] = hf_config[
        "original_max_position_embeddings"
    ]
    kwargs["rope_max_wavelength"] = hf_config["rope_theta"]
    if hf_config["rope_scaling"] is not None:
        kwargs["rope_scaling_type"] = hf_config["rope_scaling"]["type"]
        kwargs["rope_scaling_short_factor"] = hf_config["rope_scaling"][
            "short_factor"
        ]
        kwargs["rope_scaling_long_factor"] = hf_config["rope_scaling"][
            "long_factor"
        ]
    kwargs["dtype"] = dtype

    with keras.device(device):
        keras_model = Phi3Backbone(**kwargs)

    return keras_model


def convert_weights(keras_model, hf_model):
    hidden_dim = keras_model.hidden_dim
    intermediate_dim = keras_model.intermediate_dim
    num_query_heads = keras_model.num_query_heads
    num_key_value_heads = keras_model.num_key_value_heads
    head_dim = hidden_dim // num_query_heads

    # get huggingface model weights.
    hf_wts = hf_model.state_dict()

    # Embedding layer.
    keras_model.token_embedding.embeddings.assign(
        hf_wts["model.embed_tokens.weight"]
    )
    keras_model.token_embedding.reverse_embeddings.assign(
        hf_wts["lm_head.weight"].t()
    )
    # LayerNorm.
    keras_model.layer_norm.scale.assign(hf_wts["model.norm.weight"])

    # Decoder layers.
    for i, decoder_layer in enumerate(keras_model.transformer_layers):
        # LayrNorm.
        decoder_layer.pre_attention_layernorm.scale.assign(
            hf_wts[f"model.layers.{i}.input_layernorm.weight"]
        )
        decoder_layer.post_attention_layernorm.scale.assign(
            hf_wts[f"model.layers.{i}.post_attention_layernorm.weight"]
        )

        # Attention layer.
        attention_layer = decoder_layer.attention
        fused_qkv_kernel = hf_wts[
            f"model.layers.{i}.self_attn.qkv_proj.weight"
        ].t()

        query_kernel = fused_qkv_kernel[:, :hidden_dim]
        query_kernel = query_kernel.reshape(
            hidden_dim, num_query_heads, head_dim
        )
        key_kernel = fused_qkv_kernel[
            :, hidden_dim : hidden_dim + num_key_value_heads * head_dim
        ]
        key_kernel = key_kernel.reshape(
            hidden_dim, num_key_value_heads, head_dim
        )
        value_kernel = fused_qkv_kernel[
            :, hidden_dim + num_key_value_heads * head_dim :
        ]
        value_kernel = value_kernel.reshape(
            hidden_dim, num_key_value_heads, head_dim
        )

        attention_layer.query_dense._kernel.assign(query_kernel)
        attention_layer.key_dense._kernel.assign(key_kernel)
        attention_layer.value_dense._kernel.assign(value_kernel)

        attention_layer.output_dense.kernel.assign(
            hf_wts[f"model.layers.{i}.self_attn.o_proj.weight"]
            .t()
            .reshape(num_query_heads, head_dim, hidden_dim)
        )

        # feed dorward layer.
        fused_intermediate_gate_ff_kernel = hf_wts[
            f"model.layers.{i}.mlp.gate_up_proj.weight"
        ].t()
        decoder_layer.feedforward_gate_dense._kernel.assign(
            fused_intermediate_gate_ff_kernel[:, :intermediate_dim]
        )
        decoder_layer.feedforward_intermediate_dense._kernel.assign(
            fused_intermediate_gate_ff_kernel[:, intermediate_dim:]
        )
        decoder_layer.feedforward_output_dense._kernel.assign(
            hf_wts[f"model.layers.{i}.mlp.down_proj.weight"].t()
        )


def validate_output(
    hf_model,
    keras_model,
    hf_device,
    keras_device,
    hf_tokenizer,
    keras_preprocessor,
):
    # Hf
    tokens = hf_tokenizer(
        ["<|user|>\nHow to win?<|end|>\n<|assistant|>"],
        max_length=20,
        padding="max_length",
        return_tensors="pt",
    )

    hf_model_input = {
        "input_ids": tokens["input_ids"].to(hf_device),
        "attention_mask": tokens["attention_mask"].to(hf_device),
        "use_cache": False,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": False,
    }

    hf_model_outputs = hf_model(**hf_model_input)[0]

    # KerasNLP
    keras_model_input = keras_preprocessor(
        ["<|user|>\nHow to win?<|end|>\n<|assistant|>"]
    )
    keras_model_input = {
        k: v.to(keras_device) for k, v in keras_model_input.items()
    }
    keras_model_outputs = keras_model(keras_model_input)

    # Comparing the outputs.
    print("🔶 KerasNLP output:", keras_model_outputs[0, 0, :10])
    print("🔶 HF output:", hf_model_outputs[0, 0, :10])
    print(
        "🔶 Difference:",
        torch.mean(
            torch.abs(
                keras_model_outputs.detach().cpu()
                - hf_model_outputs.detach().cpu()
            )
        ),
    )


def get_torch_dtype(str_dtype):
    if str_dtype == "float32":
        return torch.float32
    elif str_dtype == "float16":
        return torch.float16
    elif str_dtype == "bfloat16":
        return torch.bfloat16


def convert_and_validate(
    hf_model_dir,
    hf_device,
    keras_device,
    validate_dtype,
    hf_tokenizer,
    keras_preprocessor,
):
    print(f"✅ Numerics Validation in {validate_dtype}.")
    # Load the causal model to convert lm_head weights.
    hf_causal_model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        device_map=hf_device,
        torch_dtype=get_torch_dtype(validate_dtype),
        trust_remote_code=True,
    )
    hf_model = hf_causal_model.model
    print("✅ Huggingface model loaded.")

    keras_model = convert_model(hf_causal_model, keras_device, validate_dtype)
    print("✅ Keras model loaded.")

    convert_weights(keras_model, hf_causal_model)
    print("✅ Weights converted")

    validate_output(
        hf_model,
        keras_model,
        hf_device,
        keras_device,
        hf_tokenizer,
        keras_preprocessor,
    )
    print("✅ Numerics validated")

    # Clean memory.
    del keras_model
    del hf_causal_model
    del hf_model
    gc.collect()
    if not (hf_device == "cpu" and keras_device == "cpu"):
        torch.cuda.empty_cache()


def convert_and_save(
    preset,
    hf_model_dir,
    hf_device,
    keras_device,
    save_dtype,
    hf_tokenizer,
    keras_preprocessor,
):
    print(f"✅ Saving model in {save_dtype}.")
    # Load the causal model to convert lm_head weights.
    hf_causal_model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        device_map=hf_device,
        torch_dtype=get_torch_dtype(save_dtype),
        trust_remote_code=True,
    )
    hf_model = hf_causal_model.model
    print("✅ Huggingface model loaded.")

    keras_model = convert_model(hf_causal_model, keras_device, save_dtype)
    print("✅ Keras model loaded.")

    convert_weights(keras_model, hf_causal_model)
    print("✅ Weights converted")

    validate_output(
        hf_model,
        keras_model,
        hf_device,
        keras_device,
        hf_tokenizer,
        keras_preprocessor,
    )
    print("✅ Numerics validated")

    keras_model.save_to_preset(preset)
    keras_preprocessor.tokenizer.save_to_preset(preset)
    print("✅ Preset saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        default="phi3_mini_4k_instruct_en",
        choices=PRESET_MAP.keys(),
        required=True,
        help=f'Preset must be one of {", ".join(PRESET_MAP.keys())}',
    )

    def device_regex(arg_value, pattern=re.compile(r"^cpu$|^cuda:[0-9]+$")):
        if not pattern.match(arg_value):
            raise argparse.ArgumentTypeError(
                "The device must be one of: "
                "'cpu', 'cuda:0', 'cuda:1', ...'cuda:n'"
            )
        return arg_value

    parser.add_argument(
        "--hf_device",
        default="cpu",
        type=device_regex,
        help=(
            "The device where huggingface model will be loaded. can be one of: "
            "'cpu', 'cuda:0', 'cuda:1', ...'cuda:n'"
        ),
    )
    parser.add_argument(
        "--keras_device",
        default="cpu",
        type=device_regex,
        help=(
            "The device where keras model will be loaded. can be one of: "
            "'cpu', 'cuda:0', 'cuda:1', ...'cuda:n'"
        ),
    )
    parser.add_argument(
        "--validate_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help=(
            "The dtype of the two models while validating numerics. "
            "can be 'float32', 'float16', or 'bfloat16'"
        ),
    )
    parser.add_argument(
        "--save_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help=(
            "The dtype that keras model will be saved with. "
            "can be 'float32', 'float16', or 'bfloat16'"
        ),
    )
    parser.add_argument(
        "--upload_link",
        type=str,
        help=(
            "The link to upload the model. can be in these formats: "
            "`kaggle://<KAGGLE_USERNAME>/<MODEL>/<FRAMEWORK>/<VARIATION>`, "
            "`hf://[<HF_USERNAME>/]<MODEL>`"
        ),
    )

    args = parser.parse_args()
    preset = args.preset
    hf_device = args.hf_device
    keras_device = args.keras_device
    validate_dtype = args.validate_dtype
    save_dtype = args.save_dtype
    upload_link = args.upload_link

    print(f"✅ Coverting {preset}.")

    hf_model_name = PRESET_MAP[preset]
    hf_model_dir = download_hf_model(hf_model_name, f"./{preset}_hf_model")
    print("✅ Huggingface model downloaded from the hub.")

    keras_tokenizer = convert_tokenizer(hf_model_dir)
    keras_preprocessor = Phi3Preprocessor(
        tokenizer=keras_tokenizer, sequence_length=20
    )
    # phi3 uses llama tokenizer
    hf_tokenizer = transformers.LlamaTokenizer.from_pretrained(
        hf_model_dir, padding_side="right"
    )

    convert_and_validate(
        hf_model_dir=hf_model_dir,
        hf_device=hf_device,
        keras_device=keras_device,
        validate_dtype=validate_dtype,
        hf_tokenizer=hf_tokenizer,
        keras_preprocessor=keras_preprocessor,
    )

    convert_and_save(
        preset=preset,
        hf_device=hf_device,
        keras_device=keras_device,
        save_dtype=save_dtype,
        hf_model_dir=hf_model_dir,
        hf_tokenizer=hf_tokenizer,
        keras_preprocessor=keras_preprocessor,
    )

    if upload_link is not None:
        upload_preset(upload_link, preset)
        print("✅ Preset uploaded")


if __name__ == "__main__":
    main()
