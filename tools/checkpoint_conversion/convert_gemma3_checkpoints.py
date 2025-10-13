"""
Convert Gemma 3 Flax checkpoints to the Keras format.

Setup:
```shell
pip install -r requirements.txt
pip install --upgrade -q gemma
```

Usage:
```shell
cd tools/checkpoint_conversion
python convert_gemma3_checkpoints.py --preset gemma3_instruct_1b --export_safetensors
python convert_gemma3_checkpoints.py --preset gemma3_instruct_4b --export_safetensors
```
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
# No GPU for conversion, makes memory management easier.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras  # noqa: E402
import tensorflow_datasets as tfds  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from checkpoint_conversion_utils import download_gcs_file
from gemma import gm  # noqa: E402
from keras import ops  # noqa: E402

import keras_hub  # noqa: E402

import numpy as np
import torch
import shutil
import json
from safetensors.torch import save_file
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM



def convert_to_hf_config(keras_config):
    """Convert Keras Gemma config to Hugging Face GemmaConfig."""
    hf_config = transformers.Gemma3TextConfig(
        vocab_size=keras_config.vocabulary_size,
        num_hidden_layers=keras_config.num_layers,
        num_attention_heads=keras_config.num_query_heads,
        num_key_value_heads=keras_config.num_key_value_heads,
        hidden_size=keras_config.hidden_dim,
        intermediate_size=keras_config.intermediate_dim,
        head_dim=keras_config.head_dim,
        max_position_embeddings=32768,
    )
    return hf_config

def export_to_hf(backbone, keras_tokenizer, path):
    """Convert a Keras Gemma model to Hugging Face format and save to path."""

    
    hf_config = convert_to_hf_config(backbone)
    weights_dict = {}

    # Helper function to convert bfloat16 weights to torch tensors
    def to_torch(weight):
        # Convert bfloat16 to float32 first, then to torch, then to bfloat16
        if hasattr(weight.dtype, 'name') and 'bfloat16' in str(weight.dtype):
            weight = np.array(weight, dtype=np.float32)
        return torch.from_numpy(weight).to(torch.bfloat16)

    # Token embeddings
    token_embedding = backbone.get_layer("token_embedding").get_weights()[0]
    weights_dict['model.embed_tokens.weight'] = to_torch(token_embedding)

    for i in range(backbone.num_layers):
        block = backbone.get_layer(f"decoder_block_{i}")
        q_kernel = block.attention.query_dense.get_weights()[0]
        q_kernel = torch.from_numpy(np.array(q_kernel, dtype=np.float32)).to(torch.bfloat16).permute(1, 0, 2).reshape(backbone.hidden_dim, -1).T
        weights_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = q_kernel

        k_kernel = block.attention.key_dense.get_weights()[0]
        k_kernel = torch.from_numpy(np.array(k_kernel, dtype=np.float32)).to(torch.bfloat16).permute(1, 0, 2).reshape(backbone.hidden_dim, -1).T
        weights_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = k_kernel

        v_kernel = block.attention.value_dense.get_weights()[0]
        v_kernel = torch.from_numpy(np.array(v_kernel, dtype=np.float32)).to(torch.bfloat16).permute(1, 0, 2).reshape(backbone.hidden_dim, -1).T
        weights_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = v_kernel

        o_kernel = block.attention.output_dense.get_weights()[0]
        o_kernel = torch.from_numpy(np.array(o_kernel, dtype=np.float32)).to(torch.bfloat16).permute(2, 0, 1).reshape(backbone.hidden_dim, -1)
        weights_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = o_kernel

        q_norm = block.attention.query_norm.get_weights()[0]
        weights_dict[f'model.layers.{i}.self_attn.q_norm.weight'] = to_torch(q_norm)

        k_norm = block.attention.key_norm.get_weights()[0]
        weights_dict[f'model.layers.{i}.self_attn.k_norm.weight'] = to_torch(k_norm)

        gate_kernel = block.gating_ffw.get_weights()[0]
        gate_kernel = torch.from_numpy(np.array(gate_kernel, dtype=np.float32)).to(torch.bfloat16).T
        weights_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = gate_kernel

        up_kernel = block.gating_ffw_2.get_weights()[0]
        up_kernel = torch.from_numpy(np.array(up_kernel, dtype=np.float32)).to(torch.bfloat16).T
        weights_dict[f'model.layers.{i}.mlp.up_proj.weight'] = up_kernel

        down_kernel = block.ffw_linear.get_weights()[0]
        down_kernel = torch.from_numpy(np.array(down_kernel, dtype=np.float32)).to(torch.bfloat16).T
        weights_dict[f'model.layers.{i}.mlp.down_proj.weight'] = down_kernel

        input_layer_norm = block.pre_attention_norm.get_weights()[0]
        weights_dict[f'model.layers.{i}.input_layernorm.weight'] = to_torch(input_layer_norm)

        post_attn_norm = block.post_attention_norm.get_weights()[0]
        weights_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = to_torch(post_attn_norm)

        pre_feedforward_layernorm_weight = block.pre_ffw_norm.get_weights()[0]
        weights_dict[f'model.layers.{i}.pre_feedforward_layernorm.weight'] = to_torch(pre_feedforward_layernorm_weight)

        post_feedforward_layernorm_weight = block.post_ffw_norm.get_weights()[0]
        weights_dict[f'model.layers.{i}.post_feedforward_layernorm.weight'] = to_torch(post_feedforward_layernorm_weight)

    final_norm = backbone.get_layer("final_normalization").get_weights()[0]
    weights_dict['model.norm.weight'] = to_torch(final_norm)
    weights_dict['lm_head.weight'] = weights_dict['model.embed_tokens.weight'].clone()

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(hf_config.to_dict(), f)
    weights_dict = {k: v.contiguous() for k, v in weights_dict.items()}
    save_file(weights_dict, os.path.join(path, "model.safetensors"))
    keras_tokenizer.save_assets(path)
    vocab_spm = os.path.join(path, "vocabulary.spm")
    tokenizer_model = os.path.join(path, "tokenizer.model")
    if os.path.exists(vocab_spm):
        shutil.move(vocab_spm, tokenizer_model)
    print("Export complete! Files saved in:", path)



def load_hf_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer

def infer(model, tokenizer, prompt, device, max_new_tokens=30, temperature=1.0, top_k=50, top_p=1.0):
    # Tokenize inpu
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True # Set do_sample to True when using sampling parameters
            )

    # Decode generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generated_text


keras.utils.set_random_seed(42)

FLAGS = flags.FLAGS

PROMPT_TEMPLATE = """<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
"""


PRESET_MAP = {
    # === Text ===
    # 270M
    "gemma3_instruct_270m": {
        "model": gm.nn.Gemma3_270M,
        "params": gm.ckpts.CheckpointPath.GEMMA3_270M_IT,
    },
    "gemma3_270m": {
        "model": gm.nn.Gemma3_270M,
        "params": gm.ckpts.CheckpointPath.GEMMA3_270M_PT,
    },
    # 1B
    "gemma3_1b": {
        "model": gm.nn.Gemma3_1B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_1B_PT,
    },
    "gemma3_instruct_1b": {
        "model": gm.nn.Gemma3_1B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_1B_IT,
    },
    # 4B
    "gemma3_4b_text": {
        "model": gm.nn.Gemma3_4B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_4B_PT,
    },
    "gemma3_instruct_4b_text": {
        "model": gm.nn.Gemma3_4B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_4B_IT,
    },
    # 12B
    "gemma3_12b_text": {
        "model": gm.nn.Gemma3_12B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_12B_PT,
    },
    "gemma3_instruct_12b_text": {
        "model": gm.nn.Gemma3_12B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_12B_IT,
    },
    # 27B
    "gemma3_27b_text": {
        "model": gm.nn.Gemma3_27B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_27B_PT,
    },
    "gemma3_instruct_27b_text": {
        "model": gm.nn.Gemma3_27B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_27B_IT,
    },
    # === Vision + Text ===
    # 4B
    "gemma3_4b": {
        "model": gm.nn.Gemma3_4B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_4B_PT,
    },
    "gemma3_instruct_4b": {
        "model": gm.nn.Gemma3_4B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_4B_IT,
    },
    # 12b
    "gemma3_12b": {
        "model": gm.nn.Gemma3_12B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_12B_PT,
    },
    "gemma3_instruct_12b": {
        "model": gm.nn.Gemma3_12B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_12B_IT,
    },
    # 27B
    "gemma3_27b": {
        "model": gm.nn.Gemma3_27B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_27B_PT,
    },
    "gemma3_instruct_27b": {
        "model": gm.nn.Gemma3_27B,
        "params": gm.ckpts.CheckpointPath.GEMMA3_27B_IT,
    },
}


flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
    required=True,
)

flags.DEFINE_bool(
    "export_safetensors",
    False,
    "Export model to Safetensors format (HuggingFace-compatible). Only for text-only models.",
    )

def convert_model(flax_config, text_only):
    vision_encoder = None
    if not text_only:
        vision_config = flax_config.vision_encoder
        siglip_config = vision_config.siglip_encoder
        vision_encoder = keras_hub.models.Gemma3VisionEncoder(
            image_size=vision_config.image_height,
            patch_size=siglip_config.patch_size[0],
            num_heads=siglip_config.num_heads,
            hidden_dim=siglip_config.width,
            num_layers=siglip_config.depth,
            intermediate_dim=siglip_config.mlp_dim,
            output_dim=2560,  # not present in Flax config
            pool_size=4,
            layer_norm_epsilon=1e-6,
            dtype="float32",  # Needs to be float32.
        )

    return keras_hub.models.Gemma3Backbone(
        vocabulary_size=flax_config.num_embed,
        image_size=None
        if text_only
        else flax_config.vision_encoder.image_height,
        num_layers=flax_config.num_layers,
        num_query_heads=flax_config.num_heads,
        num_key_value_heads=flax_config.num_kv_heads,
        hidden_dim=flax_config.embed_dim,
        intermediate_dim=flax_config.hidden_dim,
        head_dim=flax_config.head_dim,
        query_head_dim_normalize="BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS"
        not in str(flax_config.query_pre_attn_norm),
        use_post_ffw_norm=flax_config.use_post_ffw_norm,
        use_post_attention_norm=flax_config.use_post_attn_norm,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=flax_config.final_logit_softcap,
        use_sliding_window_attention=True,
        sliding_window_size=flax_config.sliding_window_size,
        local_rope_scaling_factor=flax_config.local_scale_factor,
        global_rope_scaling_factor=flax_config.global_scale_factor,
        vision_encoder=vision_encoder,
        layer_norm_epsilon=1e-6,
        dtype="bfloat16",  # Flax ckpts are in bfloat16, except for SigLIP.
    )


def convert_tokenizer(proto_path):
    return keras_hub.models.Gemma3Tokenizer(proto=proto_path)


def get_image_converter(vision_config):
    # No way to source this from Flax.
    return keras_hub.layers.Gemma3ImageConverter(
        image_size=(vision_config.image_height, vision_config.image_width),
        scale=1 / 127.5,
        crop_to_aspect_ratio=False,
        offset=-1.0,
        interpolation="bilinear",
    )


def convert_vision_encoder_weights(vision_encoder, flax_params):
    num_layers = vision_encoder.num_layers
    hidden_dim = vision_encoder.hidden_dim

    siglip_config = flax_params["vision_encoder"]["siglip_encoder"]

    for i in range(num_layers):
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.key_proj.weights[0].assign(
            ops.reshape(
                ops.squeeze(
                    siglip_config["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["key"]["kernel"]
                ),
                [hidden_dim, -1],
            )
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.key_proj.weights[1].assign(
            ops.reshape(
                ops.squeeze(
                    siglip_config["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["key"]["bias"]
                ),
                [-1],
            )
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.query_proj.weights[0].assign(
            ops.reshape(
                ops.squeeze(
                    siglip_config["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["query"]["kernel"]
                ),
                [hidden_dim, -1],
            )
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.query_proj.weights[1].assign(
            ops.reshape(
                siglip_config["Transformer"][f"encoderblock_{i}"][
                    "MultiHeadDotProductAttention_0"
                ]["query"]["bias"],
                [-1],
            )
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.value_proj.weights[0].assign(
            ops.reshape(
                ops.squeeze(
                    siglip_config["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["value"]["kernel"]
                ),
                [hidden_dim, -1],
            )
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.value_proj.weights[1].assign(
            ops.reshape(
                siglip_config["Transformer"][f"encoderblock_{i}"][
                    "MultiHeadDotProductAttention_0"
                ]["value"]["bias"],
                [-1],
            )
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.out_proj.weights[0].assign(
            ops.reshape(
                siglip_config["Transformer"][f"encoderblock_{i}"][
                    "MultiHeadDotProductAttention_0"
                ]["out"]["kernel"],
                [-1, hidden_dim],
            )
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.out_proj.weights[1].assign(
            ops.reshape(
                ops.squeeze(
                    siglip_config["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["out"]["bias"]
                ),
                [-1],
            )
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_1.weights[0].assign(
            siglip_config["Transformer"][f"encoderblock_{i}"]["LayerNorm_0"][
                "scale"
            ]
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_1.weights[1].assign(
            siglip_config["Transformer"][f"encoderblock_{i}"]["LayerNorm_0"][
                "bias"
            ]
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_2.weights[0].assign(
            siglip_config["Transformer"][f"encoderblock_{i}"]["LayerNorm_1"][
                "scale"
            ]
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_2.weights[1].assign(
            siglip_config["Transformer"][f"encoderblock_{i}"]["LayerNorm_1"][
                "bias"
            ]
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_1.weights[0].assign(
            siglip_config["Transformer"][f"encoderblock_{i}"]["MlpBlock_0"][
                "Dense_0"
            ]["kernel"]
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_1.weights[1].assign(
            siglip_config["Transformer"][f"encoderblock_{i}"]["MlpBlock_0"][
                "Dense_0"
            ]["bias"]
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_2.weights[0].assign(
            siglip_config["Transformer"][f"encoderblock_{i}"]["MlpBlock_0"][
                "Dense_1"
            ]["kernel"]
        )
        vision_encoder.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_2.weights[1].assign(
            siglip_config["Transformer"][f"encoderblock_{i}"]["MlpBlock_0"][
                "Dense_1"
            ]["bias"]
        )
    vision_encoder.get_layer("image_encoder").encoder_layer_norm.weights[
        0
    ].assign(siglip_config["Transformer"]["encoder_norm"]["scale"])
    vision_encoder.get_layer("image_encoder").encoder_layer_norm.weights[
        1
    ].assign(siglip_config["Transformer"]["encoder_norm"]["bias"])
    vision_encoder.get_layer(
        "image_encoder"
    ).vision_embeddings.patch_embedding.weights[0].assign(
        siglip_config["embedding"]["kernel"]
    )
    vision_encoder.get_layer(
        "image_encoder"
    ).vision_embeddings.patch_embedding.weights[1].assign(
        siglip_config["embedding"]["bias"]
    )
    vision_encoder.get_layer(
        "image_encoder"
    ).vision_embeddings.position_embedding.weights[0].assign(
        siglip_config["pos_embedding"][0]
    )

    (
        vision_encoder.get_layer(
            "vision_output_encoder"
        ).vision_input_projection.kernel.assign(
            flax_params["embedder"]["mm_input_projection"]["w"]
        )
    )

    (
        vision_encoder.get_layer(
            "vision_output_encoder"
        ).vision_soft_embedding_norm.scale.assign(
            flax_params["embedder"]["mm_soft_embedding_norm"]["scale"]
        )
    )


def convert_weights(keras_model, flax_params):
    # Token embedding layer (text)
    keras_model.token_embedding.embeddings.assign(
        flax_params["embedder"]["input_embedding"]
    )

    # === Vision encoder ===
    if keras_model.vision_encoder is not None:
        convert_vision_encoder_weights(keras_model.vision_encoder, flax_params)

    # === Common backbone ===

    # Transformer layers
    for i in range(keras_model.num_layers):
        keras_model.transformer_layers[i].pre_attention_norm.scale.assign(
            flax_params[f"layer_{i}"]["pre_attention_norm"]["scale"]
        )

        # QKV projection layers
        keras_model.transformer_layers[i].attention.query_dense.kernel.assign(
            flax_params[f"layer_{i}"]["attn"]["q_einsum"]["w"]
        )
        keras_model.transformer_layers[i].attention.key_dense.kernel.assign(
            flax_params[f"layer_{i}"]["attn"]["kv_einsum"]["w"][0]
        )
        keras_model.transformer_layers[i].attention.value_dense.kernel.assign(
            flax_params[f"layer_{i}"]["attn"]["kv_einsum"]["w"][1]
        )

        # QK RMSNorm layers
        keras_model.transformer_layers[i].attention.query_norm.scale.assign(
            flax_params[f"layer_{i}"]["attn"]["_query_norm"]["scale"]
        )
        keras_model.transformer_layers[i].attention.key_norm.scale.assign(
            flax_params[f"layer_{i}"]["attn"]["_key_norm"]["scale"]
        )

        # Attn output dense layer
        keras_model.transformer_layers[i].attention.output_dense.kernel.assign(
            flax_params[f"layer_{i}"]["attn"]["attn_vec_einsum"]["w"]
        )

        keras_model.transformer_layers[i].post_attention_norm.scale.assign(
            flax_params[f"layer_{i}"]["post_attention_norm"]["scale"]
        )

        keras_model.transformer_layers[i].pre_ffw_norm.scale.assign(
            flax_params[f"layer_{i}"]["pre_ffw_norm"]["scale"]
        )
        keras_model.transformer_layers[i].gating_ffw.kernel.assign(
            flax_params[f"layer_{i}"]["mlp"]["gating_einsum"][0].T
        )
        keras_model.transformer_layers[i].gating_ffw_2.kernel.assign(
            flax_params[f"layer_{i}"]["mlp"]["gating_einsum"][1].T
        )
        keras_model.transformer_layers[i].ffw_linear.kernel.assign(
            flax_params[f"layer_{i}"]["mlp"]["linear"]
        )

        keras_model.transformer_layers[i].post_ffw_norm.scale.assign(
            flax_params[f"layer_{i}"]["post_ffw_norm"]["scale"]
        )

    keras_model.layer_norm.scale.assign(flax_params["final_norm"]["scale"])


def validate_output(
    keras_model,
    keras_tokenizer,
    keras_image_converter,
    flax_model,
    flax_params,
    text_only,
):
    if text_only:
        image = None
        input_str = "What is Keras?"
        length = 50

        preprocessor_kwargs = {
            "num_vision_tokens_per_image": 0,  # hardcoded
            "max_images_per_prompt": 0,  # hardcoded
        }
    else:
        ds = tfds.data_source("oxford_flowers102", split="train")
        image = ds[0]["image"]
        input_str = "What can you say about this image: <start_of_image>?"
        length = 310

        preprocessor_kwargs = {
            "num_vision_tokens_per_image": 256,  # hardcoded
            "max_images_per_prompt": 2,  # hardcoded
        }

    # KerasHub
    preprocessor = keras_hub.models.Gemma3CausalLMPreprocessor(
        tokenizer=keras_tokenizer,
        image_converter=keras_image_converter,
        sequence_length=1024,
        **preprocessor_kwargs,
    )
    gemma_lm = keras_hub.models.Gemma3CausalLM(
        backbone=keras_model,
        preprocessor=preprocessor,
        dtype="bfloat16",  # Flax ckpts are in bfloat16, except for SigLIP.
    )
    keras_output = gemma_lm.generate(
        {
            "images": None if text_only else image,
            "prompts": PROMPT_TEMPLATE.format(question=input_str),
        },
        max_length=length,
    )
    to_skip = keras_output.find("model") + 6
    keras_output = keras_output[to_skip:]
    print("🔶 KerasHub output:", keras_output)

    # Flax
    flax_sampler = gm.text.ChatSampler(
        model=flax_model,
        params=flax_params,
        multi_turn=False,
        cache_length=256 if length <= 256 else 512,
    )
    flax_output = flax_sampler.chat(input_str, images=image)
    print("🔶 Flax output:", flax_output)


def main(_):
    preset = FLAGS.preset

    print(f"🏃 Converting {preset}")

    presets = PRESET_MAP.keys()
    assert preset in presets, (
        f"Invalid preset {preset}. Must be one of {','.join(presets)}"
    )
    text_only = "text" in preset or "1b" in preset or "270m" in preset

    print("🏃 Loading Flax model and tokeniser")
    flax_kwargs = {}
    if text_only and "1b" not in preset and "270m" not in preset:
        flax_kwargs["text_only"] = True
    flax_model = PRESET_MAP[preset]["model"](**flax_kwargs)
    flax_config = flax_model.config
    flax_params = gm.ckpts.load_params(
        PRESET_MAP[preset]["params"], **flax_kwargs
    )
    flax_tokenizer = gm.text.Gemma3Tokenizer()
    proto_path = "./tokenizer_gemma3.model"
    download_gcs_file(
        gcs_uri=flax_tokenizer.path,
        destination_file_name=proto_path,
    )
    print("✅ Flax model loaded")

    keras_tokenizer = convert_tokenizer(proto_path)
    keras_image_converter = None
    if not text_only:
        keras_image_converter = get_image_converter(flax_config.vision_encoder)
    keras_model = convert_model(flax_config, text_only)
    print("✅ Keras model loaded")

    convert_weights(keras_model, flax_params)
    print("✅ Weights converted")

    validate_output(
        keras_model,
        keras_tokenizer,
        keras_image_converter,
        flax_model,
        flax_params,
        text_only,
    )

    keras_model.save_to_preset(preset)
    keras_tokenizer.save_to_preset(preset)
    print(f"🏁 Preset saved to ./{preset}")


    if FLAGS.export_safetensors and text_only:
        export_dir = f"./{preset}_safetensors_export"
        print(f"🏃 Exporting to Safetensors (HuggingFace format) at {export_dir}")
        export_to_hf(keras_model, keras_tokenizer, export_dir)
        print(f"🏁 Safetensors export complete: {export_dir}")

        local_hf_model, local_hf_tokenizer = load_hf_model(export_dir, device="cpu")
        print("Local Hugging Face model loaded successfully!")

        print("🔶 Safetensors output:", infer(local_hf_model, local_hf_tokenizer, "Hello, my name is", "cpu", max_new_tokens=100))
    elif FLAGS.export_safetensors:
        print("⚠️ Safetensors export is only supported for text-only models. Skipping export.")




if __name__ == "__main__":
    app.run(main)
