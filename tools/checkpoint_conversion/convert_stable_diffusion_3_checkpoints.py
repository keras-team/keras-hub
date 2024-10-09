"""Convert StableDiffusion3 checkpoints.

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_stable_diffusion_3_checkpoints.py \
    --preset stable_diffusion_3_medium --upload_uri kaggle://kerashub/stablediffusion3/keras/stable_diffusion_3_medium
"""

import os
import shutil

import keras
import numpy as np
from absl import app
from absl import flags
from PIL import Image

import keras_hub
from keras_hub.src.models.clip.clip_preprocessor import CLIPPreprocessor
from keras_hub.src.models.clip.clip_text_encoder import CLIPTextEncoder
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (
    StableDiffusion3Backbone,
)
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_text_to_image import (
    StableDiffusion3TextToImage,
)
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_text_to_image_preprocessor import (
    StableDiffusion3TextToImagePreprocessor,
)
from keras_hub.src.models.vae.vae_backbone import VAEBackbone
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

FLAGS = flags.FLAGS

PRESET_MAP = {
    "stable_diffusion_3_medium": {
        # HF root
        "root": "hf://stabilityai/stable-diffusion-3-medium",
        # Model <-> Path
        "clip_l": "text_encoders/clip_l.safetensors",
        "clip_g": "text_encoders/clip_g.safetensors",
        "diffuser": "sd3_medium.safetensors",
        "vae": "sd3_medium.safetensors",
        # Tokenizer
        "clip_tokenizer": "hf://openai/clip-vit-large-patch14",
    }
}

flags.DEFINE_string(
    "preset",
    None,
    f'Must be one of {",".join(PRESET_MAP.keys())}',
    required=True,
)
flags.DEFINE_string(
    "output_dir",
    "output_dir",
    "The generated image will be saved here.",
    required=False,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/{variant}/keras/{preset}"',
    required=False,
)


def convert_model(preset, height, width):
    # The vae and text encoders are common in all presets.
    vae = VAEBackbone(
        [128, 256, 512, 512],
        [2, 2, 2, 2],
        [512, 512, 256, 128],
        [3, 3, 3, 3],
        name="vae",
    )
    clip_l = CLIPTextEncoder(
        49408,
        768,
        768,
        12,
        12,
        3072,
        "quick_gelu",
        -2,
        name="clip_l",
    )
    clip_g = CLIPTextEncoder(
        49408,
        1280,
        1280,
        32,
        20,
        5120,
        "gelu",
        -2,
        name="clip_g",
    )
    # TODO: Add T5.

    # Currently, we hardcode the model arch by preset.
    if preset == "stable_diffusion_3_medium":
        backbone = StableDiffusion3Backbone(
            2,
            64 * 24,
            24,
            24,
            192,
            vae,
            clip_l,
            clip_g,
            height=height,
            width=width,
            name="stable_diffusion_3_backbone",
        )
    return backbone


def convert_preprocessor():
    tokenizer_content = load_json(
        "hf://openai/clip-vit-large-patch14", "tokenizer.json"
    )
    vocabulary = tokenizer_content["model"]["vocab"]
    merges = tokenizer_content["model"]["merges"]
    clip_l_tokenizer = CLIPTokenizer(
        vocabulary,
        merges,
        pad_with_end_token=True,
        config_file="clip_l_tokenizer.json",
        name="clip_l_tokenizer",
    )
    clip_g_tokenizer = CLIPTokenizer(
        vocabulary,
        merges,
        config_file="clip_g_tokenizer.json",
        name="clip_g_tokenizer",
    )
    clip_l_preprocessor = CLIPPreprocessor(
        clip_l_tokenizer,
        config_file="clip_l_preprocessor.json",
        name="clip_l_preprocessor",
    )
    clip_g_preprocessor = CLIPPreprocessor(
        clip_g_tokenizer,
        config_file="clip_g_preprocessor.json",
        name="clip_g_preprocessor",
    )
    preprocessor = StableDiffusion3TextToImagePreprocessor(
        clip_l_preprocessor,
        clip_g_preprocessor,
        name="stable_diffusion_3_text_to_image_preprocessor",
    )
    return preprocessor


def convert_weights(preset, keras_model):
    # Define helper functions.
    def port_conv2d(loader, keras_variable, hf_weight_key):
        loader.port_weight(
            keras_variable.kernel,
            f"{hf_weight_key}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        loader.port_weight(keras_variable.bias, f"{hf_weight_key}.bias")

    def port_dense(loader, keras_variable, hf_weight_key):
        loader.port_weight(
            keras_variable.kernel,
            f"{hf_weight_key}.weight",
            hook_fn=lambda x, _: x.T,
        )
        loader.port_weight(keras_variable.bias, f"{hf_weight_key}.bias")

    def port_mha(loader, keras_variable, hf_weight_key, num_heads, hidden_dim):
        # query
        loader.port_weight(
            keras_variable.query_dense.kernel,
            f"{hf_weight_key}.q_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        loader.port_weight(
            keras_variable.query_dense.bias,
            f"{hf_weight_key}.q_proj.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # key
        loader.port_weight(
            keras_variable.key_dense.kernel,
            f"{hf_weight_key}.k_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        loader.port_weight(
            keras_variable.key_dense.bias,
            f"{hf_weight_key}.k_proj.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # value
        loader.port_weight(
            keras_variable.value_dense.kernel,
            f"{hf_weight_key}.v_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        loader.port_weight(
            keras_variable.value_dense.bias,
            f"{hf_weight_key}.v_proj.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # output
        loader.port_weight(
            keras_variable.output_dense.kernel,
            f"{hf_weight_key}.out_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, hidden_dim // num_heads, hidden_dim)
            ),
        )
        loader.port_weight(
            keras_variable.output_dense.bias, f"{hf_weight_key}.out_proj.bias"
        )

    def port_ln_or_gn(loader, keras_variable, hf_weight_key):
        loader.port_weight(keras_variable.gamma, f"{hf_weight_key}.weight")
        loader.port_weight(keras_variable.beta, f"{hf_weight_key}.bias")

    def port_clip(preset, filename, model, projection_layer):
        with SafetensorLoader(preset, prefix="", fname=filename) as loader:
            # Embeddings
            embedding = model.embedding
            loader.port_weight(
                embedding.token_embedding._embeddings,
                "text_model.embeddings.token_embedding.weight",
            )
            loader.port_weight(
                embedding.position_embedding.position_embeddings,
                "text_model.embeddings.position_embedding.weight",
            )

            # Encoders
            encoder_layers = model.encoder_layers
            for i in range(len(encoder_layers)):
                prefix = "text_model.encoder.layers"
                num_heads = encoder_layers[i].num_heads
                hidden_dim = encoder_layers[i].hidden_dim
                port_mha(
                    loader,
                    encoder_layers[i].attention,
                    f"{prefix}.{i}.self_attn",
                    num_heads,
                    hidden_dim,
                )
                port_ln_or_gn(
                    loader,
                    encoder_layers[i].layer_norm_1,
                    f"{prefix}.{i}.layer_norm1",
                )
                port_ln_or_gn(
                    loader,
                    encoder_layers[i].layer_norm_2,
                    f"{prefix}.{i}.layer_norm2",
                )
                port_dense(
                    loader, encoder_layers[i].dense_1, f"{prefix}.{i}.mlp.fc1"
                )
                port_dense(
                    loader, encoder_layers[i].dense_2, f"{prefix}.{i}.mlp.fc2"
                )

            # Output layers
            port_ln_or_gn(
                loader, model.layer_norm, "text_model.final_layer_norm"
            )
            try:
                loader.port_weight(
                    projection_layer.dense.kernel,
                    "text_projection.weight",
                    hook_fn=lambda x, _: x.T,
                )
            except Exception:
                pass
        return model

    def port_diffuser(preset, filename, model):
        hf_prefix = "model.diffusion_model."
        with SafetensorLoader(
            preset, prefix=hf_prefix, fname=filename
        ) as loader:
            # Embeddings
            port_conv2d(
                loader, model.patch_embedding.patch_embedding, "x_embedder.proj"
            )
            loader.port_weight(
                model.position_embedding.position_embeddings,
                "pos_embed",
                hook_fn=lambda x, _: x[0],
            )
            port_dense(loader, model.context_embedding, "context_embedder")
            port_dense(
                loader, model.vector_embedding.dense1, "y_embedder.mlp.0"
            )
            port_dense(
                loader, model.vector_embedding.dense2, "y_embedder.mlp.2"
            )
            port_dense(
                loader,
                model.timestep_embedding.mlp.dense1,
                "t_embedder.mlp.0",
            )
            port_dense(
                loader,
                model.timestep_embedding.mlp.dense2,
                "t_embedder.mlp.2",
            )

            # Blocks
            num_layers = model.num_layers
            for i in range(num_layers):
                x_block = model.joint_blocks[i].x_block
                context_block = model.joint_blocks[i].context_block
                for block_name, block in (
                    ("x_block", x_block),
                    ("context_block", context_block),
                ):
                    prefix = f"joint_blocks.{i}.{block_name}"
                    port_dense(
                        loader,
                        block.ada_layer_norm.dense,
                        f"{prefix}.adaLN_modulation.1",
                    )
                    port_dense(
                        loader, block.attention_qkv, f"{prefix}.attn.qkv"
                    )

                    if block_name == "context_block" and (i == num_layers - 1):
                        continue

                    port_dense(
                        loader, block.attention_proj, f"{prefix}.attn.proj"
                    )
                    port_dense(loader, block.mlp.dense1, f"{prefix}.mlp.fc1")
                    port_dense(loader, block.mlp.dense2, f"{prefix}.mlp.fc2")

            # Output layer
            port_dense(
                loader,
                model.output_ada_layer_norm.dense,
                "final_layer.adaLN_modulation.1",
            )
            port_dense(loader, model.output_dense, "final_layer.linear")
        return model

    def port_vae(preset, filename, model):
        hf_prefix = "first_stage_model."

        def port_resnet_block(loader, keras_variable, hf_weight_key):
            port_ln_or_gn(
                loader, keras_variable.norm1, f"{hf_weight_key}.norm1"
            )
            port_conv2d(loader, keras_variable.conv1, f"{hf_weight_key}.conv1")
            port_ln_or_gn(
                loader, keras_variable.norm2, f"{hf_weight_key}.norm2"
            )
            port_conv2d(loader, keras_variable.conv2, f"{hf_weight_key}.conv2")
            if hasattr(keras_variable, "residual_projection"):
                port_conv2d(
                    loader,
                    keras_variable.residual_projection,
                    f"{hf_weight_key}.nin_shortcut",
                )

        def port_attention(loader, keras_variable, hf_weight_key):
            port_ln_or_gn(
                loader, keras_variable.group_norm, f"{hf_weight_key}.norm"
            )
            port_conv2d(
                loader, keras_variable.query_conv2d, f"{hf_weight_key}.q"
            )
            port_conv2d(loader, keras_variable.key_conv2d, f"{hf_weight_key}.k")
            port_conv2d(
                loader, keras_variable.value_conv2d, f"{hf_weight_key}.v"
            )
            port_conv2d(
                loader,
                keras_variable.output_conv2d,
                f"{hf_weight_key}.proj_out",
            )

        # Port encdoer.
        with SafetensorLoader(
            preset, prefix=hf_prefix, fname=filename
        ) as loader:
            encoder = keras_model.vae.encoder
            # Stem.
            port_conv2d(loader, encoder.input_projection, "encoder.conv_in")

            # Blocks.
            blocks_idx = 0
            downsamples_idx = 0
            for i, _ in enumerate(encoder.stackwise_num_filters):
                for j in range(encoder.stackwise_num_blocks[i]):
                    prefix = f"encoder.down.{i}.block.{j}"
                    port_resnet_block(
                        loader, encoder.blocks[blocks_idx], prefix
                    )
                    blocks_idx += 1
                if i != len(encoder.stackwise_num_filters) - 1:
                    port_conv2d(
                        loader,
                        encoder.downsamples[downsamples_idx + 1],
                        f"encoder.down.{i}.downsample.conv",
                    )
                    downsamples_idx += 2  # Skip `ZeroPadding2D`.

            # Output layers
            port_resnet_block(
                loader, encoder.mid_block_0, "encoder.mid.block_1"
            )
            port_attention(loader, encoder.mid_attention, "encoder.mid.attn_1")
            port_resnet_block(
                loader, encoder.mid_block_1, "encoder.mid.block_2"
            )
            port_ln_or_gn(loader, encoder.output_norm, "encoder.norm_out")
            port_conv2d(loader, encoder.output_projection, "encoder.conv_out")

        # Port decoder.
        with SafetensorLoader(
            preset, prefix=hf_prefix, fname=filename
        ) as loader:
            decoder = keras_model.vae.decoder
            # Stem.
            port_conv2d(loader, decoder.input_projection, "decoder.conv_in")
            port_resnet_block(
                loader, decoder.mid_block_0, "decoder.mid.block_1"
            )
            port_attention(loader, decoder.mid_attention, "decoder.mid.attn_1")
            port_resnet_block(
                loader, decoder.mid_block_1, "decoder.mid.block_2"
            )

            # Blocks.
            blocks_idx = 0
            upsamples_idx = 0
            for i, _ in enumerate(decoder.stackwise_num_filters):
                for j in range(decoder.stackwise_num_blocks[i]):
                    n = len(decoder.stackwise_num_blocks) - 1
                    prefix = f"decoder.up.{n-i}.block.{j}"
                    port_resnet_block(
                        loader, decoder.blocks[blocks_idx], prefix
                    )
                    blocks_idx += 1
                if i != len(decoder.stackwise_num_filters) - 1:
                    port_conv2d(
                        loader,
                        decoder.upsamples[upsamples_idx + 1],
                        f"decoder.up.{n-i}.upsample.conv",
                    )
                    upsamples_idx += 2  # Skip `UpSampling2D`.

            # Output layers
            port_ln_or_gn(loader, decoder.output_norm, "decoder.norm_out")
            port_conv2d(loader, decoder.output_projection, "decoder.conv_out")

        return model

    # Start conversion.
    config = PRESET_MAP[preset]
    port_clip(
        config["root"],
        config["clip_l"],
        keras_model.clip_l,
        keras_model.clip_l_projection,
    )
    port_clip(
        config["root"],
        config["clip_g"],
        keras_model.clip_g,
        keras_model.clip_g_projection,
    )
    port_diffuser(config["root"], config["diffuser"], keras_model.diffuser)
    port_vae(config["root"], config["vae"], keras_model.vae)


def validate_output(keras_model, keras_preprocessor, output_dir):
    # TODO: Verify the numerics.
    prompt = "A cat holding a sign that says hello world"
    text_to_image = StableDiffusion3TextToImage(keras_model, keras_preprocessor)
    image = text_to_image.generate(prompt, seed=42)
    image = Image.fromarray(image)
    image.save(os.path.join(output_dir, "test.png"))


def main(_):
    preset = FLAGS.preset
    output_dir = FLAGS.output_dir
    if os.path.exists(preset):
        shutil.rmtree(preset)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(preset)
    os.makedirs(output_dir)

    print(f"🏃 Coverting {preset}")

    # Currently SD3 weights are float16 (and have much faster download
    # times for it). We follow suit with Keras weights.
    keras.config.set_dtype_policy("float16")
    height, width = 800, 800  # Use a smaller image size to speed up generation.

    keras_preprocessor = convert_preprocessor()
    keras_model = convert_model(preset, height, width)
    print("✅ KerasHub model loaded.")

    convert_weights(preset, keras_model)
    print("✅ Weights converted.")

    validate_output(keras_model, keras_preprocessor, output_dir)
    print("✅ Output validated.")

    keras_preprocessor.save_to_preset(preset)
    # Set the image size to 1024, the same as in huggingface/diffusers.
    keras_model.height = 1024
    keras_model.width = 1024
    keras_model.save_to_preset(preset)
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
