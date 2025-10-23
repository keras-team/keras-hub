"""Convert PARSeq checkpoints from https://github.com/baudm/parseq.

Make sure to install `pip install pytorch_lighning` for checkpoint convertion.

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_parseq_checkpoints.py \
    --preset parseq
"""

import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image

import keras_hub
from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.models.parseq.parseq_causal_lm import PARSeqCausalLM
from keras_hub.src.models.parseq.parseq_causal_lm_preprocessor import (
    PARSeqCausalLMPreprocessor,
)
from keras_hub.src.models.parseq.parseq_image_converter import (
    PARSeqImageConverter,
)
from keras_hub.src.models.parseq.parseq_tokenizer import PARSeqTokenizer
from keras_hub.src.models.vit.vit_backbone import ViTBackbone

FLAGS = flags.FLAGS

PRESET_MAP = {"parseq": "baudm/parseq"}

flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
    required=True,
)

flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/{variant}/keras/{preset}"',
    required=False,
)


def get_keras_backbone():
    # Config ref: https://github.com/baudm/parseq/blob/main/configs/model/parseq.yaml # noqa: E501
    image_encoder = ViTBackbone(
        image_shape=(32, 128, 3),
        patch_size=(4, 8),
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=384 * 4,
        use_class_token=False,
        name="encoder",
    )
    backbone = PARSeqBackbone(
        vocabulary_size=97,
        max_label_length=25,
        image_encoder=image_encoder,
        num_decoder_heads=12,
        num_decoder_layers=1,
        decoder_hidden_dim=384,
        decoder_mlp_dim=4 * 384,
    )

    return backbone


def convert_backbone_weights(backbone, torch_model):
    state_dict = torch_model.state_dict()
    state_dict.update(torch_model.named_buffers())

    # Helper functions.
    def port_weights(keras_variable, weight_key, hook_fn=None):
        torch_tensor = state_dict[weight_key].cpu().numpy()
        if hook_fn:
            torch_tensor = hook_fn(torch_tensor, list(keras_variable.shape))
        keras_variable.assign(torch_tensor)

    def port_ln(keras_variable, weight_key):
        port_weights(keras_variable.gamma, f"{weight_key}.weight")
        port_weights(keras_variable.beta, f"{weight_key}.bias")

    def port_dense(keras_variable, weight_key):
        port_weights(
            keras_variable.kernel,
            f"{weight_key}.weight",
            hook_fn=lambda x, _: x.T,
        )
        if keras_variable.bias is not None:
            port_weights(keras_variable.bias, f"{weight_key}.bias")

    def port_mha(
        keras_variable, weight_key, num_heads, hidden_dim, encoder=True
    ):
        # Attention layer.
        if encoder:
            fused_qkv_kernel = state_dict[f"{weight_key}.attn.qkv.weight"].t()
            fused_qkv_bias = (
                state_dict[f"{weight_key}.attn.qkv.bias"].cpu().numpy()
            )
        else:
            fused_qkv_kernel = state_dict[f"{weight_key}.in_proj_weight"].t()
            fused_qkv_bias = (
                state_dict[f"{weight_key}.in_proj_bias"].cpu().numpy()
            )

        head_dim = hidden_dim // num_heads

        # Kernel
        query_kernel = fused_qkv_kernel[:, :hidden_dim]
        query_kernel = query_kernel.reshape(hidden_dim, num_heads, head_dim)

        key_kernel = fused_qkv_kernel[
            :, hidden_dim : hidden_dim + num_heads * head_dim
        ]
        key_kernel = key_kernel.reshape(hidden_dim, num_heads, head_dim)

        value_kernel = fused_qkv_kernel[:, hidden_dim + num_heads * head_dim :]
        value_kernel = value_kernel.reshape(hidden_dim, num_heads, head_dim)

        # Bias
        query_bias = fused_qkv_bias[:hidden_dim]
        query_bias = query_bias.reshape(num_heads, head_dim)

        key_bias = fused_qkv_bias[
            hidden_dim : hidden_dim + num_heads * head_dim
        ]
        key_bias = key_bias.reshape(num_heads, head_dim)

        value_bias = fused_qkv_bias[hidden_dim + num_heads * head_dim :]
        value_bias = value_bias.reshape(num_heads, head_dim)

        keras_variable.query_dense.kernel.assign(query_kernel)
        keras_variable.key_dense.kernel.assign(key_kernel)
        keras_variable.value_dense.kernel.assign(value_kernel)

        keras_variable.query_dense.bias.assign(query_bias)
        keras_variable.key_dense.bias.assign(key_bias)
        keras_variable.value_dense.bias.assign(value_bias)

        if encoder:
            keras_variable.output_dense.kernel.assign(
                state_dict[f"{weight_key}.attn.proj.weight"]
                .t()
                .reshape(num_heads, head_dim, hidden_dim)
            )
            keras_variable.output_dense.bias.assign(
                state_dict[f"{weight_key}.attn.proj.bias"].cpu().numpy()
            )
        else:
            keras_variable.output_dense.kernel.assign(
                state_dict[f"{weight_key}.out_proj.weight"]
                .t()
                .reshape(num_heads, head_dim, hidden_dim)
            )
            keras_variable.output_dense.bias.assign(
                state_dict[f"{weight_key}.out_proj.bias"].cpu().numpy()
            )

    # Encoder weight transfer
    port_weights(
        backbone.image_encoder.layers[1].patch_embedding.kernel,
        "model.encoder.patch_embed.proj.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )

    port_weights(
        backbone.image_encoder.layers[1].patch_embedding.bias,
        "model.encoder.patch_embed.proj.bias",
    )

    port_weights(
        backbone.image_encoder.layers[1].position_embedding.embeddings,
        "model.encoder.pos_embed",
        hook_fn=lambda x, _: x[0],
    )
    encoder_layers = backbone.image_encoder.layers[2].encoder_layers
    for i, encoder_block in enumerate(encoder_layers):
        prefix = "model.encoder.blocks"
        num_heads = encoder_block.num_heads
        hidden_dim = encoder_block.hidden_dim

        # Decompose fused multihead attention layer from torch
        port_mha(
            encoder_block.mha,
            f"{prefix}.{i}",
            num_heads,
            hidden_dim,
        )

        port_ln(encoder_block.layer_norm_1, f"{prefix}.{i}.norm1")
        port_ln(encoder_block.layer_norm_2, f"{prefix}.{i}.norm2")

        port_dense(encoder_block.mlp.dense_1, f"{prefix}.{i}.mlp.fc1")
        port_dense(encoder_block.mlp.dense_2, f"{prefix}.{i}.mlp.fc2")
    port_ln(backbone.image_encoder.layers[2].layer_norm, "model.encoder.norm")

    # Decoder weights transfer
    port_weights(
        backbone.layers[4].pos_query_embeddings,
        "model.pos_queries",
    )
    port_weights(
        backbone.layers[4].token_embedding.embeddings,
        "model.text_embed.embedding.weight",
    )

    decoder_layers = backbone.layers[4].decoder_layers
    for i, decoder_block in enumerate(decoder_layers):
        prefix = "model.decoder.layers"
        num_heads = decoder_block.num_heads
        hidden_dim = decoder_block.hidden_dim

        port_mha(
            decoder_block.self_attention,
            f"{prefix}.{i}.self_attn",
            num_heads,
            hidden_dim,
            encoder=False,
        )
        port_mha(
            decoder_block.cross_attention,
            f"{prefix}.{i}.cross_attn",
            num_heads,
            hidden_dim,
            encoder=False,
        )

        port_ln(decoder_block.layer_norm_1, f"{prefix}.{i}.norm1")
        port_ln(decoder_block.layer_norm_2, f"{prefix}.{i}.norm2")
        port_ln(decoder_block.query_layer_norm, f"{prefix}.{i}.norm_q")
        port_ln(decoder_block.content_layer_norm, f"{prefix}.{i}.norm_c")
        port_dense(decoder_block.mlp.dense_1, f"{prefix}.{i}.linear1")
        port_dense(decoder_block.mlp.dense_2, f"{prefix}.{i}.linear2")
    port_ln(backbone.layers[4].layer_norm, "model.decoder.norm")
    port_dense(backbone.layers[5], "model.head")


def convert_image_converter():
    # Basic image transformations done:
    # Ref: https://github.com/baudm/parseq/blob/1902db043c029a7e03a3818c616c06600af574be/strhub/data/module.py#L77 # noqa: E501
    mean, std = 0.5, 0.5
    return PARSeqImageConverter(
        image_size=(32, 128),
        offset=-mean / std,
        scale=1.0 / 255.0 / std,
        interpolation="bicubic",
    )


def validate_output(preprocessor, keras_model, torch_model):
    file = keras.utils.get_file(
        origin="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/480px-Google_2015_logo.svg.png",  # noqa : E501
        fname="google.png",
    )
    image = Image.open(file).convert("RGB")
    images = np.expand_dims(np.array(image).astype("float32"), axis=0)

    x, _, _ = preprocessor({"images": images, "responses": ["Google"]})

    keras_output = keras_model(preprocessor.generate_preprocess(images))
    tgt_in = torch.full((1, 1), torch_model.tokenizer.bos_id, dtype=torch.long)
    torch_output = torch_model.model.head(
        torch_model.model.decode(
            tgt_in,
            torch_model.model.encoder(
                torch.from_numpy(
                    keras.ops.convert_to_numpy(x["images"]).transpose(
                        0, 3, 1, 2
                    )
                )
            ),
        )
    )

    keras_causal_output = [
        "".join(output) for output in keras_model.generate(x["images"])
    ]
    torch_image_input = torch.from_numpy(
        keras.ops.convert_to_numpy(x["images"])
    )
    torch_logits = torch_model(torch_image_input.permute(0, 3, 1, 2))
    torch_causal_output, _ = torch_model.tokenizer.decode(torch_logits)

    print("🔶 Keras Logits Output:", keras_output[0, 0, :10])
    print("🔶 Torch Logits Output:", torch_output[0, 0, :10])
    print("🔶 Keras Causal Output:", keras_causal_output)
    print("🔶 Torch Causal Output:", torch_causal_output)
    assert torch_causal_output[0] == keras_causal_output[0]


def main(_):
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]
    if os.path.exists(preset):
        shutil.rmtree(preset)
    os.makedirs(preset)

    print(f"🏃 Coverting {preset}")

    # Load model and image transforms
    torch_model = torch.hub.load(hf_preset, preset, pretrained=True).eval()
    keras_backbone = get_keras_backbone()
    print("✅ KerasHub backbone loaded.")

    convert_backbone_weights(keras_backbone, torch_model)
    print("✅ Backbone weights converted.")

    keras_image_converter = convert_image_converter()
    keras_tokenizer = PARSeqTokenizer(max_label_length=25)

    parseq_preprocessor = PARSeqCausalLMPreprocessor(
        image_converter=keras_image_converter, tokenizer=keras_tokenizer
    )

    print("✅ Loaded preprocessor configuration.")

    keras_model = PARSeqCausalLM(
        preprocessor=parseq_preprocessor, backbone=keras_backbone
    )

    validate_output(parseq_preprocessor, keras_model, torch_model)
    print("✅ Outputs Validated.")
    keras_model.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
