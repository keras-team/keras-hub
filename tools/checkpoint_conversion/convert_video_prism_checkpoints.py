"""
Convert VideoPrism checkpoints to the Keras format.

The official repo is here:
https://github.com/google-research/videoprism

Setup:

```shell
git clone https://github.com/google-research/videoprism.git
```

Usage:

```shell
python -m tools.checkpoint_conversion.convert_video_prism_checkpoints \
    --preset videoprism_public_v1_base \
    --videoprism_dir ./videoprism
```
"""

import os
import sys

import jax
import jax.numpy as jnp
import keras
import numpy as np
from absl import app
from absl import flags

import keras_hub
from keras_hub.src.models.video_prism.video_prism_backbone import (
    VideoPrismBackbone,
)
from keras_hub.src.models.video_prism.video_prism_image_converter import (
    VideoPrismImageConverter,
)
from keras_hub.src.models.video_prism.video_prism_tokenizer import (
    VideoPrismTokenizer,
)

FLAGS = flags.FLAGS

# Map KerasHub preset names to VideoPrism config names and Model IDs
CONFIG_MAP = {
    "videoprism_public_v1_base": {
        "image_size": 288,
        "num_frames": 16,
        "type": "video",
        "official_config_key": "videoprism_v1_base",
        "official_model_id": "videoprism_public_v1_base",
    },
    "videoprism_public_v1_large": {
        "image_size": 288,
        "num_frames": 8,
        "type": "video",
        "official_config_key": "videoprism_v1_large",
        "official_model_id": "videoprism_public_v1_large",
    },
    "videoprism_lvt_public_v1_base": {
        "image_size": 288,
        "num_frames": 16,
        "vocabulary_size": 32000,
        "type": "video_text",
        "official_config_key": "videoprism_lvt_v1_base",
        "official_model_id": "videoprism_lvt_public_v1_base",
    },
    "videoprism_lvt_public_v1_large": {
        "image_size": 288,
        "num_frames": 8,
        "vocabulary_size": 32000,
        "type": "video_text",
        "official_config_key": "videoprism_lvt_v1_large",
        "official_model_id": "videoprism_lvt_public_v1_large",
    },
}

flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(CONFIG_MAP.keys())}",
    required=True,
)
flags.DEFINE_string(
    "videoprism_dir",
    None,
    "Path to the cloned videoprism repository.",
    required=False,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Optional. Could be "kaggle://keras/{variant}/keras/{preset}"',
    required=False,
)


def convert_model(preset, config):
    num_frames = CONFIG_MAP[preset]["num_frames"]
    image_size = CONFIG_MAP[preset]["image_size"]
    vocabulary_size = CONFIG_MAP[preset].get("vocabulary_size", 0)
    num_auxiliary_layers = config.get("num_auxiliary_layers", 0)
    num_text_layers = config.get("num_unimodal_layers", 0)
    seq_length = 32

    keras_model = VideoPrismBackbone(
        num_frames=num_frames,
        patch_size=config["patch_size"],
        hidden_dim=config["model_dim"],
        intermediate_dim=config["mlp_dim"],
        num_heads=config["num_heads"],
        num_spatial_layers=config["num_spatial_layers"],
        num_temporal_layers=config["num_temporal_layers"],
        num_auxiliary_layers=num_auxiliary_layers,
        vocabulary_size=vocabulary_size,
        num_text_layers=num_text_layers,
        dropout_rate=0.0,
        attention_dropout=0.0,
        attention_logit_soft_cap=config["atten_logit_cap"],
        image_shape=(image_size, image_size, 3),
    )

    # Build
    if num_text_layers > 0:
        keras_inputs = {
            "pixel_values": np.random.uniform(
                size=(1, num_frames, image_size, image_size, 3)
            ).astype(np.float32),
            "token_ids": np.random.randint(
                0, vocabulary_size, size=(1, seq_length)
            ).astype(np.int32),
            "padding_mask": np.zeros((1, seq_length), dtype=np.float32),
        }
    else:
        keras_inputs = np.random.uniform(
            size=(1, num_frames, image_size, image_size, 3)
        ).astype(np.float32)
    keras_model(keras_inputs)

    return keras_model


def convert_encoder_stack(keras_encoder, flax_stack_params, num_layers):
    use_scan = "x_layers" in flax_stack_params

    for i in range(num_layers):
        keras_block = keras_encoder.encoder_layers[i]
        if use_scan:
            flax_block = jax.tree_util.tree_map(
                lambda x: x[i], flax_stack_params["x_layers"]
            )
        else:
            flax_block = flax_stack_params[f"x_layers_{i}"]

        # LN
        ln1_params = flax_block["layer_norm"]
        scale = np.array(ln1_params["scale"]) + 1.0
        bias = np.array(ln1_params["bias"])
        keras_block.layer_norm_1.set_weights([scale, bias])

        # MHA
        mha_params = flax_block["self_attention"]
        q_w = np.array(mha_params["query"]["w"])
        q_b = np.array(mha_params["query"]["b"])
        k_w = np.array(mha_params["key"]["w"])
        k_b = np.array(mha_params["key"]["b"])
        v_w = np.array(mha_params["value"]["w"])
        v_b = np.array(mha_params["value"]["b"])
        o_w = np.array(mha_params["post"]["w"]).transpose(1, 2, 0)
        o_b = np.array(mha_params["post"]["b"])
        keras_block.mha.query_dense.set_weights([q_w, q_b])
        keras_block.mha.key_dense.set_weights([k_w, k_b])
        keras_block.mha.value_dense.set_weights([v_w, v_b])
        keras_block.mha.output_dense.set_weights([o_w, o_b])

        # LN 2
        ff_params = flax_block["ff_layer"]
        ln2_params = ff_params["layer_norm"]
        keras_block.layer_norm_2.set_weights(
            [np.array(ln2_params["scale"]) + 1.0, np.array(ln2_params["bias"])]
        )

        # MLP
        mlp1_params = ff_params["ffn_layer1"]["linear"]
        if "w" in mlp1_params:
            w1 = np.array(mlp1_params["w"])
            b1 = np.array(mlp1_params["b"])
        else:
            w1 = np.array(mlp1_params["kernel"])
            b1 = np.array(mlp1_params["bias"])
        keras_block.mlp.dense_1.set_weights([w1, b1])

        mlp2_params = ff_params["ffn_layer2"]["linear"]
        if "w" in mlp2_params:
            w2 = np.array(mlp2_params["w"])
            b2 = np.array(mlp2_params["b"])
        else:
            w2 = np.array(mlp2_params["kernel"])
            b2 = np.array(mlp2_params["bias"])
        keras_block.mlp.dense_2.set_weights([w2, b2])


def convert_weights(keras_model, flax_params):
    if "params" in flax_params:
        params = flax_params["params"]
    else:
        params = flax_params

    # Determine if it's Video-Only or Video-Text
    has_text_encoder = keras_model.num_text_layers > 0

    # === Vision Encoder ===
    if "vision_encoder" in params:
        vision_params = params["vision_encoder"]
    else:
        vision_params = params.get("vision_encoder", params)

    vision_backbone = keras_model

    # Patch Projection
    patch_proj_params = vision_params["patch_projection"]
    if "linear" in patch_proj_params:
        patch_proj_params = patch_proj_params["linear"]
    if "w" in patch_proj_params:
        pp_kernel = patch_proj_params["w"]
        pp_bias = patch_proj_params["b"]
    else:
        pp_kernel = patch_proj_params["kernel"]
        pp_bias = patch_proj_params["bias"]
    pp_kernel = np.array(pp_kernel)
    pp_bias = np.array(pp_bias)
    vision_backbone.get_layer(
        "spatial_patching_and_embedding"
    ).patch_embedding.set_weights([pp_kernel, pp_bias])

    # Spatial Positional Embedding
    spos_params = vision_params["spatial_pos_emb"]
    spos_emb = np.array(spos_params["emb_var"])
    if len(spos_emb.shape) == 3 and spos_emb.shape[0] == 1:
        spos_emb = spos_emb[0]
    vision_backbone.get_layer(
        "spatial_patching_and_embedding"
    ).position_embedding.set_weights([spos_emb])

    # Spatial Encoder
    spatial_params = vision_params["spatial_encoder"]["transformers_stack"]
    convert_encoder_stack(
        vision_backbone.get_layer("spatial_encoder"),
        spatial_params,
        vision_backbone.num_spatial_layers,
    )

    # Spatial LN
    sln_params = vision_params["spatial_ln"]
    vision_backbone.get_layer("spatial_encoder").layer_norm.set_weights(
        [np.array(sln_params["scale"]) + 1.0, np.array(sln_params["bias"])]
    )

    # Temporal Embedding
    tpos_params = vision_params["temporal_pos_emb"]
    tpos_emb = np.array(tpos_params["emb_var"])
    if len(tpos_emb.shape) == 3 and tpos_emb.shape[0] == 1:
        tpos_emb = tpos_emb[0]
    vision_backbone.get_layer("temporal_embedding").embedding.set_weights(
        [tpos_emb]
    )

    # Temporal Encoder
    temporal_params = vision_params["temporal_encoder"]["transformers_stack"]
    convert_encoder_stack(
        vision_backbone.get_layer("temporal_encoder"),
        temporal_params,
        vision_backbone.num_temporal_layers,
    )

    # Temporal LN
    tln_params = vision_params["temporal_ln"]
    vision_backbone.get_layer("temporal_encoder").layer_norm.set_weights(
        [np.array(tln_params["scale"]) + 1.0, np.array(tln_params["bias"])]
    )

    # === Text Encoder ===
    if has_text_encoder:
        # Auxiliary Encoder
        if keras_model.num_auxiliary_layers > 0:
            aux_params = params["auxiliary_encoder"]["transformers_stack"]
            convert_encoder_stack(
                keras_model.get_layer("auxiliary_encoder"),
                aux_params,
                keras_model.num_auxiliary_layers,
            )

        # Pooling
        pool_params = params["contrastive_vision_pooler"]
        pool_layer = keras_model.get_layer("video_pooler")

        # Query
        query = np.array(pool_params["pooling_attention_query"])
        pool_layer.pooling_attention_query.assign(query)

        # Attention
        atten_params = pool_params["pooling_attention"]
        q_w = np.array(atten_params["query"]["w"])
        q_b = np.array(atten_params["query"]["b"])
        k_w = np.array(atten_params["key"]["w"])
        k_b = np.array(atten_params["key"]["b"])
        v_w = np.array(atten_params["value"]["w"])
        v_b = np.array(atten_params["value"]["b"])
        o_w = np.array(atten_params["post"]["w"]).transpose(1, 2, 0)
        o_b = np.array(atten_params["post"]["b"])
        pool_layer.pooling_attention.query_dense.set_weights([q_w, q_b])
        pool_layer.pooling_attention.key_dense.set_weights([k_w, k_b])
        pool_layer.pooling_attention.value_dense.set_weights([v_w, v_b])
        pool_layer.pooling_attention.output_dense.set_weights([o_w, o_b])
        if "per_dim_scale" in atten_params:
            pds_param = atten_params["per_dim_scale"]
            if "per_dim_scale" in pds_param:
                pds_val = pds_param["per_dim_scale"]
            else:
                pds_val = pds_param
            per_dim_scale = np.array(pds_val)
            pool_layer.pooling_attention.per_dim_scale_layer.set_weights(
                [per_dim_scale]
            )

        # LN
        ln_params = pool_params["pooling_attention_layer_norm"]
        pool_layer.layer_norm.set_weights(
            [np.array(ln_params["scale"]) + 1.0, np.array(ln_params["bias"])]
        )

        # === Text Encoder ===
        text_params = params["text_encoder"]

        # Text embedding
        embedding_layer = keras_model.get_layer("text_embedding")
        emb_var = np.array(text_params["token_emb"]["emb_var"])
        embedding_layer.token_embedding.set_weights([emb_var])
        cls_emb = np.array(text_params["cls_emb"])
        embedding_layer.class_token.set_weights([cls_emb])

        # Text encoding
        flax_stack = text_params["unimodal_transformer"]
        convert_encoder_stack(
            keras_model.get_layer("text_encoder"),
            flax_stack,
            keras_model.num_text_layers,
        )

        # Text LN
        ln_params = text_params["unimodal_ln"]
        keras_model.get_layer("text_layer_normalization").set_weights(
            [np.array(ln_params["scale"]) + 1.0, np.array(ln_params["bias"])]
        )


def convert_image_converter(preset):
    image_size = CONFIG_MAP[preset]["image_size"]
    return VideoPrismImageConverter(
        image_size=(image_size, image_size),
        scale=[1.0 / 255.0],
        offset=None,
        interpolation="bilinear",
        antialias=True,
    )


def convert_tokenizer():
    import tensorflow as tf

    model_path = "gs://t5-data/vocabs/cc_en.32000/sentencepiece.model"
    with tf.io.gfile.GFile(model_path, "rb") as f:
        proto = f.read()
    return VideoPrismTokenizer(proto=proto)


def validate_output(keras_model, flax_params, preset):
    from videoprism import models as vp_models

    config = CONFIG_MAP[preset]
    model_type = config["type"]
    official_model_id = config["official_model_id"]
    flax_model_fn = vp_models.get_model(official_model_id)

    # Generate Inputs
    batch_size = 1
    num_frames = config["num_frames"]
    image_size = config["image_size"]
    x_video_np = np.ones(
        (batch_size, num_frames, image_size, image_size, 3), dtype="float32"
    )

    if model_type == "video":
        print("Running Flax Video model...")
        flax_output, _ = flax_model_fn.apply(
            flax_params, jnp.array(x_video_np), train=False
        )

        print("Running Keras Video model...")
        keras_output = keras_model(x_video_np)
        keras_output_reshaped = np.reshape(keras_output, flax_output.shape)

        diff = np.abs(keras_output_reshaped - flax_output)
        print(f"  Final Max difference: {np.max(diff)}")
        print(f"  Final Mean difference: {np.mean(diff)}")
        if np.max(diff) < 2e-3:
            print("✅ SUCCESS: Numerics match!")
        else:
            print("❌ FAILURE: Numerics do not match.")

    elif model_type == "video_text":
        seq_length = 32
        x_text_np = np.ones((batch_size, seq_length), dtype=np.int32)
        paddings_np = np.zeros((batch_size, seq_length), dtype=np.float32)

        print("Running Flax Video-Text model...")
        flax_video_emb, flax_text_emb, _ = flax_model_fn.apply(
            flax_params,
            inputs=jnp.array(x_video_np),
            text_token_ids=jnp.array(x_text_np),
            text_paddings=jnp.array(paddings_np),
            train=False,
        )

        print("Running Keras Video-Text model...")
        keras_inputs = {
            "pixel_values": x_video_np,
            "token_ids": x_text_np,
            "padding_mask": 1.0 - paddings_np,
        }
        keras_outputs = keras_model(keras_inputs)
        keras_vision_emb = keras.ops.normalize(
            keras_outputs["vision_embeddings"]
        )
        keras_text_emb = keras.ops.normalize(keras_outputs["text_embeddings"])

        diff_video = np.abs(keras_vision_emb - flax_video_emb)
        diff_text = np.abs(keras_text_emb - flax_text_emb)
        print(f"Video Max Diff: {np.max(diff_video)}")
        print(f"Text Max Diff: {np.max(diff_text)}")
        if np.max(diff_video) < 2e-3 and np.max(diff_text) < 2e-3:
            print("✅ SUCCESS: Numerics match!")
        else:
            print("❌ FAILURE: Numerics do not match.")


def main(_):
    preset = FLAGS.preset
    videoprism_dir = FLAGS.videoprism_dir
    videoprism_path = os.path.abspath(videoprism_dir)
    if not os.path.exists(videoprism_path):
        raise ValueError(f"VideoPrism directory not found at {videoprism_path}")
    sys.path.append(videoprism_path)
    print(f"Added {videoprism_path} to sys.path")

    if preset not in CONFIG_MAP:
        raise ValueError(f"Invalid preset {preset}")

    print(f"🏃 Converting {preset}")
    try:
        from videoprism import models as vp_models

        official_model_id = CONFIG_MAP[preset]["official_model_id"]
        official_config_id = CONFIG_MAP[preset]["official_config_key"]
        print(f"Downloading/Loading weights for {official_model_id}...")
        flax_params = vp_models.load_pretrained_weights(official_model_id)
        config = vp_models.CONFIGS[official_config_id]
    except ImportError:
        raise ImportError(
            "Could not import videoprism. "
            "Please provide --videoprism_dir pointing to the repo."
        )

    keras_model = convert_model(preset, config)
    keras_model.summary()
    has_text_encoder = keras_model.num_text_layers > 0
    keras_image_converter = convert_image_converter(preset)
    if has_text_encoder:
        keras_tokenizer = convert_tokenizer()
    print("✅ KerasHub model loaded.")

    convert_weights(keras_model, flax_params)
    print("✅ Weights converted")

    validate_output(keras_model, flax_params, preset)
    print("✅ Output validated.")

    keras_model.save_to_preset(preset)
    keras_image_converter.save_to_preset(f"./{preset}")
    if has_text_encoder:
        keras_tokenizer.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
