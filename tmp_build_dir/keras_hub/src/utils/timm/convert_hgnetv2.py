import keras
import numpy as np

from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
from keras_hub.src.models.hgnetv2.hgnetv2_layers import HGNetV2ConvLayerLight

backbone_cls = HGNetV2Backbone


HGNETV2_CONFIGS = {
    "hgnetv2_b0": {
        "stem_channels": [3, 16, 16],
        "stackwise_stage_filters": [
            [16, 16, 64, 1, 3, 3],
            [64, 32, 256, 1, 3, 3],
            [256, 64, 512, 2, 3, 5],
            [512, 128, 1024, 1, 3, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 16,
        "hidden_sizes": [64, 256, 512, 1024],
        "depths": [1, 1, 2, 1],
    },
    "hgnetv2_b1": {
        "stem_channels": [3, 24, 32],
        "stackwise_stage_filters": [
            [32, 32, 64, 1, 3, 3],
            [64, 48, 256, 1, 3, 3],
            [256, 96, 512, 2, 3, 5],
            [512, 192, 1024, 1, 3, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 32,
        "hidden_sizes": [64, 256, 512, 1024],
        "depths": [1, 1, 2, 1],
    },
    "hgnetv2_b2": {
        "stem_channels": [3, 24, 32],
        "stackwise_stage_filters": [
            [32, 32, 96, 1, 4, 3],
            [96, 64, 384, 1, 4, 3],
            [384, 128, 768, 3, 4, 5],
            [768, 256, 1536, 1, 4, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 32,
        "hidden_sizes": [96, 384, 768, 1536],
        "depths": [1, 1, 3, 1],
    },
    "hgnetv2_b3": {
        "stem_channels": [3, 24, 32],
        "stackwise_stage_filters": [
            [32, 32, 128, 1, 5, 3],
            [128, 64, 512, 1, 5, 3],
            [512, 128, 1024, 3, 5, 5],
            [1024, 256, 2048, 1, 5, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 32,
        "hidden_sizes": [128, 512, 1024, 2048],
        "depths": [1, 1, 3, 1],
    },
    "hgnetv2_b4": {
        "stem_channels": [3, 32, 48],
        "stackwise_stage_filters": [
            [48, 48, 128, 1, 6, 3],
            [128, 96, 512, 1, 6, 3],
            [512, 192, 1024, 3, 6, 5],
            [1024, 384, 2048, 1, 6, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 48,
        "hidden_sizes": [128, 512, 1024, 2048],
        "depths": [1, 1, 3, 1],
    },
    "hgnetv2_b5": {
        "stem_channels": [3, 32, 64],
        "stackwise_stage_filters": [
            [64, 64, 128, 1, 6, 3],
            [128, 128, 512, 2, 6, 3],
            [512, 256, 1024, 5, 6, 5],
            [1024, 512, 2048, 2, 6, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 64,
        "hidden_sizes": [128, 512, 1024, 2048],
        "depths": [1, 2, 5, 2],
    },
    "hgnetv2_b6": {
        "stem_channels": [3, 48, 96],
        "stackwise_stage_filters": [
            [96, 96, 192, 2, 6, 3],
            [192, 192, 512, 3, 6, 3],
            [512, 384, 1024, 6, 6, 5],
            [1024, 768, 2048, 3, 6, 5],
        ],
        "apply_downsample": [False, True, True, True],
        "use_lightweight_conv_block": [False, False, True, True],
        "embedding_size": 96,
        "hidden_sizes": [192, 512, 1024, 2048],
        "depths": [2, 3, 6, 3],
    },
}


def _use_learnable_affine_block(timm_config):
    # SSLD-distilled checkpoints drop the learnable affine block that the
    # default (e.g. `ra_in1k`) checkpoints train with.
    tag = (timm_config.get("pretrained_cfg") or {}).get("tag", "") or ""
    return not tag.startswith("ssld")


def convert_backbone_config(timm_config):
    architecture = timm_config["architecture"]
    config = HGNETV2_CONFIGS[architecture]
    return {
        "depths": config["depths"],
        "embedding_size": config["embedding_size"],
        "hidden_sizes": config["hidden_sizes"],
        "stem_channels": config["stem_channels"],
        "hidden_act": "relu",
        "use_learnable_affine_block": _use_learnable_affine_block(timm_config),
        "stackwise_stage_filters": config["stackwise_stage_filters"],
        "apply_downsample": config["apply_downsample"],
        "use_lightweight_conv_block": config["use_lightweight_conv_block"],
    }


def convert_weights(backbone, loader, timm_config):
    def port_conv(keras_layer, hf_prefix):
        loader.port_weight(
            keras_layer.convolution.kernel,
            hf_weight_key=f"{hf_prefix}.conv.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        loader.port_weight(
            keras_layer.normalization.gamma,
            hf_weight_key=f"{hf_prefix}.bn.weight",
        )
        loader.port_weight(
            keras_layer.normalization.beta,
            hf_weight_key=f"{hf_prefix}.bn.bias",
        )
        loader.port_weight(
            keras_layer.normalization.moving_mean,
            hf_weight_key=f"{hf_prefix}.bn.running_mean",
        )
        loader.port_weight(
            keras_layer.normalization.moving_variance,
            hf_weight_key=f"{hf_prefix}.bn.running_var",
        )
        if not isinstance(keras_layer.lab, keras.layers.Identity):
            loader.port_weight(
                keras_layer.lab.scale, hf_weight_key=f"{hf_prefix}.lab.scale"
            )
            loader.port_weight(
                keras_layer.lab.bias, hf_weight_key=f"{hf_prefix}.lab.bias"
            )

    def port_conv_light(keras_layer, hf_prefix):
        port_conv(keras_layer.conv1_layer, f"{hf_prefix}.conv1")
        port_conv(keras_layer.conv2_layer, f"{hf_prefix}.conv2")

    def port_basic_layer(keras_layer, hf_prefix):
        for i, layer in enumerate(keras_layer.layer_list):
            layer_prefix = f"{hf_prefix}.layers.{i}"
            if isinstance(layer, HGNetV2ConvLayerLight):
                port_conv_light(layer, layer_prefix)
            else:
                port_conv(layer, layer_prefix)
        port_conv(
            keras_layer.aggregation_squeeze_conv,
            f"{hf_prefix}.aggregation.0",
        )
        port_conv(
            keras_layer.aggregation_excitation_conv,
            f"{hf_prefix}.aggregation.1",
        )

    def port_stage(keras_stage, hf_prefix):
        if not isinstance(keras_stage.downsample_layer, keras.layers.Identity):
            port_conv(keras_stage.downsample_layer, f"{hf_prefix}.downsample")
        for block_idx, block in enumerate(keras_stage.blocks_list):
            port_basic_layer(block, f"{hf_prefix}.blocks.{block_idx}")

    embedder = backbone.embedder_layer
    port_conv(embedder.stem1_layer, "stem.stem1")
    port_conv(embedder.stem2a_layer, "stem.stem2a")
    port_conv(embedder.stem2b_layer, "stem.stem2b")
    port_conv(embedder.stem3_layer, "stem.stem3")
    port_conv(embedder.stem4_layer, "stem.stem4")

    for i, stage in enumerate(backbone.encoder_layer.stages_list):
        port_stage(stage, f"stages.{i}")


def _has_weight_key(loader, hf_weight_key):
    try:
        loader.get_tensor(hf_weight_key)
        return True
    except Exception:
        return False


def convert_head(task, loader, timm_config):
    # `head.last_conv` is a plain `nn.Sequential(conv, [bn], [lab])`, so HF
    # weight keys are positionally indexed rather than named `.conv`/`.bn`.
    prefix = "head.last_conv"
    conv_layer = task.last_conv

    loader.port_weight(
        conv_layer.convolution.kernel,
        hf_weight_key=f"{prefix}.0.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )

    if _has_weight_key(loader, f"{prefix}.1.weight"):
        loader.port_weight(
            conv_layer.normalization.gamma, hf_weight_key=f"{prefix}.1.weight"
        )
        loader.port_weight(
            conv_layer.normalization.beta, hf_weight_key=f"{prefix}.1.bias"
        )
        loader.port_weight(
            conv_layer.normalization.moving_mean,
            hf_weight_key=f"{prefix}.1.running_mean",
        )
        loader.port_weight(
            conv_layer.normalization.moving_variance,
            hf_weight_key=f"{prefix}.1.running_var",
        )
        # when BN is present, LAB (if any) is only ever found at index 2.
        lab_prefixes = [f"{prefix}.2"]
    else:
        # BN is folded into the conv on some checkpoints; leave the
        # normalization layer as an identity transform.
        gamma = conv_layer.normalization.gamma
        identity_value = np.sqrt(1.0 + conv_layer.normalization.epsilon)
        gamma.assign(np.full_like(gamma.numpy(), identity_value))
        conv_layer.normalization.beta.assign(
            np.zeros_like(conv_layer.normalization.beta.numpy())
        )
        conv_layer.normalization.moving_mean.assign(
            np.zeros_like(conv_layer.normalization.moving_mean.numpy())
        )
        conv_layer.normalization.moving_variance.assign(
            np.ones_like(conv_layer.normalization.moving_variance.numpy())
        )
        # Matches the reference conversion script: without BN, LAB may sit
        # at index 1 (immediately after the conv) or, failing that, index 2.
        lab_prefixes = [f"{prefix}.1", f"{prefix}.2"]

    if not isinstance(conv_layer.lab, keras.layers.Identity):
        for lab_prefix in lab_prefixes:
            if _has_weight_key(loader, f"{lab_prefix}.scale"):
                loader.port_weight(
                    conv_layer.lab.scale, hf_weight_key=f"{lab_prefix}.scale"
                )
                loader.port_weight(
                    conv_layer.lab.bias, hf_weight_key=f"{lab_prefix}.bias"
                )
                break

    loader.port_weight(
        task.output_dense.kernel,
        hf_weight_key="head.fc.weight",
        hook_fn=lambda x, _: np.transpose(x, (1, 0)),
    )
    loader.port_weight(
        task.output_dense.bias,
        hf_weight_key="head.fc.bias",
    )
