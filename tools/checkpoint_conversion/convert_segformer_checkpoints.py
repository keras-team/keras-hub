# Usage example
# python tools/checkpoint_conversion/convert_mix_transformer.py --preset "B0_ade_512"

import numpy as np
from absl import app
from absl import flags
from transformers import SegformerForSemanticSegmentation

import keras_hub
from keras_hub.src.models.segformer.segformer_image_segmenter_preprocessor import (
    SegFormerImageSegmenterPreprocessor,
)

FLAGS = flags.FLAGS

PROJECTION_FILTERS = {
    "b0_ade20k_512": 256,
    "b1_ade20k_512": 256,
    "b2_ade20k_512": 768,
    "b3_ade20k_512": 768,
    "b4_ade20k_512": 768,
    "b5_ade20k_640": 768,
    "b0_cityscapes_1024": 256,
    "b1_cityscapes_1024": 256,
    "b2_cityscapes_1024": 768,
    "b3_cityscapes_1024": 768,
    "b4_cityscapes_1024": 768,
    "b5_cityscapes_1024": 768,
}


DOWNLOAD_URLS = {
    "b0_ade20k_512": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b1_ade20k_512": "nvidia/segformer-b1-finetuned-ade-512-512",
    "b2_ade20k_512": "nvidia/segformer-b2-finetuned-ade-512-512",
    "b3_ade20k_512": "nvidia/segformer-b3-finetuned-ade-512-512",
    "b4_ade20k_512": "nvidia/segformer-b4-finetuned-ade-512-512",
    "b5_ade20k_640": "nvidia/segformer-b5-finetuned-ade-640-640",
    "b0_cityscapes_1024": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "b1_cityscapes_1024": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    "b2_cityscapes_1024": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b3_cityscapes_1024": "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    "b4_cityscapes_1024": "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
    "b5_cityscapes_1024": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}

flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(DOWNLOAD_URLS.keys())}'
)


def set_conv_weights(conv_layer, state_dict):
    conv_weights = state_dict["weight"].numpy().transpose(2, 3, 1, 0)
    bias = None
    if "bias" in state_dict.keys():
        bias = state_dict["bias"].numpy()
        conv_layer.set_weights([conv_weights, bias])
    else:
        conv_layer.set_weights([conv_weights])


def set_dense_weights(dense_layer, state_dict):
    weight = state_dict["weight"].numpy().T
    bias = state_dict["bias"].numpy()
    dense_layer.set_weights([weight, bias])


def set_batchnorm_weights(bn_layer, state_dict):
    gamma = state_dict["weight"].numpy()
    beta = state_dict["bias"].numpy()
    running_mean = state_dict["running_mean"].numpy()
    running_var = state_dict["running_var"].numpy()

    bn_layer.set_weights([gamma, beta, running_mean, running_var])


def main(_):
    print("\n-> Loading HuggingFace model")
    original_segformer = SegformerForSemanticSegmentation.from_pretrained(
        DOWNLOAD_URLS[FLAGS.preset]
    )

    print("\n-> Instantiating KerasHub Model")

    resolution = int(FLAGS.preset.split("_")[-1])

    encoder = keras_hub.models.MiTBackbone.from_preset(
        "mit_" + FLAGS.preset, image_shape=(resolution, resolution, 3)
    )
    segformer_backbone = keras_hub.models.SegFormerBackbone(
        image_encoder=encoder,
        projection_filters=PROJECTION_FILTERS[FLAGS.preset],
    )
    num_classes = 150 if "ade20k" in FLAGS.preset else 19

    preprocessor = SegFormerImageSegmenterPreprocessor()
    segformer_segmenter = keras_hub.models.SegFormerImageSegmenter(
        backbone=segformer_backbone,
        num_classes=num_classes,
        preprocessor=preprocessor,
    )
    segformer_backbone(np.random.rand(1, resolution, resolution, 3))

    set_dense_weights(
        segformer_backbone.layers[5],
        original_segformer.decode_head.linear_c[0].proj.state_dict(),
    )
    set_dense_weights(
        segformer_backbone.layers[4],
        original_segformer.decode_head.linear_c[1].proj.state_dict(),
    )
    set_dense_weights(
        segformer_backbone.layers[3],
        original_segformer.decode_head.linear_c[2].proj.state_dict(),
    )
    set_dense_weights(
        segformer_backbone.layers[2],
        original_segformer.decode_head.linear_c[3].proj.state_dict(),
    )
    set_conv_weights(
        segformer_backbone.layers[-1].layers[0],
        original_segformer.decode_head.linear_fuse.state_dict(),
    )
    set_batchnorm_weights(
        segformer_backbone.layers[-1].layers[1],
        original_segformer.decode_head.batch_norm.state_dict(),
    )

    set_conv_weights(
        segformer_segmenter.layers[-2],
        original_segformer.decode_head.classifier.state_dict(),
    )

    print("\n-> Converting weights...")

    directory = f"SegFormer_{FLAGS.preset}"
    print(f"\n-> Saving converted KerasHub model in {directory}")
    segformer_segmenter.save_to_preset(directory)


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
