import argparse
import os

from keras import ops
from segment_anything import sam_model_registry

from keras_hub.src.models.sam.sam_backbone import SAMBackbone
from keras_hub.src.models.sam.sam_image_converter import SAMImageConverter
from keras_hub.src.models.sam.sam_image_segmenter import SAMImageSegmenter
from keras_hub.src.models.sam.sam_image_segmenter_preprocessor import (
    SAMImageSegmenterPreprocessor,
)
from keras_hub.src.models.sam.sam_mask_decoder import SAMMaskDecoder
from keras_hub.src.models.sam.sam_prompt_encoder import SAMPromptEncoder
from keras_hub.src.models.vit_det.vit_det_backbone import ViTDetBackbone

os.environ["KERAS_BACKEND"] = "jax"

# No GPU for conversion, makes memory management easier.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def build_sam_base_model():
    image_size = 1024
    image_encoder = ViTDetBackbone(
        hidden_size=768,
        num_layers=12,
        intermediate_dim=768 * 4,
        num_heads=12,
        global_attention_layer_indices=[2, 5, 8, 11],
        patch_size=16,
        num_output_channels=256,
        window_size=14,
        image_shape=(image_size, image_size, 3),
    )
    prompt_encoder = SAMPromptEncoder(
        hidden_size=256,
        image_embedding_size=(64, 64),
        input_image_size=(
            image_size,
            image_size,
        ),
        mask_in_channels=16,
    )
    mask_decoder = SAMMaskDecoder(
        num_layers=2,
        hidden_size=256,
        intermediate_dim=2048,
        num_heads=8,
        embedding_dim=256,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    sam_backbone = SAMBackbone(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
    )
    sam_image_converter = SAMImageConverter(
        height=1024, width=1024, scale=1.0 / 255
    )
    sam_preprocessor = SAMImageSegmenterPreprocessor(
        image_converter=sam_image_converter
    )
    sam_image_segmenter = SAMImageSegmenter(
        backbone=sam_backbone, preprocessor=sam_preprocessor
    )
    return sam_image_segmenter


def build_sam_large_model():
    image_size = 1024
    image_encoder = ViTDetBackbone(
        hidden_size=1024,
        num_layers=24,
        intermediate_dim=1024 * 4,
        num_heads=16,
        global_attention_layer_indices=[5, 11, 17, 23],
        patch_size=16,
        num_output_channels=256,
        window_size=14,
        image_shape=(image_size, image_size, 3),
    )
    prompt_encoder = SAMPromptEncoder(
        hidden_size=256,
        image_embedding_size=(64, 64),
        input_image_size=(
            image_size,
            image_size,
        ),
        mask_in_channels=16,
    )
    mask_decoder = SAMMaskDecoder(
        num_layers=2,
        hidden_size=256,
        intermediate_dim=2048,
        num_heads=8,
        embedding_dim=256,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    sam_backbone = SAMBackbone(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
    )
    sam_image_converter = SAMImageConverter(
        height=1024, width=1024, scale=1.0 / 255
    )
    sam_preprocessor = SAMImageSegmenterPreprocessor(
        image_converter=sam_image_converter
    )
    sam_image_segmenter = SAMImageSegmenter(
        backbone=sam_backbone, preprocessor=sam_preprocessor
    )
    return sam_image_segmenter


def build_sam_huge_model():
    image_size = 1024
    image_encoder = ViTDetBackbone(
        hidden_size=1280,
        num_layers=32,
        intermediate_dim=1280 * 4,
        num_heads=16,
        global_attention_layer_indices=[7, 15, 23, 31],
        patch_size=16,
        num_output_channels=256,
        window_size=14,
        image_shape=(image_size, image_size, 3),
    )
    prompt_encoder = SAMPromptEncoder(
        hidden_size=256,
        image_embedding_size=(64, 64),
        input_image_size=(
            image_size,
            image_size,
        ),
        mask_in_channels=16,
    )
    mask_decoder = SAMMaskDecoder(
        num_layers=2,
        hidden_size=256,
        intermediate_dim=2048,
        num_heads=8,
        embedding_dim=256,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    sam_backbone = SAMBackbone(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
    )
    sam_image_converter = SAMImageConverter(
        height=1024, width=1024, scale=1.0 / 255
    )
    sam_preprocessor = SAMImageSegmenterPreprocessor(
        image_converter=sam_image_converter
    )
    sam_image_segmenter = SAMImageSegmenter(
        backbone=sam_backbone, preprocessor=sam_preprocessor
    )
    return sam_image_segmenter


def convert_mask_decoder(keras_mask_decoder, torch_mask_decoder):
    for i in range(2):
        keras_mask_decoder.transformer.layers[i].self_attention.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_mask_decoder.transformer.layers[
                    i
                ].self_attn.parameters()
            ]
        )

        keras_mask_decoder.transformer.layers[i].layer_norm1.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_mask_decoder.transformer.layers[
                    i
                ].norm1.parameters()
            ]
        )
        keras_mask_decoder.transformer.layers[
            i
        ].cross_attention_token_to_image.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_mask_decoder.transformer.layers[
                    i
                ].cross_attn_token_to_image.parameters()
            ]
        )
        keras_mask_decoder.transformer.layers[i].layer_norm2.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_mask_decoder.transformer.layers[
                    i
                ].norm2.parameters()
            ]
        )
        keras_mask_decoder.transformer.layers[i].mlp_block.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_mask_decoder.transformer.layers[
                    i
                ].mlp.parameters()
            ]
        )
        keras_mask_decoder.transformer.layers[i].layer_norm3.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_mask_decoder.transformer.layers[
                    i
                ].norm3.parameters()
            ]
        )
        keras_mask_decoder.transformer.layers[
            i
        ].cross_attention_image_to_token.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_mask_decoder.transformer.layers[
                    i
                ].cross_attn_image_to_token.parameters()
            ]
        )
        keras_mask_decoder.transformer.layers[i].layer_norm4.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_mask_decoder.transformer.layers[
                    i
                ].norm4.parameters()
            ]
        )
    keras_mask_decoder.transformer.final_attention_token_to_image.set_weights(
        [
            x.cpu().detach().numpy().T
            for x in torch_mask_decoder.transformer.final_attn_token_to_image.parameters()
        ]
    )
    keras_mask_decoder.transformer.final_layer_norm.set_weights(
        [
            x.cpu().detach().numpy()
            for x in torch_mask_decoder.transformer.norm_final_attn.parameters()
        ]
    )
    keras_mask_decoder.iou_token.set_weights(
        [
            x.cpu().detach().numpy()
            for x in torch_mask_decoder.iou_token.parameters()
        ]
    )
    keras_mask_decoder.mask_tokens.set_weights(
        [
            x.cpu().detach().numpy()
            for x in torch_mask_decoder.mask_tokens.parameters()
        ]
    )
    keras_mask_decoder.output_upscaling.set_weights(
        [
            (
                x.permute(2, 3, 1, 0).cpu().detach().numpy()
                if x.ndim == 4
                else x.cpu().detach().numpy()
            )
            for x in torch_mask_decoder.output_upscaling.parameters()
        ]
    )
    for i in range(keras_mask_decoder.num_mask_tokens):
        keras_mask_decoder.output_hypernetworks_mlps[i].set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_mask_decoder.output_hypernetworks_mlps[
                    i
                ].parameters()
            ]
        )
    keras_mask_decoder.iou_prediction_head.set_weights(
        [
            x.cpu().detach().numpy().T
            for x in torch_mask_decoder.iou_prediction_head.parameters()
        ]
    )
    return keras_mask_decoder


def convert_prompt_encoder(keras_prompt_encoder, torch_prompt_encoder):
    keras_prompt_encoder.background_point_embed.set_weights(
        [torch_prompt_encoder.point_embeddings[0].weight.cpu().detach().numpy()]
    )
    keras_prompt_encoder.foreground_point_embed.set_weights(
        [torch_prompt_encoder.point_embeddings[1].weight.cpu().detach().numpy()]
    )
    keras_prompt_encoder.top_left_corner_embed.set_weights(
        [torch_prompt_encoder.point_embeddings[2].weight.cpu().detach().numpy()]
    )
    keras_prompt_encoder.bottom_right_corner_embed.set_weights(
        [torch_prompt_encoder.point_embeddings[3].weight.cpu().detach().numpy()]
    )
    keras_prompt_encoder.not_a_point_embed.set_weights(
        [torch_prompt_encoder.not_a_point_embed.weight.cpu().detach().numpy()]
    )
    keras_prompt_encoder.mask_downscaler.set_weights(
        [
            (
                x.permute(2, 3, 1, 0).cpu().detach().numpy()
                if x.ndim == 4
                else x.cpu().detach().numpy()
            )
            for x in torch_prompt_encoder.mask_downscaling.parameters()
        ]
    )
    keras_prompt_encoder.no_mask_embed.set_weights(
        [torch_prompt_encoder.no_mask_embed.weight.cpu().detach().numpy()]
    )
    keras_prompt_encoder.positional_embedding_layer.positional_encoding_gaussian_matrix.assign(
        torch_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.cpu()
        .detach()
        .numpy()
    )
    return keras_prompt_encoder


def convert_image_encoder(keras_image_encoder, torch_image_encoder):
    keras_image_encoder.get_layer(
        "vi_t_det_patching_and_embedding"
    ).projection.set_weights(
        [
            torch_image_encoder.patch_embed.proj.weight.permute(2, 3, 1, 0)
            .cpu()
            .detach()
            .numpy(),
            torch_image_encoder.patch_embed.proj.bias.cpu().detach().numpy(),
        ]
    )
    keras_image_encoder.get_layer("add_positional_embedding").set_weights(
        ops.expand_dims(
            torch_image_encoder.pos_embed.cpu().detach().numpy(), axis=0
        )
    )
    for i, block_torch in enumerate(torch_image_encoder.blocks):

        block_name = "windowed_transformer_encoder"
        if i > 0:
            block_name = "windowed_transformer_encoder_" + str(i)
        keras_image_encoder.get_layer(block_name).layer_norm1.set_weights(
            [x.cpu().detach().numpy() for x in block_torch.norm1.parameters()]
        )
        keras_image_encoder.get_layer(block_name).layer_norm2.set_weights(
            [x.cpu().detach().numpy() for x in block_torch.norm2.parameters()]
        )
        keras_image_encoder.get_layer(
            block_name
        ).attention.add_decomposed_relative_pe.rel_pos_h.assign(
            block_torch.attn.rel_pos_h.cpu().detach().numpy()
        )
        keras_image_encoder.get_layer(
            block_name
        ).attention.add_decomposed_relative_pe.rel_pos_w.assign(
            block_torch.attn.rel_pos_w.cpu().detach().numpy()
        )
        keras_image_encoder.get_layer(block_name).attention.qkv.weights[
            0
        ].assign(block_torch.attn.qkv.weight.cpu().detach().numpy().T)
        keras_image_encoder.get_layer(block_name).attention.qkv.weights[
            1
        ].assign(block_torch.attn.qkv.bias.cpu().detach().numpy())
        keras_image_encoder.get_layer(block_name).attention.projection.weights[
            0
        ].assign(block_torch.attn.proj.weight.cpu().detach().numpy().T)
        keras_image_encoder.get_layer(block_name).attention.projection.weights[
            1
        ].assign(block_torch.attn.proj.bias.cpu().detach().numpy())
        keras_image_encoder.get_layer(block_name).mlp_block.set_weights(
            [
                block_torch.mlp.lin1.weight.cpu().detach().numpy().T,
                block_torch.mlp.lin1.bias.cpu().detach().numpy(),
                block_torch.mlp.lin2.weight.cpu().detach().numpy().T,
                block_torch.mlp.lin2.bias.cpu().detach().numpy(),
            ]
        )
    keras_image_encoder.neck.set_weights(
        [
            (
                x.permute(2, 3, 1, 0).cpu().detach().numpy()
                if x.ndim == 4
                else x.cpu().detach().numpy()
            )
            for x in torch_image_encoder.neck.parameters()
        ]
    )
    return keras_image_encoder


def port_weights(keras_model, torch_model):
    keras_model.backbone.prompt_encoder = convert_prompt_encoder(
        keras_model.backbone.prompt_encoder, torch_model.prompt_encoder
    )
    keras_model.backbone.mask_decoder = convert_mask_decoder(
        keras_model.backbone.mask_decoder, torch_model.mask_decoder
    )
    keras_model.backbone.image_encoder = convert_image_encoder(
        keras_model.backbone.image_encoder, torch_model.image_encoder
    )
    return keras_model


def main(args):
    # URL to download checkpoints form
    # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    if args.model_variant == "base":
        print("Loading Keras model")
        keras_model = build_sam_base_model()
        print("Loading Torch model")
        torch_model = sam_model_registry["vit_b"](checkpoint=args.weights_path)
        keras_model = port_weights(keras_model, torch_model)
    elif args.model_variant == "large":
        print("Loading Keras model")
        keras_model = build_sam_large_model()
        print("Loading Torch model")
        torch_model = sam_model_registry["vit_l"](checkpoint=args.weights_path)
        keras_model = port_weights(keras_model, torch_model)
    elif args.model_variant == "huge":
        print("Loading Keras model")
        keras_model = build_sam_huge_model()
        print("Loading Torch model")
        torch_model = sam_model_registry["vit_h"](checkpoint=args.weights_path)
        keras_model = port_weights(keras_model, torch_model)
    print("Saving Keras model to folder ", args.checkpoint_name)
    keras_model.save_to_preset(args.checkpoint_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert segment anything weights to Keras."
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to the .pth weights file.",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="base",
        help="SAM model size: `base`, `large`, `huge`",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="sam_base",
        help="Name for Keras checkpoint, defaults to `sam_base`",
    )
    args = parser.parse_args()
    main(args)
