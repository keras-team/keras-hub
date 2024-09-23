# Copyright 2024 The KerasNLP Authors
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

import os

import numpy as np
from segment_anything import sam_model_registry

from keras_hub.src.models.sam.sam_backbone import SAMBackbone
from keras_hub.src.models.sam.sam_mask_decoder import SAMMaskDecoder
from keras_hub.src.models.sam.sam_prompt_encoder import SAMPromptEncoder
from keras_hub.src.models.vit_det.vit_det_backbone import ViTDetBackbone

os.environ["KERAS_BACKEND"] = "jax"
from keras import ops  # noqa: E402

# No GPU for conversion, makes memory management easier.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def print_keys(d, parent_key=""):
    for k, v in d.items():
        if isinstance(v, dict):
            if parent_key:
                print_keys(v, f"{parent_key}.{k}")
            else:
                print_keys(v, k)
        else:
            if parent_key:
                print(f"{parent_key}.{k}")
            else:
                print(k)


# Author: Tirth Patel (tirthasheshpatel@gmail.com)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


def port_weights(mb_model, torch_model):
    """Port weights of the PyTorch model to the Keras Core model.
    Both models must be defined the same way."""

    mb_model.prompt_encoder.background_point_embed.set_weights(
        [
            torch_model.prompt_encoder.point_embeddings[0]
            .weight.cpu()
            .detach()
            .numpy()
        ]
    )
    mb_model.prompt_encoder.foreground_point_embed.set_weights(
        [
            torch_model.prompt_encoder.point_embeddings[1]
            .weight.cpu()
            .detach()
            .numpy()
        ]
    )
    mb_model.prompt_encoder.top_left_corner_embed.set_weights(
        [
            torch_model.prompt_encoder.point_embeddings[2]
            .weight.cpu()
            .detach()
            .numpy()
        ]
    )
    mb_model.prompt_encoder.bottom_right_corner_embed.set_weights(
        [
            torch_model.prompt_encoder.point_embeddings[3]
            .weight.cpu()
            .detach()
            .numpy()
        ]
    )
    mb_model.prompt_encoder.not_a_point_embed.set_weights(
        [
            torch_model.prompt_encoder.not_a_point_embed.weight.cpu()
            .detach()
            .numpy()
        ]
    )
    mb_model.prompt_encoder.mask_downscaler.set_weights(
        [
            (
                x.permute(2, 3, 1, 0).cpu().detach().numpy()
                if x.ndim == 4
                else x.cpu().detach().numpy()
            )
            for x in torch_model.prompt_encoder.mask_downscaling.parameters()
        ]
    )
    mb_model.prompt_encoder.no_mask_embed.set_weights(
        [torch_model.prompt_encoder.no_mask_embed.weight.cpu().detach().numpy()]
    )
    mb_model.prompt_encoder.positional_embedding_layer.positional_encoding_gaussian_matrix.assign(
        torch_model.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.cpu()
        .detach()
        .numpy()
    )
    total_params = 0
    for param in torch_model.prompt_encoder.parameters():
        total_params += param.numel()
    print("torch sam prompt encoder paarmeter", total_params)
    print(
        "keras prompt encoder parameter", mb_model.prompt_encoder.count_params()
    )
    for i in range(2):
        mb_model.mask_decoder.transformer.layers[i].self_attention.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].self_attn.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[i].layer_norm1.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].norm1.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[
            i
        ].cross_attention_token_to_image.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].cross_attn_token_to_image.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[i].layer_norm2.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].norm2.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[i].mlp_block.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].mlp.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[i].layer_norm3.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].norm3.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[
            i
        ].cross_attention_image_to_token.set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].cross_attn_image_to_token.parameters()
            ]
        )
        mb_model.mask_decoder.transformer.layers[i].layer_norm4.set_weights(
            [
                x.cpu().detach().numpy()
                for x in torch_model.mask_decoder.transformer.layers[
                    i
                ].norm4.parameters()
            ]
        )
    mb_model.mask_decoder.transformer.final_attention_token_to_image.set_weights(
        [
            x.cpu().detach().numpy().T
            for x in torch_model.mask_decoder.transformer.final_attn_token_to_image.parameters()
        ]
    )
    mb_model.mask_decoder.transformer.final_layer_norm.set_weights(
        [
            x.cpu().detach().numpy()
            for x in torch_model.mask_decoder.transformer.norm_final_attn.parameters()
        ]
    )
    mb_model.mask_decoder.iou_token.set_weights(
        [
            x.cpu().detach().numpy()
            for x in torch_model.mask_decoder.iou_token.parameters()
        ]
    )
    mb_model.mask_decoder.mask_tokens.set_weights(
        [
            x.cpu().detach().numpy()
            for x in torch_model.mask_decoder.mask_tokens.parameters()
        ]
    )
    mb_model.mask_decoder.output_upscaling.set_weights(
        [
            (
                x.permute(2, 3, 1, 0).cpu().detach().numpy()
                if x.ndim == 4
                else x.cpu().detach().numpy()
            )
            for x in torch_model.mask_decoder.output_upscaling.parameters()
        ]
    )
    for i in range(mb_model.mask_decoder.num_mask_tokens):
        mb_model.mask_decoder.output_hypernetworks_mlps[i].set_weights(
            [
                x.cpu().detach().numpy().T
                for x in torch_model.mask_decoder.output_hypernetworks_mlps[
                    i
                ].parameters()
            ]
        )
    mb_model.mask_decoder.iou_prediction_head.set_weights(
        [
            x.cpu().detach().numpy().T
            for x in torch_model.mask_decoder.iou_prediction_head.parameters()
        ]
    )
    total_params = 0
    for param in torch_model.mask_decoder.parameters():
        total_params += param.numel()
    print("torch sam mask decoder paarmeter", total_params)
    print("keras mask decoder parameter", mb_model.mask_decoder.count_params())
    mb_model.image_encoder.get_layer(
        "vi_t_det_patching_and_embedding"
    ).set_weights(
        [
            (
                x.permute(2, 3, 1, 0).cpu().detach().numpy()
                if x.ndim == 4
                else x.cpu().detach().numpy()
            )
            for x in torch_model.image_encoder.patch_embed.parameters()
        ]
    )
    mb_model.image_encoder.get_layer("add_positional_embedding").set_weights(
        ops.expand_dims(
            torch_model.image_encoder.pos_embed.cpu().detach().numpy(), axis=0
        )
    )
    for i, block_torch in enumerate(torch_model.image_encoder.blocks):
        if i == 0:
            block_mb = mb_model.image_encoder.get_layer(
                "windowed_transformer_encoder"
            )
        else:
            block_mb = mb_model.image_encoder.get_layer(
                "windowed_transformer_encoder_" + str(i)
            )
        block_mb.layer_norm1.set_weights(
            [x.cpu().detach().numpy() for x in block_torch.norm1.parameters()]
        )
        block_mb.layer_norm2.set_weights(
            [x.cpu().detach().numpy() for x in block_torch.norm2.parameters()]
        )
        block_torch_attn_weights = [
            x.cpu().detach().numpy() for x in block_torch.attn.parameters()
        ]
        block_mb.attention.set_weights(
            [
                block_torch_attn_weights[2].T,
                block_torch_attn_weights[3],
                block_torch_attn_weights[4],
                block_torch_attn_weights[5],
                block_torch_attn_weights[0],
                block_torch_attn_weights[1],
            ]
        )
        block_mb.mlp_block.set_weights(
            [x.cpu().detach().numpy().T for x in block_torch.mlp.parameters()]
        )
    mb_model.image_encoder.neck.set_weights(
        [
            (
                x.permute(2, 3, 1, 0).cpu().detach().numpy()
                if x.ndim == 4
                else x.cpu().detach().numpy()
            )
            for x in torch_model.image_encoder.neck.parameters()
        ]
    )
    total_params = 0
    for param in torch_model.image_encoder.parameters():
        total_params += param.numel()
    print("torch sam image_encoder paarmeter", total_params)
    print(
        "keras image encoder parameter", mb_model.image_encoder.count_params()
    )
    return mb_model


def main():
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
    batch_size = 2
    input_data = {
        "images": np.ones(
            (batch_size, image_size, image_size, 3),
            dtype="float32",
        ),
        "points": np.ones((batch_size, 1, 2), dtype="float32"),
        "labels": np.ones((batch_size, 1), dtype="float32"),
        "boxes": np.ones((batch_size, 1, 2, 2), dtype="float32"),
        "masks": np.zeros((batch_size, 0, image_size, image_size, 1)),
    }

    sam_checkpoint = "tools/checkpoint_conversion/sam_vit_b.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_backbone = port_weights(sam_backbone, sam)
    sam_backbone(input_data)
    total_params = 0
    for param in sam.parameters():
        total_params += param.numel()
    print("total sam paarmeter", total_params)
    print("total keras parameter", sam_backbone.count_params())
    total_params = 0


if __name__ == "__main__":
    main()
