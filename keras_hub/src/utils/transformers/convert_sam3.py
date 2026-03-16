import warnings

import numpy as np
from keras import layers

from keras_hub.src.models.sam3.sam3_detr_decoder import SAM3DetrDecoder
from keras_hub.src.models.sam3.sam3_detr_encoder import SAM3DetrEncoder
from keras_hub.src.models.sam3.sam3_geometry_encoder import SAM3GeometryEncoder
from keras_hub.src.models.sam3.sam3_mask_decoder import SAM3MaskDecoder
from keras_hub.src.models.sam3.sam3_pc_backbone import (
    SAM3PromptableConceptBackbone,
)
from keras_hub.src.models.sam3.sam3_text_encoder import SAM3TextEncoder
from keras_hub.src.models.sam3.sam3_vision_encoder import SAM3VisionEncoder
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = SAM3PromptableConceptBackbone


def convert_backbone_config(transformers_config, cls, **kwargs):
    # detector_config: Promptable Concept Segmentation (PCS)
    # tracker_config: Promptable Visual Segmentation (PVS)
    if issubclass(cls, SAM3PromptableConceptBackbone):
        # Extract sub-configurations.
        transformers_config = transformers_config["detector_config"]

        vision_config = transformers_config["vision_config"]
        backbone_config = vision_config["backbone_config"]
        text_config = transformers_config["text_config"]
        geom_config = transformers_config["geometry_encoder_config"]
        detr_enc_config = transformers_config["detr_encoder_config"]
        detr_dec_config = transformers_config["detr_decoder_config"]
        mask_dec_config = transformers_config["mask_decoder_config"]
        dtype = kwargs.pop("dtype", None)
        image_shape = kwargs.pop("image_shape", None)
        if image_shape is None:
            image_shape = (
                backbone_config["image_size"],
                backbone_config["image_size"],
                3,
            )

        # Vision Encoder.
        vision_encoder_config = {
            "image_shape": image_shape,
            "patch_size": backbone_config["patch_size"],
            "num_layers": backbone_config["num_hidden_layers"],
            "hidden_dim": backbone_config["hidden_size"],
            "intermediate_dim": backbone_config["intermediate_size"],
            "num_heads": backbone_config["num_attention_heads"],
            "fpn_hidden_dim": vision_config["fpn_hidden_size"],
            "fpn_scale_factors": vision_config["scale_factors"],
            "pretrain_image_shape": (
                backbone_config["pretrain_image_size"],
                backbone_config["pretrain_image_size"],
                3,
            ),
            "hidden_activation": backbone_config["hidden_act"],
            "rope_theta": backbone_config["rope_theta"],
            "window_size": backbone_config["window_size"],
            "global_attn_indexes": backbone_config["global_attn_indexes"],
            "attention_dropout_rate": backbone_config["attention_dropout"],
            "hidden_dropout_rate": backbone_config["hidden_dropout"],
            "layer_norm_epsilon": backbone_config["layer_norm_eps"],
            "dtype": dtype,
        }
        vision_encoder = SAM3VisionEncoder(**vision_encoder_config)

        # Text Encoder.
        text_encoder_config = {
            "vocabulary_size": text_config["vocab_size"],
            "embedding_dim": text_config["hidden_size"],
            "hidden_dim": text_config["hidden_size"],
            "num_layers": text_config["num_hidden_layers"],
            "num_heads": text_config["num_attention_heads"],
            "intermediate_dim": text_config["intermediate_size"],
            "intermediate_activation": text_config["hidden_act"],
            "max_sequence_length": text_config["max_position_embeddings"],
            "layer_norm_epsilon": text_config["layer_norm_eps"],
            "dtype": dtype,
        }
        text_encoder = SAM3TextEncoder(**text_encoder_config)

        # Geometry Encoder.
        geometry_encoder_config = {
            "num_layers": geom_config["num_layers"],
            "hidden_dim": geom_config["hidden_size"],
            "intermediate_dim": geom_config["intermediate_size"],
            "num_heads": geom_config["num_attention_heads"],
            "roi_size": geom_config["roi_size"],
            "hidden_activation": geom_config["hidden_act"],
            "dropout_rate": geom_config["hidden_dropout"],
            "layer_norm_epsilon": geom_config["layer_norm_eps"],
            "dtype": dtype,
        }
        geometry_encoder = SAM3GeometryEncoder(**geometry_encoder_config)

        # DETR Encoder.
        detr_encoder_config = {
            "num_layers": detr_enc_config["num_layers"],
            "hidden_dim": detr_enc_config["hidden_size"],
            "intermediate_dim": detr_enc_config["intermediate_size"],
            "num_heads": detr_enc_config["num_attention_heads"],
            "hidden_activation": detr_enc_config["hidden_act"],
            "dropout_rate": detr_enc_config["dropout"],
            "layer_norm_epsilon": detr_enc_config["layer_norm_eps"],
            "dtype": dtype,
        }
        detr_encoder = SAM3DetrEncoder(**detr_encoder_config)

        # DETR Decoder.
        detr_decoder_config = {
            "image_shape": image_shape,
            "patch_size": backbone_config["patch_size"],
            "num_layers": detr_dec_config["num_layers"],
            "hidden_dim": detr_dec_config["hidden_size"],
            "intermediate_dim": detr_dec_config["intermediate_size"],
            "num_heads": detr_dec_config["num_attention_heads"],
            "num_queries": detr_dec_config["num_queries"],
            "hidden_activation": detr_dec_config["hidden_act"],
            "dropout_rate": detr_dec_config["dropout"],
            "layer_norm_epsilon": detr_dec_config["layer_norm_eps"],
            "dtype": dtype,
        }
        detr_decoder = SAM3DetrDecoder(**detr_decoder_config)

        # Mask Decoder.
        mask_decoder_config = {
            "num_upsampling_stages": mask_dec_config["num_upsampling_stages"],
            "hidden_dim": mask_dec_config["hidden_size"],
            "num_heads": mask_dec_config["num_attention_heads"],
            "dropout_rate": 0.0,
            "layer_norm_epsilon": mask_dec_config["layer_norm_eps"],
            "dtype": dtype,
        }
        mask_decoder = SAM3MaskDecoder(**mask_decoder_config)

        return {
            "vision_encoder": vision_encoder,
            "text_encoder": text_encoder,
            "geometry_encoder": geometry_encoder,
            "detr_encoder": detr_encoder,
            "detr_decoder": detr_decoder,
            "mask_decoder": mask_decoder,
        }
    else:
        # TODO: Add SAM3Tracker support.
        raise ValueError(
            "The provided class is not a subclass of "
            f"SAM3PromptableConceptBackbone. Received: {cls}"
        )


def convert_weights(backbone, loader, transformers_config):
    if not isinstance(backbone, SAM3PromptableConceptBackbone):
        raise ValueError(
            "The provided backbone must be an instance of "
            f"SAM3PromptableConceptBackbone. Received: {type(backbone)}"
        )

    def port_dense(keras_dense, hf_name):
        loader.port_weight(
            keras_dense.kernel, f"{hf_name}.weight", hook_fn=lambda x, _: x.T
        )
        if keras_dense.bias is not None:
            loader.port_weight(keras_dense.bias, f"{hf_name}.bias")

    def port_ln(keras_ln, hf_name):
        loader.port_weight(keras_ln.gamma, f"{hf_name}.weight")
        loader.port_weight(keras_ln.beta, f"{hf_name}.bias")

    def port_conv(keras_conv, hf_name):
        if not keras_conv.built:
            # https://github.com/huggingface/transformers/issues/43065
            warnings.warn(f"Skipping {hf_name}")
            return
        loader.port_weight(
            keras_conv.kernel,
            f"{hf_name}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        if keras_conv.bias is not None:
            loader.port_weight(keras_conv.bias, f"{hf_name}.bias")

    def port_gn(keras_gn, hf_name):
        if not keras_gn.built:
            # https://github.com/huggingface/transformers/issues/43065
            warnings.warn(f"Skipping {hf_name}")
            return
        loader.port_weight(keras_gn.gamma, f"{hf_name}.weight")
        loader.port_weight(keras_gn.beta, f"{hf_name}.bias")

    def port_attention(keras_attn, hf_name):
        port_dense(keras_attn.q_proj, f"{hf_name}.q_proj")
        port_dense(keras_attn.k_proj, f"{hf_name}.k_proj")
        port_dense(keras_attn.v_proj, f"{hf_name}.v_proj")
        port_dense(keras_attn.o_proj, f"{hf_name}.o_proj")

    def port_mlp(keras_mlp, hf_name):
        port_dense(keras_mlp.fc1, f"{hf_name}.fc1")
        port_dense(keras_mlp.fc2, f"{hf_name}.fc2")

    def port_decoder_mlp(keras_mlp, hf_name):
        port_dense(keras_mlp.layer1, f"{hf_name}.layer1")
        port_dense(keras_mlp.layer2, f"{hf_name}.layer2")
        if hasattr(keras_mlp, "layer3") and keras_mlp.layer3 is not None:
            port_dense(keras_mlp.layer3, f"{hf_name}.layer3")

    # Vision Encoder.
    vision_prefix = "vision_encoder"
    backbone_prefix = f"{vision_prefix}.backbone"
    emb = backbone.vision_encoder.backbone.embeddings
    port_conv(
        emb.patch_embeddings.projection,
        f"{backbone_prefix}.embeddings.patch_embeddings.projection",
    )
    loader.port_weight(
        emb.position_embeddings,
        f"{backbone_prefix}.embeddings.position_embeddings",
    )
    emb.tiled_position_embeddings.assign(
        emb._tile_position_embeddings(
            emb.position_embeddings,
            patch_size=emb.patch_size,
            source_shape=emb.pretrain_image_shape,
            target_shape=emb.image_shape,
        )
    )
    port_ln(
        backbone.vision_encoder.backbone.layer_norm,
        f"{backbone_prefix}.layer_norm",
    )
    for i, layer in enumerate(backbone.vision_encoder.backbone.layers):
        p = f"{backbone_prefix}.layers.{i}"
        port_ln(layer.layer_norm1, f"{p}.layer_norm1")
        port_attention(layer.attention, f"{p}.attention")
        port_ln(layer.layer_norm2, f"{p}.layer_norm2")
        port_mlp(layer.mlp, f"{p}.mlp")

    neck_prefix = f"{vision_prefix}.neck"
    for i, layer in enumerate(backbone.vision_encoder.vision_neck.fpn_layers):
        p = f"{neck_prefix}.fpn_layers.{i}"
        # FPN scale layers
        for j, scale_layer in enumerate(layer.scale_layers):
            if isinstance(scale_layer, (layers.Conv2DTranspose, layers.Conv2D)):
                port_conv(scale_layer, f"{p}.scale_layers.{j}")

        port_conv(layer.proj1, f"{p}.proj1")
        port_conv(layer.proj2, f"{p}.proj2")

    # Text Encoder.
    text_prefix = "text_encoder.text_model"
    loader.port_weight(
        backbone.text_encoder.embedding.token_embedding.embeddings,
        f"{text_prefix}.embeddings.token_embedding.weight",
    )
    loader.port_weight(
        backbone.text_encoder.embedding.position_embedding.position_embeddings,
        f"{text_prefix}.embeddings.position_embedding.weight",
    )
    for i, layer in enumerate(backbone.text_encoder.encoder_layers):
        p = f"{text_prefix}.encoder.layers.{i}"
        port_ln(layer.layer_norm_1, f"{p}.layer_norm1")
        num_heads = backbone.text_encoder.num_heads
        hidden_dim = backbone.text_encoder.hidden_dim
        head_dim = hidden_dim // num_heads

        def port_mha_weight(keras_dense, hf_name, is_output=False):
            def hook(x, _):
                w = x.T
                if is_output:
                    return w.reshape(num_heads, head_dim, hidden_dim)
                else:
                    return w.reshape(hidden_dim, num_heads, head_dim)

            loader.port_weight(
                keras_dense.kernel,
                f"{hf_name}.weight",
                hook_fn=hook,
            )
            if keras_dense.bias is not None:

                def bias_hook(x, _):
                    if is_output:
                        return x  # (hidden,)
                    else:
                        return x.reshape(num_heads, head_dim)

                loader.port_weight(
                    keras_dense.bias, f"{hf_name}.bias", hook_fn=bias_hook
                )

        port_mha_weight(layer.attention._query_dense, f"{p}.self_attn.q_proj")
        port_mha_weight(layer.attention._key_dense, f"{p}.self_attn.k_proj")
        port_mha_weight(layer.attention._value_dense, f"{p}.self_attn.v_proj")
        port_mha_weight(
            layer.attention._output_dense,
            f"{p}.self_attn.out_proj",
            is_output=True,
        )
        port_ln(layer.layer_norm_2, f"{p}.layer_norm2")
        port_dense(layer.dense_1, f"{p}.mlp.fc1")
        port_dense(layer.dense_2, f"{p}.mlp.fc2")

    port_ln(backbone.text_encoder.layer_norm, f"{text_prefix}.final_layer_norm")
    port_dense(backbone.text_projection, "text_projection")

    # Geometry Encoder.
    geo_prefix = "geometry_encoder"
    loader.port_weight(
        backbone.geometry_encoder.label_embed.embeddings,
        f"{geo_prefix}.label_embed.weight",
    )
    loader.port_weight(
        backbone.geometry_encoder.cls_embed.embeddings,
        f"{geo_prefix}.cls_embed.weight",
    )
    port_dense(
        backbone.geometry_encoder.boxes_direct_project,
        f"{geo_prefix}.boxes_direct_project",
    )
    port_conv(
        backbone.geometry_encoder.boxes_pool_project,
        f"{geo_prefix}.boxes_pool_project",
    )
    port_dense(
        backbone.geometry_encoder.boxes_pos_enc_project,
        f"{geo_prefix}.boxes_pos_enc_project",
    )
    port_ln(
        backbone.geometry_encoder.vision_layer_norm,
        f"{geo_prefix}.vision_layer_norm",
    )
    port_dense(backbone.geometry_encoder.final_proj, f"{geo_prefix}.final_proj")
    port_ln(
        backbone.geometry_encoder.prompt_layer_norm,
        f"{geo_prefix}.prompt_layer_norm",
    )
    for i, layer in enumerate(backbone.geometry_encoder.layers):
        p = f"{geo_prefix}.layers.{i}"
        port_ln(layer.layer_norm1, f"{p}.layer_norm1")
        port_attention(layer.self_attn, f"{p}.self_attn")
        port_ln(layer.layer_norm2, f"{p}.layer_norm2")
        port_attention(layer.cross_attn, f"{p}.cross_attn")
        port_ln(layer.layer_norm3, f"{p}.layer_norm3")
        port_mlp(layer.mlp, f"{p}.mlp")
    port_ln(
        backbone.geometry_encoder.output_layer_norm,
        f"{geo_prefix}.output_layer_norm",
    )

    # DETR Encoder.
    detr_enc_prefix = "detr_encoder"
    for i, layer in enumerate(backbone.detr_encoder.layers):
        p = f"{detr_enc_prefix}.layers.{i}"
        port_ln(layer.layer_norm1, f"{p}.layer_norm1")
        port_attention(layer.self_attn, f"{p}.self_attn")
        port_attention(layer.cross_attn, f"{p}.cross_attn")
        port_ln(layer.layer_norm2, f"{p}.layer_norm2")
        port_mlp(layer.mlp, f"{p}.mlp")
        port_ln(layer.layer_norm3, f"{p}.layer_norm3")

    # DETR Decoder.
    detr_dec_prefix = "detr_decoder"
    port_ln(
        backbone.detr_decoder.output_layer_norm,
        f"{detr_dec_prefix}.output_layer_norm",
    )
    port_decoder_mlp(
        backbone.detr_decoder.box_head, f"{detr_dec_prefix}.box_head"
    )
    loader.port_weight(
        backbone.detr_decoder.query_embed.embeddings,
        f"{detr_dec_prefix}.query_embed.weight",
    )
    loader.port_weight(
        backbone.detr_decoder.reference_points.embeddings,
        f"{detr_dec_prefix}.reference_points.weight",
    )
    loader.port_weight(
        backbone.detr_decoder.presence_token.embeddings,
        f"{detr_dec_prefix}.presence_token.weight",
    )
    port_decoder_mlp(
        backbone.detr_decoder.presence_head, f"{detr_dec_prefix}.presence_head"
    )
    port_ln(
        backbone.detr_decoder.presence_layer_norm,
        f"{detr_dec_prefix}.presence_layer_norm",
    )
    port_decoder_mlp(
        backbone.detr_decoder.ref_point_head,
        f"{detr_dec_prefix}.ref_point_head",
    )
    port_decoder_mlp(
        backbone.detr_decoder.box_rpb_embed_x,
        f"{detr_dec_prefix}.box_rpb_embed_x",
    )
    port_decoder_mlp(
        backbone.detr_decoder.box_rpb_embed_y,
        f"{detr_dec_prefix}.box_rpb_embed_y",
    )
    for i, layer in enumerate(backbone.detr_decoder.layers):
        p = f"{detr_dec_prefix}.layers.{i}"
        port_attention(layer.self_attn, f"{p}.self_attn")
        port_ln(layer.self_attn_layer_norm, f"{p}.self_attn_layer_norm")
        port_attention(layer.text_cross_attn, f"{p}.text_cross_attn")
        port_ln(
            layer.text_cross_attn_layer_norm, f"{p}.text_cross_attn_layer_norm"
        )
        port_attention(layer.vision_cross_attn, f"{p}.vision_cross_attn")
        port_ln(
            layer.vision_cross_attn_layer_norm,
            f"{p}.vision_cross_attn_layer_norm",
        )
        port_mlp(layer.mlp, f"{p}.mlp")
        port_ln(layer.mlp_layer_norm, f"{p}.mlp_layer_norm")

    # Mask Decoder.
    mask_prefix = "mask_decoder"
    for i in range(len(backbone.mask_decoder.pixel_decoder.conv_layers)):
        p = f"{mask_prefix}.pixel_decoder"
        port_conv(
            backbone.mask_decoder.pixel_decoder.conv_layers[i],
            f"{p}.conv_layers.{i}",
        )
        port_gn(backbone.mask_decoder.pixel_decoder.norms[i], f"{p}.norms.{i}")
    for i in range(len(backbone.mask_decoder.mask_embedder.layers)):
        port_dense(
            backbone.mask_decoder.mask_embedder.layers[i],
            f"{mask_prefix}.mask_embedder.layers.{i}",
        )
    port_conv(
        backbone.mask_decoder.instance_projection,
        f"{mask_prefix}.instance_projection",
    )
    port_conv(
        backbone.mask_decoder.semantic_projection,
        f"{mask_prefix}.semantic_projection",
    )
    port_attention(
        backbone.mask_decoder.prompt_cross_attn,
        f"{mask_prefix}.prompt_cross_attn",
    )
    port_ln(
        backbone.mask_decoder.prompt_cross_attn_norm,
        f"{mask_prefix}.prompt_cross_attn_norm",
    )

    # Top Level Backbone Layers.
    scoring_prefix = "dot_product_scoring"
    port_decoder_mlp(
        backbone.dot_product_scoring.text_mlp, f"{scoring_prefix}.text_mlp"
    )
    port_ln(
        backbone.dot_product_scoring.text_mlp_out_norm,
        f"{scoring_prefix}.text_mlp_out_norm",
    )
    port_dense(
        backbone.dot_product_scoring.text_proj, f"{scoring_prefix}.text_proj"
    )
    port_dense(
        backbone.dot_product_scoring.query_proj, f"{scoring_prefix}.query_proj"
    )


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]
    merges = [" ".join(item) for item in merges]
    return cls(vocabulary=vocab, merges=merges, **kwargs)
