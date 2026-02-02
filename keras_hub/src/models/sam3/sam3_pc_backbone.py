import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.sam3.sam3_dot_product_scoring import (
    SAM3DotProductScoring,
)
from keras_hub.src.models.sam3.sam3_layers import SAM3BoxDecoder


@keras_hub_export("keras_hub.models.SAM3PromptableConceptBackbone")
class SAM3PromptableConceptBackbone(Backbone):
    """A backbone for the Segment Anything Model 3 (SAM3).

    SAM3 is a multi-modal model that supports text and geometry prompts (boxes)
    to perform object segmentation. It consists of a vision encoder, a text
    encoder, a geometry encoder for processing box prompts, and a DETR-based
    encoder-decoder architecture to fuse multi-modal features and predict
    segmentation masks.

    Args:
        vision_encoder: `keras_hub.layers.SAM3VisionEncoder`. A feature
            extractor for the input images.
        text_encoder: `keras_hub.layers.SAM3TextEncoder`. A Keras layer to
            compute embeddings for text prompts.
        geometry_encoder: `keras_hub.layers.SAM3GeometryEncoder`. A Keras layer
            to compute embeddings for geometry (box) prompts.
        detr_encoder: `keras_hub.layers.SAM3DetrEncoder`. A transformer-based
            encoder that fuses vision and prompt features.
        detr_decoder: `keras_hub.layers.SAM3DetrDecoder`. A transformer-based
            decoder that predicts object queries.
        mask_decoder: `keras_hub.layers.SAM3MaskDecoder`. A Keras layer to
            generate segmentation masks given the object queries and fused
            features.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done in float32 precision regardless of dtype. Defaults to
            `bfloat16`.

    Example:
    ```python
    import numpy as np
    import keras_hub

    vision_encoder = keras_hub.layers.SAM3VisionEncoder(
        image_shape=(224, 224, 3),
        patch_size=14,
        num_layers=2,
        hidden_dim=32,
        intermediate_dim=128,
        num_heads=2,
        fpn_hidden_dim=32,
        fpn_scale_factors=[4.0, 2.0, 1.0, 0.5],
        pretrain_image_shape=(112, 112, 3),
        window_size=2,
        global_attn_indexes=[1, 2],
    )
    text_encoder = keras_hub.layers.SAM3TextEncoder(
        vocabulary_size=1024,
        embedding_dim=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        intermediate_dim=128,
    )
    geometry_encoder = keras_hub.layers.SAM3GeometryEncoder(
        num_layers=3,
        hidden_dim=32,
        intermediate_dim=128,
        num_heads=2,
        roi_size=7,
    )
    detr_encoder = keras_hub.layers.SAM3DetrEncoder(
        num_layers=3,
        hidden_dim=32,
        intermediate_dim=128,
        num_heads=2,
    )
    detr_decoder = keras_hub.layers.SAM3DetrDecoder(
        image_shape=(224, 224, 3),
        patch_size=14,
        num_layers=2,
        hidden_dim=32,
        intermediate_dim=128,
        num_heads=2,
        num_queries=100,
    )
    mask_decoder = keras_hub.layers.SAM3MaskDecoder(
        num_upsampling_stages=3,
        hidden_dim=32,
        num_heads=2,
    )
    backbone = keras_hub.models.SAM3PromptableConceptBackbone(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        geometry_encoder=geometry_encoder,
        detr_encoder=detr_encoder,
        detr_decoder=detr_decoder,
        mask_decoder=mask_decoder,
    )
    input_data = {
        "pixel_values": np.ones((2, 224, 224, 3), dtype="float32"),
        "token_ids": np.ones((2, 32), dtype="int32"),
        "padding_mask": np.ones((2, 32), dtype="bool"),
        "boxes": np.zeros((2, 1, 5), dtype="float32"),
        "box_labels": np.zeros((2, 1), dtype="int32"),
    }
    outputs = backbone(input_data)
    ```
    """

    def __init__(
        self,
        vision_encoder,
        text_encoder,
        geometry_encoder,
        detr_encoder,
        detr_decoder,
        mask_decoder,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.geometry_encoder = geometry_encoder
        self.detr_encoder = detr_encoder
        self.detr_decoder = detr_decoder
        self.mask_decoder = mask_decoder

        self.text_projection = layers.Dense(
            self.detr_encoder.hidden_dim, dtype=dtype, name="text_projection"
        )
        self.dot_product_scoring = SAM3DotProductScoring(
            hidden_dim=self.detr_decoder.hidden_dim,
            intermediate_dim=self.detr_decoder.intermediate_dim,
            dropout_rate=self.detr_decoder.dropout_rate,
            layer_norm_epsilon=1e-6,
            dtype=dtype,
            name="dot_product_scoring",
        )
        self.box_decoder = SAM3BoxDecoder(dtype=dtype, name="box_decoder")

        # === Functional Model ===
        pixel_value_input = layers.Input(
            shape=self.vision_encoder.image_shape, name="pixel_values"
        )
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="bool", name="padding_mask"
        )
        box_input = keras.Input(shape=(None, 5), dtype="float32", name="boxes")
        box_label_input = keras.Input(
            shape=(None,), dtype="int32", name="box_labels"
        )
        box_masks = ops.cast(
            ops.where(ops.not_equal(box_label_input, -10), 1, 0), dtype="bool"
        )

        fpn_hidden_states, fpn_position_encodings = self.vision_encoder(
            pixel_value_input
        )
        fpn_hidden_states = fpn_hidden_states[:-1]
        fpn_position_encodings = fpn_position_encodings[:-1]
        text_features = self.text_encoder(token_id_input, padding_mask_input)
        text_features = self.text_projection(text_features)
        geometry_prompt_features, geometry_prompt_mask = self.geometry_encoder(
            box_input,
            box_label_input,
            box_masks,
            fpn_hidden_states=fpn_hidden_states[-1],
            fpn_position_encodings=fpn_position_encodings[-1],
        )
        combined_prompt_features = ops.concatenate(
            [text_features, geometry_prompt_features], axis=1
        )
        combined_prompt_masks = ops.concatenate(
            [padding_mask_input, geometry_prompt_mask], axis=1
        )
        encoder_outputs = self.detr_encoder(
            vision_features=fpn_hidden_states[-1],
            text_features=combined_prompt_features,
            vision_pos_embeds=fpn_position_encodings[-1],
            text_masks=combined_prompt_masks,
        )
        decoder_outputs = self.detr_decoder(
            vision_features=encoder_outputs[0],
            text_features=encoder_outputs[2],
            vision_pos_encodings=encoder_outputs[1],
            text_masks=combined_prompt_masks,
        )
        decoder_hidden_states = decoder_outputs[0]
        decoder_presence_logits = decoder_outputs[2]
        all_box_offsets = self.detr_decoder.box_head(decoder_hidden_states)
        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_hidden_states,
            text_features=encoder_outputs[2],
            text_masks=combined_prompt_masks,
        )
        pred_boxes, pred_logits, presence_logits = self.box_decoder(
            box_offsets=all_box_offsets,
            reference_boxes=decoder_outputs[1],
            pred_logits=all_pred_logits,
            presence_logits=decoder_presence_logits,
        )
        pred_masks, semantic_segs = self.mask_decoder(
            decoder_queries=decoder_hidden_states[:, -1],
            backbone_features=fpn_hidden_states,
            encoder_hidden_states=encoder_outputs[0],
            prompt_features=combined_prompt_features,
            prompt_masks=combined_prompt_masks,
        )

        super().__init__(
            inputs={
                "pixel_values": pixel_value_input,
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
                "boxes": box_input,
                "box_labels": box_label_input,
            },
            outputs={
                "pred_masks": pred_masks,
                "pred_boxes": pred_boxes,
                "pred_logits": pred_logits,
                "presence_logits": presence_logits,
                "semantic_segs": semantic_segs,
            },
            dtype=dtype,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder": keras.layers.serialize(self.vision_encoder),
                "text_encoder": keras.layers.serialize(self.text_encoder),
                "geometry_encoder": keras.layers.serialize(
                    self.geometry_encoder
                ),
                "detr_encoder": keras.layers.serialize(self.detr_encoder),
                "detr_decoder": keras.layers.serialize(self.detr_decoder),
                "mask_decoder": keras.layers.serialize(self.mask_decoder),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()

        # Propagate `dtype` to submodels if needed.
        if "dtype" in config and config["dtype"] is not None:
            dtype_config = config["dtype"]
            if "dtype" not in config["vision_encoder"]["config"]:
                config["vision_encoder"]["config"]["dtype"] = dtype_config
            if "dtype" not in config["text_encoder"]["config"]:
                config["text_encoder"]["config"]["dtype"] = dtype_config
            if "dtype" not in config["geometry_encoder"]["config"]:
                config["geometry_encoder"]["config"]["dtype"] = dtype_config
            if "dtype" not in config["detr_encoder"]["config"]:
                config["detr_encoder"]["config"]["dtype"] = dtype_config
            if "dtype" not in config["detr_decoder"]["config"]:
                config["detr_decoder"]["config"]["dtype"] = dtype_config
            if "dtype" not in config["mask_decoder"]["config"]:
                config["mask_decoder"]["config"]["dtype"] = dtype_config

        # Propagate `image_shape` to submodels if needed.
        if "image_shape" in config and config["image_shape"] is not None:
            image_shape = config.pop("image_shape")
            if "image_shape" in config["vision_encoder"]["config"]:
                config["vision_encoder"]["config"]["image_shape"] = image_shape
            if "image_shape" in config["detr_decoder"]["config"]:
                config["detr_decoder"]["config"]["image_shape"] = image_shape

        config.update(
            {
                "vision_encoder": keras.layers.deserialize(
                    config["vision_encoder"]
                ),
                "text_encoder": keras.layers.deserialize(
                    config["text_encoder"]
                ),
                "geometry_encoder": keras.layers.deserialize(
                    config["geometry_encoder"]
                ),
                "detr_encoder": keras.layers.deserialize(
                    config["detr_encoder"]
                ),
                "detr_decoder": keras.layers.deserialize(
                    config["detr_decoder"]
                ),
                "mask_decoder": keras.layers.deserialize(
                    config["mask_decoder"]
                ),
            }
        )
        return super().from_config(config)
