import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.sam3.roi_align import roi_align
from keras_hub.src.models.sam3.sam3_layers import SAM3MLP
from keras_hub.src.models.sam3.sam3_layers import SAM3Attention
from keras_hub.src.models.sam3.sam3_layers import SAM3SinePositionEmbedding
from keras_hub.src.models.sam3.sam3_utils import box_cxcywh_to_xyxy
from keras_hub.src.models.sam3.sam3_utils import concatenate_padded_sequences


class SAM3GeometryEncoderLayer(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_heads,
        hidden_activation="relu",
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.num_heads = int(num_heads)
        self.hidden_activation = hidden_activation
        self.dropout_rate = float(dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm1",
        )
        self.self_attn = SAM3Attention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dtype=self.dtype_policy,
            name="self_attn",
        )
        self.dropout = layers.Dropout(
            rate=self.dropout_rate, dtype=self.dtype_policy, name="dropout"
        )
        self.cross_attn = SAM3Attention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dtype=self.dtype_policy,
            name="cross_attn",
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm2",
        )
        self.mlp = SAM3MLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            activation=self.hidden_activation,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.layer_norm3 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm3",
        )

    def build(
        self,
        prompt_feats_shape,
        vision_feats_shape,
        vision_pos_encodings_shape,
        prompt_masks_shape,
    ):
        self.layer_norm1.build(prompt_feats_shape)
        self.self_attn.build(
            prompt_feats_shape, prompt_feats_shape, prompt_feats_shape
        )
        self.dropout.build(prompt_feats_shape)
        self.layer_norm2.build(prompt_feats_shape)
        self.cross_attn.build(
            prompt_feats_shape, vision_feats_shape, vision_feats_shape
        )
        self.layer_norm3.build(prompt_feats_shape)
        self.mlp.build(prompt_feats_shape)

    def call(
        self,
        prompt_feats,
        vision_feats,
        vision_pos_encodings,
        prompt_masks,
        training=None,
    ):
        residual = prompt_feats
        hidden_states = self.layer_norm1(prompt_feats, training=training)
        hidden_states = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attention_mask=prompt_masks,
            training=training,
        )
        hidden_states = ops.add(
            self.dropout(hidden_states, training=training), residual
        )

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states, training=training)
        key = ops.add(vision_feats, vision_pos_encodings)
        hidden_states = self.cross_attn(
            query=hidden_states, key=key, value=vision_feats, training=training
        )
        hidden_states = ops.add(
            self.dropout(hidden_states, training=training), residual
        )

        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states, training=training)
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = ops.add(
            self.dropout(hidden_states, training=training), residual
        )
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "hidden_activation": self.hidden_activation,
                "dropout_rate": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(
        self,
        prompt_feats_shape,
        vision_feats_shape,
        vision_pos_encodings_shape,
        prompt_masks_shape,
    ):
        return prompt_feats_shape


@keras_hub_export("keras_hub.layers.SAM3GeometryEncoder")
class SAM3GeometryEncoder(layers.Layer):
    """A geometry encoder for the Segment Anything Model 3 (SAM3).

    This layer implements a transformer-based encoder for processing geometry
    prompts (boxes). It extracts features from the input boxes, pools vision
    features based on the boxes, and fuses them with transformer layers.

    Args:
        num_layers: int. The number of transformer layers.
        hidden_dim: int. The hidden dimension of the transformer layers.
        intermediate_dim: int. The dimension of the intermediate layer in the
            transformer's MLP.
        num_heads: int. The number of attention heads.
        roi_size: int. The size of the ROI pooling for boxes.
        hidden_activation: str. The activation function for the transformer
            layers. Defaults to `"relu"`.
        dropout_rate: float. The dropout rate for the MLP and attention.
            Defaults to `0.0`.
        layer_norm_epsilon: float. The epsilon value for layer normalization.
            Defaults to `1e-6`.
    """

    def __init__(
        self,
        num_layers,
        hidden_dim,
        intermediate_dim,
        num_heads,
        roi_size,
        hidden_activation="relu",
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.num_heads = int(num_heads)
        self.roi_size = int(roi_size)
        self.hidden_activation = hidden_activation
        self.dropout_rate = float(dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        self.position_encoding = SAM3SinePositionEmbedding(
            num_pos_feats=self.hidden_dim // 2,
            normalize=True,
            dtype=self.dtype_policy,
            name="position_encoding",
        )
        self.label_embed = layers.Embedding(
            2, self.hidden_dim, dtype=self.dtype_policy, name="label_embed"
        )
        self.cls_embed = layers.Embedding(
            1, self.hidden_dim, dtype=self.dtype_policy, name="cls_embed"
        )

        # Box encoding layers.
        self.boxes_direct_project = layers.Dense(
            self.hidden_dim,
            dtype=self.dtype_policy,
            name="boxes_direct_project",
        )
        self.boxes_pool_project = layers.Conv2D(
            self.hidden_dim,
            kernel_size=self.roi_size,
            dtype=self.dtype_policy,
            name="boxes_pool_project",
        )
        self.boxes_pos_enc_project = layers.Dense(
            self.hidden_dim,
            dtype=self.dtype_policy,
            name="boxes_pos_enc_project",
        )

        # Image feature normalization.
        self.vision_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="vision_layer_norm",
        )

        # Prompt projection and normalization.
        self.final_proj = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="final_proj"
        )
        self.prompt_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="prompt_layer_norm",
        )

        # Transformer layers.
        self.layers = [
            SAM3GeometryEncoderLayer(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                hidden_activation=self.hidden_activation,
                layer_norm_epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name=f"layer_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.output_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="output_layer_norm",
        )

    def build(
        self,
        box_embeddings_shape,
        box_masks_shape,
        box_labels_shape,
        fpn_hidden_states_shape,
        fpn_position_encodings_shape,
    ):
        batch_size = fpn_hidden_states_shape[0]
        self.height = fpn_hidden_states_shape[1]
        self.width = fpn_hidden_states_shape[2]
        self.input_hidden_dim = fpn_hidden_states_shape[-1]

        self.position_encoding.build()
        self.vision_layer_norm.build(fpn_hidden_states_shape)

        box_proj_input_shape = list(box_embeddings_shape)
        box_proj_input_shape[-1] = box_embeddings_shape[-1] - 1
        self.boxes_direct_project.build(tuple(box_proj_input_shape))

        sampled_feature_shape = [
            batch_size,
            self.roi_size,
            self.roi_size,
            self.input_hidden_dim,
        ]
        self.boxes_pool_project.build(sampled_feature_shape)

        pos_enc_shape = [batch_size, None, self.input_hidden_dim + 2]
        self.boxes_pos_enc_project.build(pos_enc_shape)
        self.label_embed.build([batch_size, 1])
        self.cls_embed.build([batch_size, 1])

        prompt_embed_shape = [batch_size, None, self.hidden_dim]
        self.final_proj.build(prompt_embed_shape)
        self.prompt_layer_norm.build(prompt_embed_shape)

        vision_feat_flat_shape = [
            batch_size,
            self.height * self.width,
            self.input_hidden_dim,
        ]
        for layer in self.layers:
            layer.build(
                prompt_embed_shape,
                vision_feat_flat_shape,
                vision_feat_flat_shape,
                None,
            )
        self.output_layer_norm.build(prompt_embed_shape)

    def _encode_box_coordinates(self, center_x, center_y, width, height):
        pos_x, pos_y = self.position_encoding.encode_1d_positions(
            center_x, center_y
        )
        pos = ops.concatenate(
            (pos_y, pos_x, height[:, None], width[:, None]), axis=1
        )
        return pos

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, vision_features):
        # Keras passes the masks as concrete tensors for both the
        # true and false functions to build the output shape. So, we
        # need to handle the case when 0 size masks is passed and
        # dispatch the call to `_no_box_embeddings`. Note that we can't call
        # the lambda directly since the inputs are bound to different
        # values when called with concrete values.
        if boxes.shape[1] == 0:
            return self._no_box_embeddings(boxes, boxes_mask)

        # The shape of boxes is different from HF's implementation.
        # boxes: [batch_size, num_boxes, 5] where the last dimension is
        # (batch_index, cx, cy, w, h)
        boxes_indices = boxes[..., 0:1]
        boxes = boxes[..., 1:]
        batch_size = ops.shape(boxes)[0]
        boxes_embed = self.boxes_direct_project(boxes)

        # Pool features using ROI align.
        # Convert boxes from cxcywh to xyxy format and denormalize.
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = ops.array(
            [[[self.width, self.height, self.width, self.height]]],
            dtype=boxes.dtype,
        )
        boxes_xyxy = ops.multiply(boxes_xyxy, scale)
        boxes_xyxy = ops.reshape(boxes_xyxy, (-1, 4))
        # Add batch indices to boxes for roi_align.
        rois = ops.concatenate(
            [ops.reshape(boxes_indices, (-1, 1)), boxes_xyxy], axis=-1
        )
        sampled_features = roi_align(
            vision_features,
            rois,
            (self.roi_size, self.roi_size),
            spatial_scale=1.0,
            height=self.height,
            width=self.width,
            hidden_dim=self.input_hidden_dim,
        )

        pooled_projection = self.boxes_pool_project(sampled_features)
        pooled_projection = ops.reshape(
            pooled_projection, (batch_size, -1, self.hidden_dim)
        )
        boxes_embed = ops.add(boxes_embed, pooled_projection)

        # Add position encoding.
        center_x, center_y, box_width, box_height = ops.unstack(
            boxes, num=4, axis=-1
        )
        pos_enc = self._encode_box_coordinates(
            ops.reshape(center_x, (-1,)),
            ops.reshape(center_y, (-1,)),
            ops.reshape(box_width, (-1,)),
            ops.reshape(box_height, (-1,)),
        )
        pos_enc = ops.reshape(
            pos_enc,
            (batch_size, -1, self.position_encoding.num_pos_feats * 2 + 2),
        )
        pos_projection = self.boxes_pos_enc_project(pos_enc)
        boxes_embed = ops.add(boxes_embed, pos_projection)

        # Add label embeddings (positive / negative).
        label_embed = self.label_embed(ops.cast(boxes_labels, dtype="int32"))
        return ops.add(label_embed, boxes_embed), boxes_mask

    def _no_box_embeddings(self, box_embeddings, box_masks):
        batch_size = ops.shape(box_embeddings)[0]
        num_boxes = ops.shape(box_embeddings)[1]
        return (
            ops.zeros(
                (batch_size, num_boxes, self.hidden_dim),
                dtype=box_embeddings.dtype,
            ),
            box_masks,
        )

    def call(
        self,
        box_embeddings,
        box_masks,
        box_labels,
        fpn_hidden_states,
        fpn_position_encodings,
        training=None,
    ):
        # Prepare vision features for cross-attention.
        vision_feats_flat = ops.reshape(
            fpn_hidden_states,
            (-1, self.height * self.width, self.input_hidden_dim),
        )
        vision_pos_embeds_flat = ops.reshape(
            fpn_position_encodings,
            (-1, self.height * self.width, self.input_hidden_dim),
        )

        # Normalize image features for pooling operations.
        normalized_image_feats = self.vision_layer_norm(fpn_hidden_states)

        prompt_embeds, prompt_mask = ops.cond(
            ops.equal(ops.shape(box_embeddings)[1], 0),
            lambda: self._no_box_embeddings(box_embeddings, box_masks),
            lambda: self._encode_boxes(
                box_embeddings, box_masks, box_labels, normalized_image_feats
            ),
        )

        # Add CLS token (always valid).
        cls_embed = ops.reshape(
            self.cls_embed._embeddings, (1, 1, self.hidden_dim)
        )
        cls_embed = ops.tile(cls_embed, (ops.shape(prompt_embeds)[0], 1, 1))
        cls_mask = ops.ones_like(cls_embed[:, :, 0], dtype=prompt_mask.dtype)

        prompt_embeds, prompt_mask = concatenate_padded_sequences(
            prompt_embeds,
            prompt_mask,
            ops.shape(prompt_embeds)[1],
            cls_embed,
            cls_mask,
            1,
            self.hidden_dim,
        )
        prompt_embeds = self.prompt_layer_norm(self.final_proj(prompt_embeds))

        # Apply transformer layers with cross-attention to vision features.
        for layer in self.layers:
            prompt_embeds = layer(
                prompt_embeds,
                vision_feats_flat,
                vision_pos_embeds_flat,
                prompt_mask,
                training=training,
            )

        # Final output normalization.
        prompt_embeds = self.output_layer_norm(prompt_embeds, training=training)
        return prompt_embeds, prompt_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "roi_size": self.roi_size,
                "hidden_activation": self.hidden_activation,
                "dropout_rate": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(
        self,
        box_embeddings_shape,
        box_masks_shape,
        box_labels_shape,
        fpn_hidden_states_shape,
        fpn_position_encodings_shape,
    ):
        batch_size = fpn_hidden_states_shape[0]
        num_boxes = box_embeddings_shape[1]
        seq_len = None
        if num_boxes is not None:
            seq_len = num_boxes + 1
        return [batch_size, seq_len, self.hidden_dim], [batch_size, seq_len]

    def compute_output_spec(
        self,
        box_embeddings,
        box_masks,
        box_labels,
        fpn_hidden_states,
        fpn_position_encodings,
    ):
        prompt_embeds_shape, prompt_mask_shape = self.compute_output_shape(
            box_embeddings.shape,
            box_masks.shape,
            box_labels.shape,
            fpn_hidden_states.shape,
            fpn_position_encodings.shape,
        )
        return (
            keras.KerasTensor(prompt_embeds_shape, dtype=self.compute_dtype),
            keras.KerasTensor(prompt_mask_shape, dtype="bool"),
        )
