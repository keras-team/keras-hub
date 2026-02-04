import math

from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.sam3.sam3_layers import SAM3MLP
from keras_hub.src.models.sam3.sam3_layers import SAM3Attention
from keras_hub.src.models.sam3.sam3_layers import SAM3DecoderMLP
from keras_hub.src.models.sam3.sam3_layers import SAM3SinePositionEmbedding
from keras_hub.src.models.sam3.sam3_utils import box_cxcywh_to_xyxy
from keras_hub.src.models.sam3.sam3_utils import create_bidirectional_mask
from keras_hub.src.models.sam3.sam3_utils import inverse_sigmoid


class SAM3DetrDecoderLayer(layers.Layer):
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
        self.dropout_rate = float(dropout_rate)
        self.hidden_activation = hidden_activation
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        self.self_attn = SAM3Attention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dtype=self.dtype_policy,
            name="self_attn",
        )
        self.self_attn_dropout = layers.Dropout(
            rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="self_attn_dropout",
        )
        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attn_layer_norm",
        )
        self.text_cross_attn = SAM3Attention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dtype=self.dtype_policy,
            name="text_cross_attn",
        )
        self.text_cross_attn_dropout = layers.Dropout(
            rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="text_cross_attn_dropout",
        )
        self.text_cross_attn_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="text_cross_attn_layer_norm",
        )
        self.vision_cross_attn = SAM3Attention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dtype=self.dtype_policy,
            name="vision_cross_attn",
        )
        self.vision_cross_attn_dropout = layers.Dropout(
            rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="vision_cross_attn_dropout",
        )
        self.vision_cross_attn_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="vision_cross_attn_layer_norm",
        )
        self.mlp = SAM3MLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            activation=self.hidden_activation,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.mlp_dropout = layers.Dropout(
            rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="mlp_dropout",
        )
        self.mlp_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="mlp_layer_norm",
        )

    def build(
        self,
        hidden_states_shape,
        query_pos_shape,
        text_features_shape,
        vision_features_shape,
        vision_pos_encodings_shape,
        text_cross_attn_masks_shape,
        vision_cross_attn_masks_shape,
    ):
        self.self_attn.build(
            hidden_states_shape, hidden_states_shape, hidden_states_shape
        )
        self.self_attn_dropout.build(hidden_states_shape)
        self.self_attn_layer_norm.build(hidden_states_shape)
        self.text_cross_attn.build(
            hidden_states_shape, text_features_shape, text_features_shape
        )
        self.text_cross_attn_dropout.build(hidden_states_shape)
        self.text_cross_attn_layer_norm.build(hidden_states_shape)
        self.vision_cross_attn.build(
            hidden_states_shape, hidden_states_shape, vision_features_shape
        )
        self.vision_cross_attn_dropout.build(hidden_states_shape)
        self.vision_cross_attn_layer_norm.build(hidden_states_shape)
        self.mlp.build(hidden_states_shape)
        self.mlp_dropout.build(hidden_states_shape)
        self.mlp_layer_norm.build(hidden_states_shape)

    def call(
        self,
        hidden_states,
        query_pos,
        text_features,
        vision_features,
        vision_pos_encodings,
        text_cross_attn_masks,
        vision_cross_attn_masks,
        training=None,
    ):
        # Prepend zeros to query_pos for presence token.
        query_pos = ops.pad(query_pos, [[0, 0], [1, 0], [0, 0]])

        # Self-attention with query position encoding.
        residual = hidden_states
        query_with_pos = ops.add(hidden_states, query_pos)
        attn_output = self.self_attn(
            query=query_with_pos,
            key=query_with_pos,
            value=hidden_states,
            attention_mask=None,
            training=training,
        )
        hidden_states = ops.add(
            residual, self.self_attn_dropout(attn_output, training=training)
        )
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Text cross-attention: queries attend to text features.
        residual = hidden_states
        query_with_pos = ops.add(hidden_states, query_pos)
        attn_output = self.text_cross_attn(
            query=query_with_pos,
            key=text_features,
            value=text_features,
            attention_mask=text_cross_attn_masks,
            training=training,
        )
        hidden_states = ops.add(
            residual,
            self.text_cross_attn_dropout(attn_output, training=training),
        )
        hidden_states = self.text_cross_attn_layer_norm(hidden_states)

        # Vision cross-attention: queries attend to vision features (with RPB)
        residual = hidden_states
        query_with_pos = ops.add(hidden_states, query_pos)
        key_with_pos = ops.add(vision_features, vision_pos_encodings)
        attn_output = self.vision_cross_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=vision_features,
            attention_bias=vision_cross_attn_masks,
            training=training,
        )
        hidden_states = ops.add(
            residual,
            self.vision_cross_attn_dropout(attn_output, training=training),
        )
        hidden_states = self.vision_cross_attn_layer_norm(hidden_states)

        # MLP.
        residual = hidden_states
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = ops.add(
            residual, self.mlp_dropout(hidden_states, training=training)
        )
        hidden_states = self.mlp_layer_norm(hidden_states)
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
        hidden_states_shape,
        query_pos_shape,
        text_features_shape,
        vision_features_shape,
        vision_pos_encodings_shape,
        text_cross_attn_masks_shape,
        vision_cross_attn_masks_shape,
    ):
        return hidden_states_shape


@keras_hub_export("keras_hub.layers.SAM3DetrDecoder")
class SAM3DetrDecoder(layers.Layer):
    """A DETR decoder for the Segment Anything Model 3 (SAM3).

    This layer implements a transformer-based decoder that predicts object
    queries. It processes object queries and fused features through multiple
    layers of self-attention and cross-attention.

    Args:
        image_shape: tuple. The shape of the input image
            (height, width, channels).
        patch_size: int. The size of the patches to be extracted from the image.
        num_layers: int. The number of transformer layers.
        hidden_dim: int. The hidden dimension of the transformer layers.
        intermediate_dim: int. The dimension of the intermediate layer in the
            transformer's MLP.
        num_heads: int. The number of attention heads.
        num_queries: int. The number of object queries.
        hidden_activation: str. The activation function for the transformer
            layers. Defaults to `"relu"`.
        dropout_rate: float. The dropout rate for the MLP and attention.
            Defaults to `0.0`.
        layer_norm_epsilon: float. The epsilon value for layer normalization.
            Defaults to `1e-6`.
    """

    def __init__(
        self,
        image_shape,
        patch_size,
        num_layers,
        hidden_dim,
        intermediate_dim,
        num_heads,
        num_queries,
        hidden_activation="relu",
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_shape = (
            int(image_shape[0]),
            int(image_shape[1]),
            int(image_shape[2]),
        )
        self.patch_size = int(patch_size)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.num_heads = int(num_heads)
        self.num_queries = int(num_queries)
        self.hidden_activation = hidden_activation
        self.dropout_rate = float(dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.height = self.image_shape[0] // self.patch_size
        self.width = self.image_shape[1] // self.patch_size

        self.layers = [
            SAM3DetrDecoderLayer(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                num_heads=self.num_heads,
                hidden_activation=self.hidden_activation,
                dropout_rate=self.dropout_rate,
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
        self.box_head = SAM3DecoderMLP(
            num_layers=3,
            hidden_dim=self.hidden_dim,
            output_dim=4,
            dtype=self.dtype_policy,
            name="box_head",
        )
        self.query_embed = layers.Embedding(
            self.num_queries,
            self.hidden_dim,
            dtype=self.dtype_policy,
            name="query_embed",
        )
        self.reference_points = layers.Embedding(
            self.num_queries,
            4,
            dtype=self.dtype_policy,
            name="reference_points",
        )
        self.presence_token = layers.Embedding(
            1,
            self.hidden_dim,
            dtype=self.dtype_policy,
            name="presence_token",
        )
        self.presence_head = SAM3DecoderMLP(
            num_layers=3,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            dtype=self.dtype_policy,
            name="presence_head",
        )
        self.presence_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="presence_layer_norm",
        )
        self.clamp_presence_logit_max_val = 10.0
        self.ref_point_head = SAM3DecoderMLP(
            num_layers=2,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dtype=self.dtype_policy,
            name="ref_point_head",
        )
        self.box_rpb_embed_x = SAM3DecoderMLP(
            num_layers=2,
            hidden_dim=self.hidden_dim,
            output_dim=self.num_heads,
            dtype=self.dtype_policy,
            name="box_rpb_embed_x",
        )
        self.box_rpb_embed_y = SAM3DecoderMLP(
            num_layers=2,
            hidden_dim=self.hidden_dim,
            output_dim=self.num_heads,
            dtype=self.dtype_policy,
            name="box_rpb_embed_y",
        )
        self.position_encoding = SAM3SinePositionEmbedding(
            num_pos_feats=self.hidden_dim // 2,
            normalize=False,
            dtype=self.dtype_policy,
            name="position_encoding",
        )

    def build(
        self,
        vision_features_shape,
        text_features_shape,
        vision_pos_encodings_shape,
        text_masks_shape,
    ):
        self.query_embed.build()
        self.reference_points.build()
        self.presence_token.build()
        self.position_encoding.build()
        batch_size = vision_features_shape[0]
        vision_len = vision_features_shape[1]
        hidden_states_shape = [
            batch_size,
            1 + self.num_queries,
            self.hidden_dim,
        ]
        text_cross_attn_masks_shape = [
            batch_size,
            1,
            1 + self.num_queries,
            text_masks_shape[-1],
        ]
        query_pos_shape = [batch_size, self.num_queries, self.hidden_dim]
        vision_cross_attn_masks_shape = [
            batch_size,
            self.num_heads,
            1 + self.num_queries,
            vision_len,
        ]
        query_hidden_state_shape = [
            batch_size,
            self.num_queries,
            self.hidden_dim,
        ]
        presence_hidden_shape = [batch_size, 1, self.hidden_dim]
        query_sine_embed_shape = [
            batch_size,
            self.num_queries,
            self.hidden_dim // 2 * 4,
        ]
        deltas_x_log_shape = [batch_size, self.num_queries, self.width, 2]
        deltas_y_log_shape = [batch_size, self.num_queries, self.height, 2]

        self.output_layer_norm.build(query_hidden_state_shape)
        self.box_head.build(query_hidden_state_shape)
        self.presence_layer_norm.build(presence_hidden_shape)
        self.presence_head.build(presence_hidden_shape)
        self.ref_point_head.build(query_sine_embed_shape)
        self.box_rpb_embed_x.build(deltas_x_log_shape)
        self.box_rpb_embed_y.build(deltas_y_log_shape)
        for layer in self.layers:
            layer.build(
                hidden_states_shape,
                query_pos_shape,
                text_features_shape,
                vision_features_shape,
                vision_pos_encodings_shape,
                text_cross_attn_masks_shape,
                vision_cross_attn_masks_shape,
            )

    def _get_coords(self, height, width, dtype):
        coords_h = ops.divide(ops.arange(height, dtype=dtype), height)
        coords_w = ops.divide(ops.arange(width, dtype=dtype), width)
        return coords_h, coords_w

    def _get_rpb_matrix(self, reference_boxes):
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)

        # Generate coordinate grids.
        coords_h, coords_w = self._get_coords(
            self.height, self.width, reference_boxes.dtype
        )

        # Compute deltas between coordinates and box boundaries.
        deltas_y = ops.subtract(
            ops.reshape(coords_h, (1, -1, 1)),
            ops.reshape(boxes_xyxy, (-1, 1, 4))[:, :, 1:4:2],
        )
        deltas_y = ops.reshape(deltas_y, (-1, self.num_queries, self.height, 2))
        deltas_x = ops.subtract(
            ops.reshape(coords_w, (1, -1, 1)),
            ops.reshape(boxes_xyxy, (-1, 1, 4))[:, :, 0:3:2],
        )
        deltas_x = ops.reshape(deltas_x, (-1, self.num_queries, self.width, 2))

        # Apply log-scale encoding.
        deltas_x_log = ops.multiply(deltas_x, 8.0)
        deltas_x_log = ops.divide(
            ops.multiply(
                ops.sign(deltas_x_log),
                ops.log2(ops.add(ops.abs(deltas_x_log), 1.0)),
            ),
            math.log2(8),
        )
        deltas_y_log = ops.multiply(deltas_y, 8.0)
        deltas_y_log = ops.divide(
            ops.multiply(
                ops.sign(deltas_y_log),
                ops.log2(ops.add(ops.abs(deltas_y_log), 1.0)),
            ),
            math.log2(8),
        )

        # Embed deltas.
        deltas_x = self.box_rpb_embed_x(deltas_x_log)
        deltas_y = self.box_rpb_embed_y(deltas_y_log)

        # Combine into 2D bias matrix.
        rpb_matrix = ops.add(
            ops.expand_dims(deltas_y, axis=3),
            ops.expand_dims(deltas_x, axis=2),
        )
        rpb_matrix = ops.reshape(
            rpb_matrix,
            (-1, self.num_queries, self.height * self.width, self.num_heads),
        )
        rpb_matrix = ops.transpose(rpb_matrix, (0, 3, 1, 2))
        return rpb_matrix

    def call(
        self,
        vision_features,
        text_features,
        vision_pos_encodings,
        text_masks,
        training=None,
    ):
        batch_size = ops.shape(vision_features)[0]
        query_embeds = ops.tile(
            ops.expand_dims(self.query_embed.embeddings, axis=0),
            [batch_size, 1, 1],
        )
        query_embeds = ops.cast(query_embeds, vision_features.dtype)
        reference_boxes = ops.tile(
            ops.expand_dims(self.reference_points.embeddings, axis=0),
            [batch_size, 1, 1],
        )
        reference_boxes = ops.cast(reference_boxes, vision_features.dtype)
        reference_boxes = ops.sigmoid(reference_boxes)
        presence_token = ops.tile(
            ops.expand_dims(self.presence_token.embeddings, axis=0),
            [batch_size, 1, 1],
        )
        presence_token = ops.cast(presence_token, vision_features.dtype)

        # Concatenate presence token with query embeddings
        hidden_states = ops.concatenate([presence_token, query_embeds], axis=1)
        text_cross_attn_masks = create_bidirectional_mask(
            hidden_states, text_masks
        )

        intermediate_outputs = []
        intermediate_boxes = [reference_boxes]
        intermediate_presence_logits = []
        for layer in self.layers:
            # Generate sine embeddings for conditional queries.
            reference_points_input = ops.expand_dims(reference_boxes, axis=2)
            query_sine_embed = self.position_encoding.encode_boxes(
                reference_points_input[:, :, 0, :]
            )
            query_pos = self.ref_point_head(query_sine_embed)

            # Compute box relative position bias (RPB) attention mask.
            rpb_matrix = self._get_rpb_matrix(reference_boxes)
            vision_cross_attn_masks = ops.pad(
                rpb_matrix, [[0, 0], [0, 0], [1, 0], [0, 0]]
            )

            hidden_states = layer(
                hidden_states,
                query_pos,
                text_features,
                vision_features,
                vision_pos_encodings,
                text_cross_attn_masks,
                vision_cross_attn_masks,
                training=training,
            )

            # Extract query hidden states (without presence token) for box
            # refinement.
            query_hidden_states = hidden_states[:, 1:]

            # Box refinement: predict delta and update reference boxes.
            reference_boxes_before_sigmoid = inverse_sigmoid(reference_boxes)
            output_hidden_states = self.output_layer_norm(
                query_hidden_states, training=training
            )
            delta_boxes = self.box_head(output_hidden_states, training=training)
            new_reference_boxes = ops.sigmoid(
                ops.add(delta_boxes, reference_boxes_before_sigmoid)
            )
            # For next layer.
            reference_boxes = ops.stop_gradient(new_reference_boxes)

            intermediate_outputs.append(output_hidden_states)
            intermediate_boxes.append(reference_boxes)

            # Process presence token.
            presence_hidden = hidden_states[:, :1]
            presence_logits = self.presence_head(
                self.presence_layer_norm(presence_hidden, training=training),
                training=training,
            )
            presence_logits = ops.squeeze(presence_logits, axis=-1)
            presence_logits = ops.clip(
                presence_logits,
                -self.clamp_presence_logit_max_val,
                self.clamp_presence_logit_max_val,
            )
            intermediate_presence_logits.append(presence_logits)

        # Stack outputs from all layers.
        intermediate_outputs = ops.stack(intermediate_outputs, axis=1)
        intermediate_boxes = ops.stack(intermediate_boxes[:-1], axis=1)
        intermediate_presence_logits = ops.stack(
            intermediate_presence_logits, axis=1
        )
        return (
            intermediate_outputs,
            intermediate_boxes,
            intermediate_presence_logits,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape,
                "patch_size": self.patch_size,
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "num_queries": self.num_queries,
                "hidden_activation": self.hidden_activation,
                "dropout_rate": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(
        self,
        vision_features_shape,
        text_features_shape,
        vision_pos_encodings_shape,
        text_masks_shape,
    ):
        batch_size = vision_features_shape[0]
        intermediate_output_shape = [
            batch_size,
            self.num_layers,
            self.num_queries,
            self.hidden_dim,
        ]
        intermediate_boxes_shape = [
            batch_size,
            self.num_layers,
            self.num_queries,
            4,
        ]
        intermediate_presence_logits_shape = [batch_size, self.num_layers, 1]
        return (
            intermediate_output_shape,
            intermediate_boxes_shape,
            intermediate_presence_logits_shape,
        )
