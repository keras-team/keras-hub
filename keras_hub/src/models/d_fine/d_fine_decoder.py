import math

import keras
import numpy as np

from keras_hub.src.models.d_fine.d_fine_attention import DFineMultiheadAttention
from keras_hub.src.models.d_fine.d_fine_attention import (
    DFineMultiscaleDeformableAttention,
)
from keras_hub.src.models.d_fine.d_fine_layers import DFineGate
from keras_hub.src.models.d_fine.d_fine_layers import DFineIntegral
from keras_hub.src.models.d_fine.d_fine_layers import DFineLQE
from keras_hub.src.models.d_fine.d_fine_layers import DFineMLP
from keras_hub.src.models.d_fine.d_fine_layers import DFineMLPPredictionHead
from keras_hub.src.models.d_fine.d_fine_utils import d_fine_kernel_initializer
from keras_hub.src.models.d_fine.d_fine_utils import distance2bbox
from keras_hub.src.models.d_fine.d_fine_utils import inverse_sigmoid
from keras_hub.src.models.d_fine.d_fine_utils import weighting_function
from keras_hub.src.utils.keras_utils import clone_initializer


class DFineDecoderLayer(keras.layers.Layer):
    """Single decoder layer for D-FINE models.

    This layer is the fundamental building block of the `DFineDecoder`. It
    refines a set of object queries by first allowing them to interact with
    each other via self-attention (`DFineMultiheadAttention`), and then
    attending to the image features from the encoder via cross-attention
    (`DFineMultiscaleDeformableAttention`). A feed-forward network with a
    gating mechanism (`DFineGate`) further processes the queries.

    Args:
        hidden_dim: int, Hidden dimension size for all attention and
            feed-forward layers.
        decoder_attention_heads: int, Number of attention heads for both
            self-attention and cross-attention mechanisms.
        attention_dropout: float, Dropout probability for attention weights.
        decoder_activation_function: str, Activation function name for the
            feed-forward network (e.g., `"relu"`, `"gelu"`, etc).
        dropout: float, General dropout probability applied to layer outputs.
        activation_dropout: float, Dropout probability applied after activation
            in the feed-forward network.
        layer_norm_eps: float, Epsilon value for layer normalization to prevent
            division by zero.
        decoder_ffn_dim: int, Hidden dimension size for the feed-forward
            network.
        num_feature_levels: int, Number of feature pyramid levels to attend to.
        decoder_offset_scale: float, Scaling factor for deformable attention
            offsets.
        decoder_method: str, Method used for deformable attention computation.
        decoder_n_points: int or list, Number of sampling points per feature
            level.
            If int, same number for all levels. If list, specific count per
            level.
        spatial_shapes: list, List of spatial dimensions `(height, width)`
            for each feature level.
        num_queries: int, Number of object queries processed by the decoder.
        kernel_initializer: str or Initializer, optional, Initializer for
            the kernel weights. Defaults to `"glorot_uniform"`.
        bias_initializer: str or Initializer, optional, Initializer for
            the bias weights. Defaults to `"zeros"`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        hidden_dim,
        decoder_attention_heads,
        attention_dropout,
        decoder_activation_function,
        dropout,
        activation_dropout,
        layer_norm_eps,
        decoder_ffn_dim,
        num_feature_levels,
        decoder_offset_scale,
        decoder_method,
        decoder_n_points,
        spatial_shapes,
        num_queries,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.decoder_attention_heads = decoder_attention_heads
        self.attention_dropout_rate = attention_dropout
        self.decoder_activation_function = decoder_activation_function
        self.layer_norm_eps = layer_norm_eps
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_feature_levels = num_feature_levels
        self.decoder_offset_scale = decoder_offset_scale
        self.decoder_method = decoder_method
        self.decoder_n_points = decoder_n_points
        self.spatial_shapes = spatial_shapes
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        self.self_attn = DFineMultiheadAttention(
            embedding_dim=self.hidden_dim,
            num_heads=self.decoder_attention_heads,
            dropout=self.attention_dropout_rate,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="self_attn",
        )
        self.dropout_layer = keras.layers.Dropout(
            rate=dropout, name="dropout_layer", dtype=self.dtype_policy
        )
        self.activation_dropout_layer = keras.layers.Dropout(
            rate=activation_dropout,
            name="activation_dropout_layer",
            dtype=self.dtype_policy,
        )
        self.activation_fn = keras.layers.Activation(
            self.decoder_activation_function,
            name="activation_fn",
            dtype=self.dtype_policy,
        )
        self.self_attn_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name="self_attn_layer_norm",
            dtype=self.dtype_policy,
        )
        self.encoder_attn = DFineMultiscaleDeformableAttention(
            hidden_dim=self.hidden_dim,
            decoder_attention_heads=self.decoder_attention_heads,
            num_feature_levels=self.num_feature_levels,
            decoder_offset_scale=self.decoder_offset_scale,
            dtype=self.dtype_policy,
            decoder_method=self.decoder_method,
            decoder_n_points=self.decoder_n_points,
            spatial_shapes=self.spatial_shapes,
            num_queries=self.num_queries,
            name="encoder_attn",
        )
        self.fc1 = keras.layers.Dense(
            self.decoder_ffn_dim,
            name="fc1",
            dtype=self.dtype_policy,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self.fc2 = keras.layers.Dense(
            self.hidden_dim,
            name="fc2",
            dtype=self.dtype_policy,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self.final_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name="final_layer_norm",
            dtype=self.dtype_policy,
        )
        self.gateway = DFineGate(
            self.hidden_dim, name="gateway", dtype=self.dtype_policy
        )

    def build(self, input_shape):
        batch_size = input_shape[0]
        num_queries = input_shape[1]
        hidden_dim = self.hidden_dim
        attention_input_shape = (batch_size, num_queries, hidden_dim)
        self.self_attn.build(attention_input_shape)
        self.encoder_attn.build(attention_input_shape)
        self.fc1.build(attention_input_shape)
        self.fc2.build((batch_size, num_queries, self.decoder_ffn_dim))
        self.gateway.build(attention_input_shape)
        self.self_attn_layer_norm.build(attention_input_shape)
        self.final_layer_norm.build(attention_input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        encoder_hidden_states=None,
        attention_mask=None,
        output_attentions=False,
        training=None,
    ):
        self_attn_output, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states_2 = self_attn_output
        hidden_states_2 = self.dropout_layer(hidden_states_2, training=training)
        hidden_states = hidden_states + hidden_states_2
        hidden_states = self.self_attn_layer_norm(
            hidden_states, training=training
        )
        residual = hidden_states
        query_for_cross_attn = residual
        if position_embeddings is not None:
            query_for_cross_attn = query_for_cross_attn + position_embeddings
        encoder_attn_output_tensor, cross_attn_weights_tensor = (
            self.encoder_attn(
                hidden_states=query_for_cross_attn,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                training=training,
            )
        )
        hidden_states_2 = encoder_attn_output_tensor
        current_cross_attn_weights = (
            cross_attn_weights_tensor if output_attentions else None
        )
        hidden_states_2 = self.dropout_layer(hidden_states_2, training=training)
        hidden_states = self.gateway(
            residual, hidden_states_2, training=training
        )
        hidden_states_ffn = self.fc1(hidden_states)
        hidden_states_2 = self.activation_fn(
            hidden_states_ffn, training=training
        )
        hidden_states_2 = self.activation_dropout_layer(
            hidden_states_2, training=training
        )
        hidden_states_2 = self.fc2(hidden_states_2)
        hidden_states_2 = self.dropout_layer(hidden_states_2, training=training)
        hidden_states = hidden_states + hidden_states_2
        dtype_name = keras.backend.standardize_dtype(self.compute_dtype)
        if dtype_name == "float16":
            clamp_value = np.finfo(np.float16).max - 1000.0
        else:  # float32, bfloat16
            clamp_value = np.finfo(np.float32).max - 1000.0
        hidden_states_clamped = keras.ops.clip(
            hidden_states, x_min=-clamp_value, x_max=clamp_value
        )
        hidden_states = self.final_layer_norm(
            hidden_states_clamped, training=training
        )
        return hidden_states, self_attn_weights, current_cross_attn_weights

    def compute_output_spec(
        self,
        hidden_states,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        encoder_hidden_states=None,
        attention_mask=None,
        output_attentions=False,
        training=None,
    ):
        hidden_states_output_spec = keras.KerasTensor(
            shape=hidden_states.shape, dtype=self.compute_dtype
        )
        self_attn_output_spec = self.self_attn.compute_output_spec(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        _, self_attn_weights_spec = self_attn_output_spec
        _, cross_attn_weights_spec = self.encoder_attn.compute_output_spec(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
        )
        if not output_attentions:
            self_attn_weights_spec = None
            cross_attn_weights_spec = None
        return (
            hidden_states_output_spec,
            self_attn_weights_spec,
            cross_attn_weights_spec,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "decoder_attention_heads": self.decoder_attention_heads,
                "attention_dropout": self.attention_dropout_rate,
                "decoder_activation_function": self.decoder_activation_function,
                "dropout": self.dropout_layer.rate,
                "activation_dropout": self.activation_dropout_layer.rate,
                "layer_norm_eps": self.layer_norm_eps,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "num_feature_levels": self.num_feature_levels,
                "decoder_offset_scale": self.decoder_offset_scale,
                "decoder_method": self.decoder_method,
                "decoder_n_points": self.decoder_n_points,
                "spatial_shapes": self.spatial_shapes,
                "num_queries": self.num_queries,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config


class DFineDecoder(keras.layers.Layer):
    """Complete decoder module for D-FINE object detection models.

    This class implements the full D-FINE decoder, which is responsible for
    transforming a set of object queries into final bounding box and class
    predictions. It consists of a stack of `DFineDecoderLayer` instances that
    iteratively refine the queries. At each layer, prediction heads
    (`class_embed`, `bbox_embed`) generate intermediate outputs, which are used
    for auxiliary loss calculation during training. The final layer's output
    represents the model's predictions.

    Args:
        eval_idx: int, Index of decoder layer used for evaluation. Negative
            values count from the end (e.g., -1 for last layer).
        num_decoder_layers: int, Number of decoder layers in the stack.
        dropout: float, General dropout probability applied throughout the
            decoder.
        hidden_dim: int, Hidden dimension size for all components.
        reg_scale: float, Scaling factor for regression loss and coordinate
            prediction.
        max_num_bins: int, Maximum number of bins for integral-based coordinate
            prediction.
        upsampling_factor: float, Upsampling factor used in coordinate
            prediction weighting.
        decoder_attention_heads: int, Number of attention heads in each decoder
            layer.
        attention_dropout: float, Dropout probability for attention mechanisms.
        decoder_activation_function: str, Activation function for feed-forward
            networks.
        activation_dropout: float, Dropout probability after activation
            functions.
        layer_norm_eps: float, Epsilon for layer normalization stability.
        decoder_ffn_dim: int, Hidden dimension for feed-forward networks.
        num_feature_levels: int, Number of feature pyramid levels.
        decoder_offset_scale: float, Scaling factor for deformable attention
            offsets.
        decoder_method: str, Method for deformable attention computation,
            either `"default"` or `"discrete"`.
        decoder_n_points: int or list, Number of sampling points per feature
            level.
        top_prob_values: int, Number of top probability values used in LQE.
        lqe_hidden_dim: int, Hidden dimension for LQE networks.
        num_lqe_layers: int, Number of layers in LQE networks.
        num_labels: int, Number of object classes for classification.
        spatial_shapes: list, Spatial dimensions for each feature level.
        layer_scale: float, Scaling factor for layer-wise feature dimensions.
        num_queries: int, Number of object queries processed by the decoder.
        initializer_bias_prior_prob: float, optional, Prior probability for
            the bias of the classification head. Used to initialize the bias
            of the `class_embed` layers. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        eval_idx,
        num_decoder_layers,
        dropout,
        hidden_dim,
        reg_scale,
        max_num_bins,
        upsampling_factor,
        decoder_attention_heads,
        attention_dropout,
        decoder_activation_function,
        activation_dropout,
        layer_norm_eps,
        decoder_ffn_dim,
        num_feature_levels,
        decoder_offset_scale,
        decoder_method,
        decoder_n_points,
        top_prob_values,
        lqe_hidden_dim,
        num_lqe_layers,
        num_labels,
        spatial_shapes,
        layer_scale,
        num_queries,
        initializer_bias_prior_prob=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.eval_idx = (
            eval_idx if eval_idx >= 0 else num_decoder_layers + eval_idx
        )
        self.dropout_rate = dropout
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        self.reg_scale_val = reg_scale
        self.max_num_bins = max_num_bins
        self.upsampling_factor = upsampling_factor
        self.decoder_attention_heads = decoder_attention_heads
        self.attention_dropout_rate = attention_dropout
        self.decoder_activation_function = decoder_activation_function
        self.activation_dropout_rate = activation_dropout
        self.layer_norm_eps = layer_norm_eps
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_feature_levels = num_feature_levels
        self.decoder_offset_scale = decoder_offset_scale
        self.decoder_method = decoder_method
        self.decoder_n_points = decoder_n_points
        self.top_prob_values = top_prob_values
        self.lqe_hidden_dim = lqe_hidden_dim
        self.num_lqe_layers = num_lqe_layers
        self.num_labels = num_labels
        self.spatial_shapes = spatial_shapes
        self.layer_scale = layer_scale
        self.initializer_bias_prior_prob = initializer_bias_prior_prob
        self.initializer = d_fine_kernel_initializer()
        self.decoder_layers = []
        for i in range(self.num_decoder_layers):
            self.decoder_layers.append(
                DFineDecoderLayer(
                    self.hidden_dim,
                    self.decoder_attention_heads,
                    self.attention_dropout_rate,
                    self.decoder_activation_function,
                    self.dropout_rate,
                    self.activation_dropout_rate,
                    self.layer_norm_eps,
                    self.decoder_ffn_dim,
                    self.num_feature_levels,
                    self.decoder_offset_scale,
                    self.decoder_method,
                    self.decoder_n_points,
                    self.spatial_shapes,
                    num_queries=self.num_queries,
                    kernel_initializer=clone_initializer(self.initializer),
                    bias_initializer="zeros",
                    dtype=self.dtype_policy,
                    name=f"decoder_layer_{i}",
                )
            )

        self.query_pos_head = DFineMLPPredictionHead(
            input_dim=4,
            hidden_dim=(2 * self.hidden_dim),
            output_dim=self.hidden_dim,
            num_layers=2,
            dtype=self.dtype_policy,
            kernel_initializer=clone_initializer(self.initializer),
            bias_initializer="zeros",
            name="query_pos_head",
        )

        num_pred = self.num_decoder_layers
        scaled_dim = round(self.hidden_dim * self.layer_scale)
        if initializer_bias_prior_prob is None:
            prior_prob = 1 / (self.num_labels + 1)
        else:
            prior_prob = initializer_bias_prior_prob
        class_embed_bias = float(-math.log((1 - prior_prob) / prior_prob))
        self.class_embed = [
            keras.layers.Dense(
                self.num_labels,
                name=f"class_embed_{i}",
                dtype=self.dtype_policy,
                kernel_initializer="glorot_uniform",
                bias_initializer=keras.initializers.Constant(class_embed_bias),
            )
            for i in range(num_pred)
        ]
        self.bbox_embed = [
            DFineMLPPredictionHead(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                output_dim=4 * (self.max_num_bins + 1),
                num_layers=3,
                name=f"bbox_embed_{i}",
                dtype=self.dtype_policy,
                kernel_initializer=clone_initializer(self.initializer),
                bias_initializer="zeros",
                last_layer_initializer="zeros",
            )
            for i in range(self.eval_idx + 1)
        ] + [
            DFineMLPPredictionHead(
                input_dim=scaled_dim,
                hidden_dim=scaled_dim,
                output_dim=4 * (self.max_num_bins + 1),
                num_layers=3,
                name=f"bbox_embed_{i + self.eval_idx + 1}",
                dtype=self.dtype_policy,
                kernel_initializer=clone_initializer(self.initializer),
                bias_initializer="zeros",
                last_layer_initializer="zeros",
            )
            for i in range(self.num_decoder_layers - self.eval_idx - 1)
        ]
        self.pre_bbox_head = DFineMLP(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=4,
            num_layers=3,
            activation_function="relu",
            dtype=self.dtype_policy,
            kernel_initializer=clone_initializer(self.initializer),
            bias_initializer="zeros",
            name="pre_bbox_head",
        )

        self.integral = DFineIntegral(
            max_num_bins=self.max_num_bins,
            name="integral",
            dtype=self.dtype_policy,
        )

        self.num_head = self.decoder_attention_heads

        self.lqe_layers = []
        for i in range(self.num_decoder_layers):
            self.lqe_layers.append(
                DFineLQE(
                    top_prob_values=self.top_prob_values,
                    max_num_bins=self.max_num_bins,
                    lqe_hidden_dim=self.lqe_hidden_dim,
                    num_lqe_layers=self.num_lqe_layers,
                    dtype=self.dtype_policy,
                    name=f"lqe_layer_{i}",
                )
            )

    def build(self, input_shape):
        if isinstance(input_shape, dict):
            if "inputs_embeds" not in input_shape:
                raise ValueError(
                    "DFineDecoder.build() received a dict input_shape "
                    "missing 'inputs_embeds' key. Please ensure 'inputs_embeds'"
                    " is passed correctly."
                )
            inputs_embeds_shape = input_shape["inputs_embeds"]
        elif (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) > 0
            and isinstance(input_shape[0], (list, tuple))
        ):
            inputs_embeds_shape = input_shape[0]
        else:
            inputs_embeds_shape = input_shape
        if not isinstance(inputs_embeds_shape, tuple):
            raise TypeError(
                f"Internal error: inputs_embeds_shape was expected to be a "
                f"tuple, but got {type(inputs_embeds_shape)} with value "
                f"{inputs_embeds_shape}. Original input_shape: {input_shape}"
            )

        batch_size_ph = (
            inputs_embeds_shape[0]
            if inputs_embeds_shape
            and len(inputs_embeds_shape) > 0
            and inputs_embeds_shape[0] is not None
            else None
        )
        num_queries_ph = (
            inputs_embeds_shape[1]
            if inputs_embeds_shape
            and len(inputs_embeds_shape) > 1
            and inputs_embeds_shape[1] is not None
            else None
        )
        current_decoder_layer_input_shape = inputs_embeds_shape
        for decoder_layer_instance in self.decoder_layers:
            decoder_layer_instance.build(current_decoder_layer_input_shape)
        qph_input_shape = (batch_size_ph, num_queries_ph, 4)
        self.query_pos_head.build(qph_input_shape)
        pre_bbox_head_input_shape = (
            batch_size_ph,
            num_queries_ph,
            self.hidden_dim,
        )
        self.pre_bbox_head.build(pre_bbox_head_input_shape)
        lqe_scores_shape = (batch_size_ph, num_queries_ph, 1)
        lqe_pred_corners_dim = 4 * (self.max_num_bins + 1)
        lqe_pred_corners_shape = (
            batch_size_ph,
            num_queries_ph,
            lqe_pred_corners_dim,
        )
        lqe_build_input_shape_tuple = (lqe_scores_shape, lqe_pred_corners_shape)
        for lqe_layer in self.lqe_layers:
            lqe_layer.build(lqe_build_input_shape_tuple)
        self.reg_scale = self.add_weight(
            name="reg_scale",
            shape=(1,),
            initializer=keras.initializers.Constant(self.reg_scale_val),
            trainable=False,
        )
        self.upsampling_factor = self.add_weight(
            name="upsampling_factor",
            shape=(1,),
            initializer=keras.initializers.Constant(self.upsampling_factor),
            trainable=False,
        )
        input_shape_for_class_embed = (
            batch_size_ph,
            num_queries_ph,
            self.hidden_dim,
        )
        for class_embed_layer in self.class_embed:
            class_embed_layer.build(input_shape_for_class_embed)
        input_shape_for_bbox_embed = (
            batch_size_ph,
            num_queries_ph,
            self.hidden_dim,
        )
        for bbox_embed_layer in self.bbox_embed:
            bbox_embed_layer.build(input_shape_for_bbox_embed)
        super().build(input_shape)

    def compute_output_spec(
        self,
        inputs_embeds,
        encoder_hidden_states,
        reference_points,
        spatial_shapes,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        training=None,
    ):
        output_attentions = (
            False if output_attentions is None else output_attentions
        )
        output_hidden_states = (
            False if output_hidden_states is None else output_hidden_states
        )
        batch_size = inputs_embeds.shape[0]
        num_queries = inputs_embeds.shape[1]
        hidden_dim = inputs_embeds.shape[2]
        last_hidden_state_spec = keras.KerasTensor(
            shape=(batch_size, num_queries, hidden_dim),
            dtype=self.compute_dtype,
        )
        intermediate_hidden_states_spec = None
        if output_hidden_states:
            intermediate_hidden_states_spec = keras.KerasTensor(
                shape=(
                    batch_size,
                    self.num_decoder_layers,
                    num_queries,
                    hidden_dim,
                ),
                dtype=self.compute_dtype,
            )
        num_layers_with_logits = self.num_decoder_layers + 1
        intermediate_logits_spec = keras.KerasTensor(
            shape=(
                batch_size,
                num_layers_with_logits,
                num_queries,
                self.num_labels,
            ),
            dtype=self.compute_dtype,
        )
        intermediate_reference_points_spec = keras.KerasTensor(
            shape=(batch_size, num_layers_with_logits, num_queries, 4),
            dtype=self.compute_dtype,
        )
        intermediate_predicted_corners_spec = keras.KerasTensor(
            shape=(
                batch_size,
                num_layers_with_logits,
                num_queries,
                4 * (self.max_num_bins + 1),
            ),
            dtype=self.compute_dtype,
        )
        initial_reference_points_spec = keras.KerasTensor(
            shape=(batch_size, num_layers_with_logits, num_queries, 4),
            dtype=self.compute_dtype,
        )
        all_hidden_states_spec = None
        all_self_attns_spec = None
        all_cross_attentions_spec = None
        if output_hidden_states:
            all_hidden_states_spec = tuple(
                [last_hidden_state_spec] * (self.num_decoder_layers + 1)
            )
        if output_attentions:
            (
                _,
                self_attn_spec,
                cross_attn_spec,
            ) = self.decoder_layers[0].compute_output_spec(
                hidden_states=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                output_attentions=True,
            )
            all_self_attns_spec = tuple(
                [self_attn_spec] * self.num_decoder_layers
            )
            if encoder_hidden_states is not None:
                all_cross_attentions_spec = tuple(
                    [cross_attn_spec] * self.num_decoder_layers
                )
        outputs_tuple = [
            last_hidden_state_spec,
            intermediate_hidden_states_spec,
            intermediate_logits_spec,
            intermediate_reference_points_spec,
            intermediate_predicted_corners_spec,
            initial_reference_points_spec,
            all_hidden_states_spec,
            all_self_attns_spec,
            all_cross_attentions_spec,
        ]
        return tuple(v for v in outputs_tuple if v is not None)

    def call(
        self,
        inputs_embeds,
        encoder_hidden_states,
        reference_points,
        spatial_shapes,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        training=None,
    ):
        output_attentions = (
            False if output_attentions is None else output_attentions
        )
        output_hidden_states = (
            False if output_hidden_states is None else output_hidden_states
        )

        hidden_states = inputs_embeds

        all_hidden_states = [] if output_hidden_states else None
        all_self_attns = [] if output_attentions else None
        all_cross_attentions = (
            []
            if (output_attentions and encoder_hidden_states is not None)
            else None
        )

        intermediate_hidden_states = []
        intermediate_reference_points = []
        intermediate_logits = []
        intermediate_predicted_corners = []
        initial_reference_points = []

        output_detach = (
            keras.ops.zeros_like(hidden_states)
            if hidden_states is not None
            else 0
        )
        pred_corners_undetach = 0

        project_flat = weighting_function(
            self.max_num_bins, self.upsampling_factor, self.reg_scale
        )
        project = keras.ops.expand_dims(project_flat, axis=0)

        ref_points_detach = keras.ops.sigmoid(reference_points)

        for i, decoder_layer_instance in enumerate(self.decoder_layers):
            ref_points_input = keras.ops.expand_dims(ref_points_detach, axis=2)
            query_pos_embed = self.query_pos_head(
                ref_points_detach, training=training
            )
            query_pos_embed = keras.ops.clip(query_pos_embed, -10.0, 10.0)

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            output_tuple = decoder_layer_instance(
                hidden_states=hidden_states,
                position_embeddings=query_pos_embed,
                reference_points=ref_points_input,
                spatial_shapes=spatial_shapes,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = output_tuple[0]
            self_attn_weights_from_layer = output_tuple[1]
            cross_attn_weights_from_layer = output_tuple[2]

            if i == 0:
                pre_bbox_head_output = self.pre_bbox_head(
                    hidden_states, training=training
                )
                new_reference_points = keras.ops.sigmoid(
                    pre_bbox_head_output + inverse_sigmoid(ref_points_detach)
                )
                ref_points_initial = keras.ops.stop_gradient(
                    new_reference_points
                )

            if self.bbox_embed is not None:
                bbox_embed_input = hidden_states + output_detach
                pred_corners = (
                    self.bbox_embed[i](bbox_embed_input, training=training)
                    + pred_corners_undetach
                )
                integral_output = self.integral(
                    pred_corners, project, training=training
                )
                inter_ref_bbox = distance2bbox(
                    ref_points_initial, integral_output, self.reg_scale
                )
                pred_corners_undetach = pred_corners
                ref_points_detach = keras.ops.stop_gradient(inter_ref_bbox)

            output_detach = keras.ops.stop_gradient(hidden_states)

            intermediate_hidden_states.append(hidden_states)

            if self.class_embed is not None and self.bbox_embed is not None:
                class_scores = self.class_embed[i](hidden_states)
                refined_scores = self.lqe_layers[i](
                    class_scores, pred_corners, training=training
                )
                if i == 0:
                    # NOTE: For first layer, output both, pre-LQE and post-LQE
                    # predictions, to provide an initial estimate. In the orig.
                    # implementation, the `torch.stack()` op would've thrown
                    # an error due to mismatched lengths.
                    intermediate_logits.append(class_scores)
                    intermediate_reference_points.append(new_reference_points)
                    initial_reference_points.append(ref_points_initial)
                    intermediate_predicted_corners.append(pred_corners)
                intermediate_logits.append(refined_scores)
                intermediate_reference_points.append(inter_ref_bbox)
                initial_reference_points.append(ref_points_initial)
                intermediate_predicted_corners.append(pred_corners)

            if output_attentions:
                if self_attn_weights_from_layer is not None:
                    all_self_attns.append(self_attn_weights_from_layer)
                if (
                    encoder_hidden_states is not None
                    and cross_attn_weights_from_layer is not None
                ):
                    all_cross_attentions.append(cross_attn_weights_from_layer)

        intermediate_stacked = (
            keras.ops.stack(intermediate_hidden_states, axis=1)
            if intermediate_hidden_states
            else None
        )

        if self.class_embed is not None and self.bbox_embed is not None:
            intermediate_logits_stacked = (
                keras.ops.stack(intermediate_logits, axis=1)
                if intermediate_logits
                else None
            )
            intermediate_predicted_corners_stacked = (
                keras.ops.stack(intermediate_predicted_corners, axis=1)
                if intermediate_predicted_corners
                else None
            )
            initial_reference_points_stacked = (
                keras.ops.stack(initial_reference_points, axis=1)
                if initial_reference_points
                else None
            )
            intermediate_reference_points_stacked = (
                keras.ops.stack(intermediate_reference_points, axis=1)
                if intermediate_reference_points
                else None
            )
        else:
            intermediate_logits_stacked = None
            intermediate_predicted_corners_stacked = None
            initial_reference_points_stacked = None
            intermediate_reference_points_stacked = None

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        all_hidden_states_tuple = (
            tuple(all_hidden_states) if output_hidden_states else None
        )
        all_self_attns_tuple = (
            tuple(all_self_attns) if output_attentions else None
        )
        all_cross_attentions_tuple = (
            tuple(all_cross_attentions)
            if (output_attentions and encoder_hidden_states is not None)
            else None
        )

        outputs_tuple = [
            hidden_states,
            intermediate_stacked,
            intermediate_logits_stacked,
            intermediate_reference_points_stacked,
            intermediate_predicted_corners_stacked,
            initial_reference_points_stacked,
            all_hidden_states_tuple,
            all_self_attns_tuple,
            all_cross_attentions_tuple,
        ]
        return tuple(v for v in outputs_tuple if v is not None)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "eval_idx": self.eval_idx,
                "num_decoder_layers": self.num_decoder_layers,
                "dropout": self.dropout_rate,
                "hidden_dim": self.hidden_dim,
                "reg_scale": self.reg_scale_val,
                "max_num_bins": self.max_num_bins,
                "upsampling_factor": self.upsampling_factor,
                "decoder_attention_heads": self.decoder_attention_heads,
                "attention_dropout": self.attention_dropout_rate,
                "decoder_activation_function": self.decoder_activation_function,
                "activation_dropout": self.activation_dropout_rate,
                "layer_norm_eps": self.layer_norm_eps,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "num_feature_levels": self.num_feature_levels,
                "decoder_offset_scale": self.decoder_offset_scale,
                "decoder_method": self.decoder_method,
                "decoder_n_points": self.decoder_n_points,
                "top_prob_values": self.top_prob_values,
                "lqe_hidden_dim": self.lqe_hidden_dim,
                "num_lqe_layers": self.num_lqe_layers,
                "num_labels": self.num_labels,
                "spatial_shapes": self.spatial_shapes,
                "layer_scale": self.layer_scale,
                "num_queries": self.num_queries,
                "initializer_bias_prior_prob": self.initializer_bias_prior_prob,
            }
        )
        return config
