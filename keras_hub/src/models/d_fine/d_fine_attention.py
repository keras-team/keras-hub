import math

import keras

from keras_hub.src.models.d_fine.d_fine_utils import (
    multi_scale_deformable_attention_v2,
)


class DFineMultiscaleDeformableAttention(keras.layers.Layer):
    """Multi-scale deformable attention layer for D-FINE models.

    This layer implements the multi-scale deformable attention mechanism, which
    is the core of the cross-attention in each `DFineDecoderLayer`. It allows
    the model to attend to a small set of key sampling points around a reference
    point across multiple feature levels from the encoder.

    The layer computes sampling locations and attention weights based on the
    input queries, enabling the model to focus on relevant features across
    multiple feature levels and spatial positions.

    Args:
        hidden_dim: int, Hidden dimension size for the attention mechanism.
        decoder_attention_heads: int, Number of attention heads.
        num_feature_levels: int, Number of feature levels to attend to.
        decoder_offset_scale: float, Scaling factor for sampling offsets.
        decoder_method: str, Method used for deformable attention computation.
        decoder_n_points: int or list, Number of sampling points per level.
            If int, the same number of points is used for all levels.
            If list, specifies points for each level individually.
        num_queries: int, Number of queries in the attention mechanism.
        spatial_shapes: list, List of spatial shapes for different
            feature levels.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        hidden_dim,
        decoder_attention_heads,
        num_feature_levels,
        decoder_offset_scale,
        decoder_method,
        decoder_n_points,
        num_queries,
        spatial_shapes,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.n_heads = decoder_attention_heads
        self.n_levels = num_feature_levels
        self.offset_scale = decoder_offset_scale
        self.decoder_method = decoder_method
        self.decoder_n_points = decoder_n_points
        self.spatial_shapes = spatial_shapes
        if isinstance(self.decoder_n_points, list):
            self.num_points = self.decoder_n_points
        else:
            self.num_points = [
                self.decoder_n_points for _ in range(self.n_levels)
            ]
        self._num_points_scale = [
            1.0 / n_points_at_level
            for n_points_at_level in self.num_points
            for _ in range(n_points_at_level)
        ]
        self.total_points = self.n_heads * sum(self.num_points)
        self.ms_deformable_attn_core = multi_scale_deformable_attention_v2

    def build(self, input_shape):
        sampling_offsets_output_shape = (
            input_shape[1],
            self.n_heads,
            sum(self.num_points),
            2,
        )
        self.sampling_offsets = keras.layers.EinsumDense(
            "abc,cdef->abdef",
            output_shape=sampling_offsets_output_shape,
            bias_axes="def",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="sampling_offsets",
            dtype=self.dtype_policy,
        )
        self.sampling_offsets.build(input_shape)
        attention_weights_output_shape = (
            input_shape[1],
            self.n_heads,
            sum(self.num_points),
        )
        self.attention_weights = keras.layers.EinsumDense(
            "abc,cde->abde",
            output_shape=attention_weights_output_shape,
            bias_axes="de",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="attention_weights",
            dtype=self.dtype_policy,
        )
        self.attention_weights.build(input_shape)
        if self.sampling_offsets.bias is not None:
            thetas = keras.ops.arange(
                self.n_heads, dtype=self.variable_dtype
            ) * (2.0 * math.pi / self.n_heads)
            grid_init = keras.ops.stack(
                [keras.ops.cos(thetas), keras.ops.sin(thetas)], axis=-1
            )
            grid_init = grid_init / keras.ops.max(
                keras.ops.abs(grid_init), axis=-1, keepdims=True
            )
            grid_init = keras.ops.reshape(grid_init, (self.n_heads, 1, 2))
            grid_init = keras.ops.tile(grid_init, [1, sum(self.num_points), 1])
            scaling = []
            for n in self.num_points:
                scaling.append(
                    keras.ops.arange(1, n + 1, dtype=self.variable_dtype)
                )
            scaling = keras.ops.concatenate(scaling, axis=0)
            scaling = keras.ops.reshape(scaling, (1, -1, 1))
            grid_init *= scaling
            self.sampling_offsets.bias.assign(grid_init)
        self.num_points_scale = self.add_weight(
            name="num_points_scale",
            shape=(len(self._num_points_scale),),
            initializer=keras.initializers.Constant(self._num_points_scale),
            trainable=False,
        )
        super().build(input_shape)

    def compute_attention(
        self, hidden_states, reference_points, spatial_shapes
    ):
        batch_size = keras.ops.shape(hidden_states)[0]
        num_queries = keras.ops.shape(hidden_states)[1]
        sampling_offsets = self.sampling_offsets(hidden_states)
        attention_weights = self.attention_weights(hidden_states)
        attention_weights = keras.ops.softmax(attention_weights, axis=-1)

        if keras.ops.shape(reference_points)[-1] == 2:
            offset_normalizer = keras.ops.cast(
                spatial_shapes, dtype=hidden_states.dtype
            )
            offset_normalizer = keras.ops.flip(offset_normalizer, axis=1)
            offset_normalizer = keras.ops.reshape(
                offset_normalizer, (1, 1, 1, self.n_levels, 1, 2)
            )
            sampling_locations = (
                keras.ops.reshape(
                    reference_points,
                    (batch_size, num_queries, 1, self.n_levels, 1, 2),
                )
                + sampling_offsets / offset_normalizer
            )
        elif keras.ops.shape(reference_points)[-1] == 4:
            num_points_scale_t = keras.ops.cast(
                self.num_points_scale, dtype=hidden_states.dtype
            )
            num_points_scale_t = keras.ops.expand_dims(
                num_points_scale_t, axis=-1
            )
            offset = (
                sampling_offsets
                * num_points_scale_t
                * keras.ops.expand_dims(reference_points[..., 2:], axis=-2)
                * self.offset_scale
            )
            sampling_locations = (
                keras.ops.expand_dims(reference_points[..., :2], axis=-2)
                + offset
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get "
                f"{keras.ops.shape(reference_points)[-1]} instead."
            )
        return sampling_locations, attention_weights

    def call(
        self,
        hidden_states,
        encoder_hidden_states,
        reference_points,
        spatial_shapes,
    ):
        batch_size = keras.ops.shape(hidden_states)[0]
        num_queries = keras.ops.shape(hidden_states)[1]
        sequence_length = keras.ops.shape(encoder_hidden_states)[1]
        value = keras.ops.reshape(
            encoder_hidden_states,
            (
                batch_size,
                sequence_length,
                self.n_heads,
                self.hidden_dim // self.n_heads,
            ),
        )
        sampling_locations, attention_weights = self.compute_attention(
            hidden_states, reference_points, spatial_shapes
        )

        # NOTE: slice_sizes_values passed down to ms_deformable_attn_core
        # since JAX tracing doesn't support dynamic shapes.
        slice_sizes = [h * w for h, w in self.spatial_shapes]
        output = self.ms_deformable_attn_core(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points,
            slice_sizes,
            self.spatial_shapes,
            self.n_levels,
            num_queries,
            self.decoder_method,
        )
        return output, attention_weights

    def compute_output_spec(
        self,
        hidden_states,
        encoder_hidden_states,
        reference_points,
        spatial_shapes,
    ):
        input_shape = hidden_states.shape
        batch_size = input_shape[0] if len(input_shape) > 0 else None
        num_queries = input_shape[1] if len(input_shape) > 1 else None
        output_shape = (batch_size, num_queries, self.hidden_dim)
        output_spec = keras.KerasTensor(output_shape, dtype=self.compute_dtype)
        attention_weights_shape = (
            batch_size,
            num_queries,
            self.n_heads,
            sum(self.num_points),
        )
        attention_weights_spec = keras.KerasTensor(
            attention_weights_shape, dtype=self.compute_dtype
        )
        return output_spec, attention_weights_spec

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "decoder_attention_heads": self.n_heads,
                "num_feature_levels": self.n_levels,
                "decoder_offset_scale": self.offset_scale,
                "decoder_method": self.decoder_method,
                "decoder_n_points": self.decoder_n_points,
                "num_queries": self.num_queries,
                "spatial_shapes": self.spatial_shapes,
            }
        )
        return config


class DFineMultiheadAttention(keras.layers.Layer):
    """Multi-head attention layer for D-FINE models.

    This layer implements a standard multi-head attention mechanism. It is used
    in two key places within the D-FINE architecture:
    1.  In `DFineEncoderLayer` as the self-attention mechanism to process the
        sequence of image features from the `HGNetV2Backbone` class.
    2.  In `DFineDecoderLayer` as the self-attention mechanism to allow object
        queries to interact with each other.

    It supports position embeddings to incorporate positional information and
    attention masking to prevent attending to certain positions.

    Args:
        embedding_dim: int, Embedding dimension size.
        num_heads: int, Number of attention heads.
        dropout: float, optional, Dropout probability for attention weights.
            Defaults to `0.0`.
        bias: bool, optional, Whether to include bias in projection layers.
            Defaults to `True`.
        kernel_initializer: str or initializer, optional, Initializer for
            kernel weights. Defaults to `"glorot_uniform"`.
        bias_initializer: str or initializer, optional, Initializer for
            bias weights. Defaults to `"zeros"`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.head_dim = embedding_dim // num_heads
        if self.head_dim * self.num_heads != self.embedding_dim:
            raise ValueError(
                f"embedding_dim must be divisible by num_heads (got "
                f"`embedding_dim`: {self.embedding_dim} and `num_heads`: "
                f"{self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.bias = bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.dropout = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy
        )

    def build(self, input_shape):
        embedding_dim = self.embedding_dim
        proj_equation = "abc,cde->abde"
        proj_bias_axes = "de"
        proj_output_shape = (None, self.num_heads, self.head_dim)
        proj_input_shape = (None, None, embedding_dim)
        self.q_proj = keras.layers.EinsumDense(
            proj_equation,
            output_shape=proj_output_shape,
            bias_axes=proj_bias_axes if self.bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer if self.bias else None,
            dtype=self.dtype_policy,
            name="q_proj",
        )
        self.q_proj.build(proj_input_shape)
        self.k_proj = keras.layers.EinsumDense(
            proj_equation,
            output_shape=proj_output_shape,
            bias_axes=proj_bias_axes if self.bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer if self.bias else None,
            dtype=self.dtype_policy,
            name="k_proj",
        )
        self.k_proj.build(proj_input_shape)
        self.v_proj = keras.layers.EinsumDense(
            proj_equation,
            output_shape=proj_output_shape,
            bias_axes=proj_bias_axes if self.bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer if self.bias else None,
            dtype=self.dtype_policy,
            name="v_proj",
        )
        self.v_proj.build(proj_input_shape)
        out_proj_input_shape = (None, None, self.num_heads * self.head_dim)
        out_proj_output_shape = (None, self.embedding_dim)
        self.out_proj = keras.layers.EinsumDense(
            "abc,cd->abd",
            output_shape=out_proj_output_shape,
            bias_axes="d" if self.bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer if self.bias else None,
            dtype=self.dtype_policy,
            name="out_proj",
        )
        self.out_proj.build(out_proj_input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        output_attentions=False,
        training=None,
    ):
        batch_size = keras.ops.shape(hidden_states)[0]
        target_len = keras.ops.shape(hidden_states)[1]

        def with_pos_embed(tensor, position_embeddings_k):
            return (
                tensor
                if position_embeddings_k is None
                else tensor + position_embeddings_k
            )

        hidden_states_with_pos = with_pos_embed(
            hidden_states, position_embeddings
        )
        query_states = self.q_proj(hidden_states_with_pos)
        key_states = self.k_proj(hidden_states_with_pos)
        value_states = self.v_proj(hidden_states)
        attn_weights = keras.ops.einsum(
            "bthd,bshd->bhts", query_states * self.scaling, key_states
        )
        if attention_mask is not None:
            if keras.ops.ndim(attention_mask) == 2:
                attention_mask = keras.ops.expand_dims(attention_mask, axis=0)
            attention_mask = keras.ops.expand_dims(attention_mask, axis=1)
            attn_weights = attn_weights + attention_mask
        attn_weights = keras.ops.softmax(attn_weights, axis=-1)
        attn_weights_for_output = attn_weights if output_attentions else None
        attn_probs = self.dropout(attn_weights, training=training)
        attn_output = keras.ops.einsum(
            "bhts,bshd->bthd", attn_probs, value_states
        )
        attn_output = keras.ops.reshape(
            attn_output, (batch_size, target_len, self.embedding_dim)
        )
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            return attn_output, attn_weights_for_output
        else:
            return attn_output

    def compute_output_spec(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        output_attentions=False,
        training=None,
    ):
        input_shape = hidden_states.shape
        batch_size = input_shape[0] if len(input_shape) > 0 else None
        target_len = input_shape[1] if len(input_shape) > 1 else None
        source_len = target_len
        attn_output_shape = (batch_size, target_len, self.embedding_dim)
        attn_output_spec = keras.KerasTensor(
            attn_output_shape, dtype=self.compute_dtype
        )
        if output_attentions:
            attn_weights_shape = (
                batch_size,
                self.num_heads,
                target_len,
                source_len,
            )
            attn_weights_spec = keras.KerasTensor(
                attn_weights_shape, dtype=self.compute_dtype
            )
            return attn_output_spec, attn_weights_spec
        return attn_output_spec

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout_rate,
                "bias": self.bias,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config
