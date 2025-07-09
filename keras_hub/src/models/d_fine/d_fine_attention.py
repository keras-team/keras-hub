import keras

from keras_hub.src.models.d_fine.d_fine_utils import (
    multi_scale_deformable_attention_v2,
)
from keras_hub.src.models.whisper.whisper_cached_multi_head_attention import (
    _build_proj_equation,
)


@keras.saving.register_keras_serializable(package="keras_hub")
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
        kernel_initializer: str or initializer, optional, Initializer for
            kernel weights. Defaults to `"glorot_uniform"`.
        spatial_shapes_list: list, optional, List of spatial shapes for
            different feature levels. Defaults to `None`.
        bias_initializer: str or initializer, optional, Initializer for
            bias weights. Defaults to `"zeros"`.
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
        kernel_initializer="glorot_uniform",
        spatial_shapes_list=None,
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.n_heads = decoder_attention_heads
        self.n_levels = num_feature_levels
        self.offset_scale = decoder_offset_scale
        self.decoder_method = decoder_method
        self.decoder_n_points = decoder_n_points
        self.spatial_shapes_list = spatial_shapes_list
        if isinstance(self.decoder_n_points, list):
            self.num_points_list = self.decoder_n_points
        else:
            self.num_points_list = [
                self.decoder_n_points for _ in range(self.n_levels)
            ]
        self._num_points_scale = [
            1.0 / n_points_at_level
            for n_points_at_level in self.num_points_list
            for _ in range(n_points_at_level)
        ]
        self.total_points = self.n_heads * sum(self.num_points_list)
        self.ms_deformable_attn_core = multi_scale_deformable_attention_v2
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        equation, bias_axes, _ = _build_proj_equation(
            free_dims=len(input_shape) - 1, bound_dims=1, output_dims=1
        )
        output_shape_sampling_offsets = (input_shape[1], self.total_points * 2)
        self.sampling_offsets = keras.layers.EinsumDense(
            equation,
            output_shape=output_shape_sampling_offsets,
            bias_axes=bias_axes,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="sampling_offsets",
        )
        self.sampling_offsets.build(input_shape)
        output_shape_attention_weights = (input_shape[1], self.total_points)
        self.attention_weights = keras.layers.EinsumDense(
            equation,
            output_shape=output_shape_attention_weights,
            bias_axes=bias_axes,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="attention_weights",
        )
        self.attention_weights.build(input_shape)
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
        _sampling_offsets = self.sampling_offsets(hidden_states)
        _sampling_offsets = keras.ops.reshape(
            _sampling_offsets,
            (
                batch_size,
                num_queries,
                self.n_heads,
                sum(self.num_points_list),
                2,
            ),
        )
        _attention_weights = self.attention_weights(hidden_states)
        _attention_weights = keras.ops.reshape(
            _attention_weights,
            (batch_size, num_queries, self.n_heads, sum(self.num_points_list)),
        )
        _attention_weights = keras.ops.softmax(_attention_weights, axis=-1)

        if keras.ops.shape(reference_points)[-1] == 2:
            offset_normalizer = keras.ops.cast(
                spatial_shapes, dtype=hidden_states.dtype
            )
            offset_normalizer = keras.ops.flip(offset_normalizer, axis=1)
            offset_normalizer = keras.ops.reshape(
                offset_normalizer, (1, 1, 1, self.n_levels, 1, 2)
            )
            _sampling_locations = (
                keras.ops.reshape(
                    reference_points,
                    (batch_size, num_queries, 1, self.n_levels, 1, 2),
                )
                + _sampling_offsets / offset_normalizer
            )
        elif keras.ops.shape(reference_points)[-1] == 4:
            _num_points_scale_t = keras.ops.cast(
                self.num_points_scale, dtype=hidden_states.dtype
            )
            _num_points_scale_t = keras.ops.expand_dims(
                _num_points_scale_t, axis=-1
            )
            offset = (
                _sampling_offsets
                * _num_points_scale_t
                * keras.ops.expand_dims(reference_points[..., 2:], axis=-2)
                * self.offset_scale
            )
            _sampling_locations = (
                keras.ops.expand_dims(reference_points[..., :2], axis=-2)
                + offset
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get "
                f"{keras.ops.shape(reference_points)[-1]} instead."
            )
        return _sampling_locations, _attention_weights

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
        _sampling_locations, _attention_weights = self.compute_attention(
            hidden_states, reference_points, spatial_shapes
        )

        # NOTE: slice_sizes_values passed down to ms_deformable_attn_core
        # since JAX tracing doesn't support dynamic shapes.
        slice_sizes = [h * w for h, w in self.spatial_shapes_list]
        output = self.ms_deformable_attn_core(
            value,
            spatial_shapes,
            _sampling_locations,
            _attention_weights,
            self.num_points_list,
            slice_sizes,
            self.spatial_shapes_list,
            self.n_levels,
            num_queries,
            self.decoder_method,
        )
        return output, _attention_weights

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
                "spatial_shapes_list": self.spatial_shapes_list,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
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
        embed_dim: int, Embedding dimension size.
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
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.bias = bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.dropout = keras.layers.Dropout(
            self.dropout, dtype=self.dtype_policy
        )

    def build(self, input_shape):
        embed_dim = self.embed_dim
        proj_equation, proj_bias_axes, _ = _build_proj_equation(
            free_dims=2, bound_dims=1, output_dims=2
        )
        proj_output_shape = (None, self.num_heads, self.head_dim)
        proj_input_shape = (None, None, embed_dim)
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
        out_proj_equation, out_proj_bias_axes, _ = _build_proj_equation(
            free_dims=2, bound_dims=1, output_dims=1
        )
        out_proj_input_shape = (None, None, self.num_heads * self.head_dim)
        out_proj_output_shape = (None, self.embed_dim)
        self.out_proj = keras.layers.EinsumDense(
            out_proj_equation,
            output_shape=out_proj_output_shape,
            bias_axes=out_proj_bias_axes if self.bias else None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer if self.bias else None,
            dtype=self.dtype_policy,
            name="out_proj",
        )
        self.out_proj.build(out_proj_input_shape)
        super().build(input_shape)

    def compute_attention(
        self,
        hidden_states,
        position_embeddings,
        hidden_states_original,
        attention_mask=None,
    ):
        def _with_pos_embed(tensor, position_embeddings_k):
            return (
                tensor
                if position_embeddings_k is None
                else tensor + position_embeddings_k
            )

        hidden_states_with_pos = _with_pos_embed(
            hidden_states, position_embeddings
        )
        query_states = self.q_proj(hidden_states_with_pos)
        key_states = self.k_proj(hidden_states_with_pos)
        value_states = self.v_proj(hidden_states_original)
        query_states = query_states * self.scaling
        batch_size = keras.ops.shape(query_states)[0]
        target_len = keras.ops.shape(query_states)[1]
        query_states_transposed = keras.ops.transpose(
            query_states, axes=(0, 2, 1, 3)
        )
        key_states_transposed = keras.ops.transpose(
            key_states, axes=(0, 2, 1, 3)
        )
        value_states_transposed = keras.ops.transpose(
            value_states, axes=(0, 2, 1, 3)
        )
        proj_shape_k = (batch_size * self.num_heads, target_len, self.head_dim)
        query_states_reshaped = keras.ops.reshape(
            query_states_transposed, proj_shape_k
        )
        key_states_reshaped = keras.ops.reshape(
            key_states_transposed, proj_shape_k
        )
        value_states_reshaped = keras.ops.reshape(
            value_states_transposed, proj_shape_k
        )
        attn_weights = keras.ops.matmul(
            query_states_reshaped,
            keras.ops.transpose(key_states_reshaped, axes=(0, 2, 1)),
        )
        if attention_mask is not None:
            source_len = keras.ops.shape(key_states_reshaped)[1]
            attn_weights = keras.ops.reshape(
                attn_weights,
                (
                    batch_size,
                    self.num_heads,
                    target_len,
                    source_len,
                ),
            )
            if keras.ops.ndim(attention_mask) == 2:
                attention_mask = keras.ops.expand_dims(attention_mask, axis=0)
            attention_mask = keras.ops.expand_dims(attention_mask, axis=1)
            attn_weights = attn_weights + attention_mask
            attn_weights = keras.ops.reshape(
                attn_weights,
                (batch_size * self.num_heads, target_len, source_len),
            )
        attn_weights = keras.ops.softmax(attn_weights, axis=-1)
        return (
            query_states_reshaped,
            key_states_reshaped,
            value_states_reshaped,
            attn_weights,
        )

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
        if position_embeddings is not None:
            hidden_states_original = hidden_states
        else:
            hidden_states_original = hidden_states
        _, key_states, value_states, attn_weights = self.compute_attention(
            hidden_states,
            position_embeddings,
            hidden_states_original,
            attention_mask,
        )
        source_len = keras.ops.shape(key_states)[1]
        attn_weights_for_output = attn_weights
        attn_probs = self.dropout(attn_weights, training=training)
        attn_output = keras.ops.matmul(attn_probs, value_states)
        attn_output = keras.ops.reshape(
            attn_output, (batch_size, self.num_heads, target_len, self.head_dim)
        )
        attn_output = keras.ops.transpose(attn_output, axes=(0, 2, 1, 3))
        attn_output = keras.ops.reshape(
            attn_output, (batch_size, target_len, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            attn_weights_reshaped_out = keras.ops.reshape(
                attn_weights_for_output,
                (batch_size, self.num_heads, target_len, source_len),
            )
            return attn_output, attn_weights_reshaped_out
        else:
            return attn_output, None

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        target_len = input_shape[1]
        source_len = input_shape[1]
        attn_output_shape = (batch_size, target_len, self.embed_dim)
        attn_weights_shape = (
            batch_size,
            self.num_heads,
            target_len,
            source_len,
        )
        return attn_output_shape, attn_weights_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
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
