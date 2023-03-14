# Copyright 2023 The KerasNLP Authors
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

"""T5 backbone model."""

import copy

import tensorflow as tf
from tensorflow import keras
from tensorflow.compiler.tf2xla.python.xla import dynamic_slice

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.backbone import Backbone
from keras_nlp.models.t5.t5_presets import backbone_presets
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.T5Backbone")
class T5Backbone(Backbone):
    def __init__(
        self,
        vocabulary_size,
        hidden_dim,
        intermediate_dim,
        num_blocks,
        num_heads,
        use_gated_activation,
        activation,
        dropout,
        layer_norm_epsilon,
        **kwargs,
    ):
        token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(1.0),
            name="token_embedding",
        )
        encoder = T5MainLayer(
            is_decoder=False,
            use_gated_activation=use_gated_activation,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_epsilon=layer_norm_epsilon,
            num_heads=num_heads,
            num_blocks=num_blocks,
            token_embeddings=token_embedding,
            name="encoder",
        )
        decoder = T5MainLayer(
            is_decoder=True,
            use_gated_activation=use_gated_activation,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_epsilon=layer_norm_epsilon,
            num_heads=num_heads,
            num_blocks=num_blocks,
            token_embeddings=token_embedding,
            name="decoder",
        )
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        decoder_token_ids = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        encoder_outputs = encoder(token_ids)
        decoder_outputs = decoder(
            decoder_token_ids,
            encoder_hidden_states=encoder_outputs,
        )
        super().__init__(
            {"token_ids": token_ids, "decoder_token_ids": decoder_token_ids},
            decoder_outputs,
            **kwargs,
        )
        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.use_gated_activation = use_gated_activation
        self.activation = activation
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_blocks": self.num_blocks,
                "num_heads": self.num_heads,
                "use_gated_activation": self.use_gated_activation,
                "activation": self.activation,
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
    
    @property
    def token_embedding(self):
        return self.get_layer("token_embedding")

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)


def shape_list(tensor):
    dynamic = tf.shape(tensor)
    if tensor.shape == tf.TensorShape(None):
        return dynamic
    static = tensor.shape.as_list()
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class T5LayerNorm(keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.weight = self.add_weight(
            "weight", shape=(input_shape[-1],), initializer="ones"
        )

    def call(self, hidden_states):
        variance = tf.math.reduce_mean(
            tf.math.square(hidden_states), axis=-1, keepdims=True
        )
        hidden_states = hidden_states * tf.math.rsqrt(variance + self.epsilon)
        return self.weight * hidden_states


class T5DenseBlock(keras.layers.Layer):
    def __init__(
        self,
        use_gated_activation,
        hidden_dim,
        intermediate_dim,
        dropout,
        activation,
        layer_norm_epsilon,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_gated_activation = use_gated_activation

        self.input_projector = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            name="input_projector",
            activation=keras.activations.get(activation),
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=hidden_dim**-0.5
            ),
        )
        if self.use_gated_activation:
            self.gate_projector = keras.layers.Dense(
                intermediate_dim,
                use_bias=False,
                name="gate_projector",
                kernel_initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=hidden_dim**-0.5
                ),
            )
        self.output_projector = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            name="output_projector",
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=intermediate_dim**-0.5
            ),
        )
        self.layer_norm = T5LayerNorm(epsilon=layer_norm_epsilon)
        self.dropout_layer = keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        hidden_states = self.layer_norm(inputs)
        if self.use_gated_activation:
            hidden_act = self.input_projector(hidden_states)
            hidden_linear = self.gate_projector(hidden_states)
            hidden_states = hidden_act * hidden_linear
        else:
            hidden_states = self.input_projector(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, training=training)
        hidden_states = self.output_projector(hidden_states)
        return inputs + self.dropout_layer(hidden_states, training=training)


class T5Attention(keras.layers.Layer):
    # This layer is adapted from Hugging Face
    # Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_tf_t5.py
    def __init__(
        self,
        is_decoder,
        hidden_dim,
        num_heads,
        dropout,
        use_relative_attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_decoder = is_decoder
        self.hidden_dim = hidden_dim
        self.key_value_dim = hidden_dim // num_heads
        self.num_heads = num_heads
        self.use_relative_attention_bias = use_relative_attention_bias

        self.inner_dim = self.num_heads * self.key_value_dim
        self.relative_attention_buckets = 32
        self.relative_attention_max_distance = 128

        self.query_projector = keras.layers.Dense(
            self.inner_dim,
            use_bias=False,
            name="query_projector",
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=(self.inner_dim * self.key_value_dim) ** -0.5
            ),
        )
        self.key_projector = keras.layers.Dense(
            self.inner_dim,
            use_bias=False,
            name="key_projector",
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=self.inner_dim**-0.5
            ),
        )
        self.value_projector = keras.layers.Dense(
            self.inner_dim,
            use_bias=False,
            name="value_projector",
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=self.inner_dim**-0.5
            ),
        )
        self.output_projector = keras.layers.Dense(
            self.hidden_dim,
            use_bias=False,
            name="output_projector",
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=self.inner_dim**-0.5
            ),
        )
        self.dropout_layer = keras.layers.Dropout(dropout)

    def build(self, input_shape):
        if self.use_relative_attention_bias:
            self.relative_attention_bias = self.add_weight(
                name="embeddings",
                shape=[self.relative_attention_buckets, self.num_heads],
                initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=self.inner_dim**-0.5
                ),
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """Adapted from Mesh Tensorflow.

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position,
        i.e. the distance in tokens from the attending position to the
        attended-to position. If bidirectional=False, then positive relative
        positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute
        relative_positions. All relative positions >=max_distance map to
        the same bucket. All relative positions <=-max_distance map to
        the same bucket. This should allow for more graceful generalization to
        longer sequences than the model has been trained on.

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            Tensor with the same shape as relative_position,
            containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                tf.cast(
                    tf.math.greater(relative_position, 0),
                    dtype=relative_position.dtype,
                )
                * num_buckets
            )
            relative_position = tf.math.abs(relative_position)
        else:
            relative_position = -tf.math.minimum(relative_position, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = tf.math.less(relative_position, max_exact)
        relative_position_if_large = max_exact + tf.cast(
            tf.math.log(
                tf.cast(relative_position, tf.float32)
                / tf.cast(max_exact, tf.float32)
            )
            / tf.math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        relative_position_if_large = tf.math.minimum(
            relative_position_if_large, num_buckets - 1
        )
        relative_buckets += tf.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = tf.range(query_length)[:, None]
        memory_position = tf.range(key_length)[None, :]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = tf.gather(
            self.relative_attention_bias, relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = tf.expand_dims(
            tf.transpose(values, [2, 0, 1]), axis=0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def call(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        training=False,
    ):
        # Input is (batch_size, query_length, dim)
        # Mask is (batch_size, key_length) (non-causal)
        # or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, num_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = shape_list(hidden_states)[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"Argument `past_key_value` should have 2 past states: "
                    f"keys and values. Got {len(past_key_value)} past states."
                )
            real_seq_length += (
                shape_list(past_key_value[0])[2]
                if query_length is None
                else query_length
            )

        key_length = (
            real_seq_length
            if key_value_states is None
            else shape_list(key_value_states)[1]
        )

        def shape(hidden_states):
            return tf.transpose(
                tf.reshape(
                    hidden_states,
                    (batch_size, -1, self.num_heads, self.key_value_dim),
                ),
                perm=(0, 2, 1, 3),
            )

        def unshape(hidden_states):
            return tf.reshape(
                tf.transpose(hidden_states, perm=(0, 2, 1, 3)),
                (batch_size, -1, self.inner_dim),
            )

        def project(
            hidden_states, proj_layer, key_value_states, past_key_value
        ):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attention
                # (batch_size, num_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attention
                # (batch_size, num_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attention
                    # (batch_size, num_heads, key_length, dim_per_head)
                    hidden_states = tf.concat(
                        [past_key_value, hidden_states], axis=2
                    )
                else:
                    # cross-attention
                    hidden_states = past_key_value
            return hidden_states

        # get query
        query_states = shape(
            self.query_projector(hidden_states)
        )  # (batch_size, num_heads, query_length, dim_per_head)

        # get key/value
        key_states = project(
            hidden_states,
            self.key_projector,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.value_projector,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        scores = tf.einsum(
            "bnqd,bnkd->bnqk", query_states, key_states
        )  # (batch_size, num_heads, query_length, key_length)

        if position_bias is None:
            if not self.use_relative_attention_bias:
                position_bias = tf.zeros(
                    (1, self.num_heads, real_seq_length, key_length)
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated we want only
            # the last query position bias
            if past_key_value is not None:
                if not self.use_relative_attention_bias:
                    position_bias = position_bias[:, :, -seq_length:, :]
                else:
                    # we might have a padded past structure,
                    # in which case we want to fetch the position bias slice
                    # right after the most recently filled past index
                    most_recently_filled_past_index = tf.reduce_max(
                        tf.where(past_key_value[0][0, 0, :, 0] != 0.0)
                    )
                    position_bias = dynamic_slice(
                        position_bias,
                        (0, 0, most_recently_filled_past_index + 1, 0),
                        (1, self.num_heads, seq_length, real_seq_length),
                    )

            if mask is not None:
                position_bias = tf.cast(position_bias, dtype=mask.dtype)
                position_bias = (
                    position_bias + mask
                )  # (batch_size, num_heads, query_length, key_length)

        scores += position_bias
        weights = tf.nn.softmax(
            scores, axis=-1
        )  # (batch_size, num_heads, query_length, key_length)
        weights = self.dropout_layer(
            weights, training=training
        )  # (batch_size, num_heads, query_length, key_length)

        # Opitonally mask heads
        if layer_head_mask is not None:
            weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * weights

        attention_output = tf.matmul(
            weights, value_states
        )  # (batch_size, num_heads, query_length, dim_per_head)

        attention_output = self.output_projector(unshape(attention_output))
        return (attention_output, position_bias)


class T5AttentionBlock(keras.layers.Layer):
    def __init__(
        self,
        is_decoder,
        hidden_dim,
        num_heads,
        dropout,
        layer_norm_epsilon,
        use_relative_attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention = T5Attention(
            is_decoder,
            hidden_dim,
            num_heads,
            dropout,
            use_relative_attention_bias=use_relative_attention_bias,
        )
        self.layer_norm = T5LayerNorm(epsilon=layer_norm_epsilon)
        self.dropout_layer = keras.layers.Dropout(dropout)

    def call(
        self,
        hidden_states,
        key_value_states=None,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        query_length=None,
        training=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias = self.attention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            query_length=query_length,
            training=training,
        )
        hidden_states = hidden_states + self.dropout_layer(
            attention_output, training=training
        )
        return (hidden_states, position_bias)


class T5TransformerDecoder(keras.layers.Layer):
    # This layer is adapted from Hugging Face
    # Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_tf_t5.py
    def __init__(
        self,
        is_decoder,
        use_gated_activation,
        hidden_dim,
        intermediate_dim,
        dropout,
        activation,
        layer_norm_epsilon,        
        num_heads,
        use_relative_attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_decoder = is_decoder

        self.self_attention = T5AttentionBlock(
            is_decoder,
            hidden_dim,
            num_heads,
            dropout,
            layer_norm_epsilon,
            use_relative_attention_bias=use_relative_attention_bias,
        )
        if self.is_decoder:
            self.cross_attention = T5AttentionBlock(
                is_decoder,
                hidden_dim,
                num_heads,
                dropout,
                layer_norm_epsilon,
            )
        self.dense_block = T5DenseBlock(
            use_gated_activation,
            hidden_dim,
            intermediate_dim,
            dropout,
            activation,
            layer_norm_epsilon,
        )

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        encoder_layer_head_mask=None,
        past_key_value=None,
        training=False,
    ):
        if past_key_value is not None:
            self_attention_past_key_value = past_key_value[:2]
            cross_attention_past_key_value = past_key_value[2:]
        else:
            self_attention_past_key_value = None
            cross_attention_past_key_value = None

        hidden_states, position_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attention_past_key_value,
            training=training,
        )

        encoder_decoder_position_bias = None
        if self.is_decoder and encoder_hidden_states is not None:
            hidden_states, encoder_decoder_position_bias = self.cross_attention(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attention_past_key_value,
                query_length=None,
                training=training,
            )

        hidden_states = self.dense_block(hidden_states, training=training)
        return {
            "hidden_states": hidden_states,
            "position_bias": position_bias,
            "encoder_decoder_position_bias": encoder_decoder_position_bias,
        }


class T5MainLayer(keras.layers.Layer):
    # This layer is adapted from Hugging Face
    # Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_tf_t5.py
    def __init__(
        self,
        is_decoder,
        use_gated_activation,
        hidden_dim,
        intermediate_dim,
        dropout,
        activation,
        layer_norm_epsilon,
        num_heads,
        num_blocks,
        token_embeddings=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_embeddings = token_embeddings
        self.is_decoder = is_decoder
        self.num_hidden_layers = num_blocks

        self.blocks = [
            T5TransformerDecoder(
                is_decoder=is_decoder,
                use_gated_activation=use_gated_activation,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_epsilon=layer_norm_epsilon,
                num_heads=num_heads,
                use_relative_attention_bias=bool(i == 0),
            )
            for i in range(num_blocks)
        ]
        self.final_layer_norm = T5LayerNorm(
            epsilon=layer_norm_epsilon, name="final_layer_norm"
        )
        self.dropout_layer = keras.layers.Dropout(dropout)

    def call(
        self,
        token_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeddings=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        training=False,
    ):
        if token_ids is not None and inputs_embeddings is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}token_ids and "
                f"{err_msg_prefix}inputs_embeddings at the same time."
            )
        elif token_ids is not None:
            input_shape = shape_list(token_ids)
            token_ids = tf.reshape(token_ids, (-1, input_shape[-1]))
        elif inputs_embeddings is not None:
            input_shape = shape_list(inputs_embeddings)[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either `{err_msg_prefix}token_ids` "
                f"or `{err_msg_prefix}inputs_embeddings`."
            )

        if inputs_embeddings is None:
            inputs_embeddings = self.token_embeddings(token_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            shape_list(past_key_values[0][0])[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if attention_mask is None:
            attention_mask = tf.fill((batch_size, mask_seq_length), 1)
        if (
            self.is_decoder
            and encoder_attention_mask is None
            and encoder_hidden_states is not None
        ):
            encoder_seq_length = shape_list(encoder_hidden_states)[1]
            encoder_attention_mask = tf.fill(
                (batch_size, encoder_seq_length), 1
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length] ourselves
        # in which case we just need to make it broadcastable to all heads.
        attention_mask = tf.cast(attention_mask, dtype=inputs_embeddings.dtype)
        num_dims_attention_mask = len(shape_list(attention_mask))
        if num_dims_attention_mask == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif num_dims_attention_mask == 2:
            # Provided a padding mask of dimensions [batch_size, mask_seq_length]
            # - if decoder, apply a causal mask in addition to the padding mask
            # - if encoder, make the mask broadcastable to
            # [batch_size, num_heads, mask_seq_length, mask_seq_length]
            if self.is_decoder:
                seq_ids = tf.range(mask_seq_length)
                causal_mask = tf.less_equal(
                    tf.tile(
                        seq_ids[None, None, :], (batch_size, mask_seq_length, 1)
                    ),
                    seq_ids[None, :, None],
                )
                causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
                extended_attention_mask = (
                    causal_mask[:, None, :, :]
                    * attention_mask[:, None, None, :]
                )
                if past_key_values[0] is not None:
                    extended_attention_mask = extended_attention_mask[
                        :, :, -seq_length:, :
                    ]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
        if self.is_decoder and encoder_attention_mask is not None:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to
            # [batch_size, num_heads, mask_seq_length, mask_seq_length]
            encoder_attention_mask = tf.cast(
                encoder_attention_mask, dtype=extended_attention_mask.dtype
            )
            num_dims_encoder_attention_mask = len(
                shape_list(encoder_attention_mask)
            )
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[
                    :, None, :, :
                ]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[
                    :, None, None, :
                ]

            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask
            ) * -1e9
        else:
            encoder_extended_attention_mask = None

        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.dropout_layer(inputs_embeddings, training=training)

        for idx, (block, past_key_value) in enumerate(
            zip(self.blocks, past_key_values)
        ):
            layer_head_mask = head_mask[idx] if head_mask is not None else None
            encoder_layer_head_mask = (
                encoder_head_mask[idx]
                if encoder_head_mask is not None
                else None
            )
            layer_outputs = block(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                encoder_layer_head_mask=encoder_layer_head_mask,
                past_key_value=past_key_value,
                training=training,
            )
            hidden_states = layer_outputs["hidden_states"]
            position_bias = layer_outputs["position_bias"]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    "encoder_decoder_position_bias"
                ]

        hidden_states = self.final_layer_norm(hidden_states)
        return self.dropout_layer(hidden_states, training=training)
