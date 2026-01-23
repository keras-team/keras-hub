import math

from keras import backend
from keras import config
from keras import initializers
from keras import layers
from keras import ops

from keras_hub.src.models.sam3.sam3_utils import box_cxcywh_to_xyxy
from keras_hub.src.models.sam3.sam3_utils import inverse_sigmoid
from keras_hub.src.utils.keras_utils import standardize_data_format


class SAM3MLP(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        activation="gelu",
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.activation = activation
        self.dropout_rate = float(dropout_rate)

        self.fc1 = layers.Dense(
            intermediate_dim, dtype=self.dtype_policy, name="fc1"
        )
        self.act = layers.Activation(activation, dtype=self.dtype_policy)
        self.fc2 = layers.Dense(hidden_dim, dtype=self.dtype_policy, name="fc2")
        self.dropout = layers.Dropout(
            dropout_rate, dtype=self.dtype_policy, name="dropout"
        )

    def build(self, input_shape):
        self.fc1.build(input_shape)
        input_shape = self.fc1.compute_output_shape(input_shape)
        self.dropout.build(input_shape)
        self.act.build(input_shape)
        self.fc2.build(input_shape)
        input_shape = self.fc2.compute_output_shape(input_shape)

    def call(self, inputs, training=None):
        x = self.fc1(inputs, training=training)
        x = self.dropout(x, training=training)
        x = self.act(x)
        return self.fc2(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class SAM3Attention(layers.Layer):
    def __init__(self, hidden_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="q_proj"
        )
        self.k_proj = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="k_proj"
        )
        self.v_proj = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="v_proj"
        )
        self.o_proj = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="o_proj"
        )

    def build(self, query_shape, key_shape, value_shape):
        self.q_proj.build(query_shape)
        self.k_proj.build(key_shape)
        self.v_proj.build(value_shape)
        self.o_proj.build(value_shape)

    def call(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attention_bias=None,
        training=None,
    ):
        batch_size = ops.shape(query)[0]

        query = self.q_proj(query, training=training)
        query = ops.reshape(
            query, (batch_size, -1, self.num_heads, self.head_dim)
        )
        key = self.k_proj(key, training=training)
        key = ops.reshape(key, (batch_size, -1, self.num_heads, self.head_dim))
        value = self.v_proj(value, training=training)
        value = ops.reshape(
            value, (batch_size, -1, self.num_heads, self.head_dim)
        )

        if (
            backend.backend() == "torch"
            and attention_mask is None
            and attention_bias is not None
        ):
            # TODO: Torch backend doesn't support attention_bias in
            # ops.dot_product_attention yet.
            # Fixed by https://github.com/keras-team/keras/pull/22045
            import torch

            query = torch.transpose(query, 1, 2).contiguous()
            key = torch.transpose(key, 1, 2).contiguous()
            value = torch.transpose(value, 1, 2).contiguous()
            attention_bias = attention_bias.contiguous()
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_bias,
                is_causal=False,
                scale=self.scale,
            )
            attn_output = torch.transpose(attn_output, 2, 1)
        else:
            if attention_mask is not None:
                attention_mask = ops.cast(attention_mask, dtype="bool")
            attn_output = ops.dot_product_attention(
                query,
                key,
                value,
                bias=attention_bias,
                mask=attention_mask,
                scale=self.scale,
                is_causal=False,
            )
        attn_output = ops.reshape(
            attn_output, (batch_size, -1, self.num_heads * self.head_dim)
        )
        return self.o_proj(attn_output, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class SAM3RoPEAttention(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        attention_dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="q_proj"
        )
        self.k_proj = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="k_proj"
        )
        self.v_proj = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="v_proj"
        )
        self.o_proj = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="o_proj"
        )

    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.o_proj.build(input_shape)

    def apply_rotary_pos_emb_2d(self, query, key, cos, sin):
        def rotate_pairwise(x):
            x = ops.reshape(
                x,
                (
                    -1,
                    self.num_heads,
                    self.height * self.width,
                    self.head_dim // 2,
                    2,
                ),
            )
            x1 = x[..., 0]
            x2 = x[..., 1]
            x = ops.stack((-x2, x1), axis=-1)
            return ops.reshape(
                x, (-1, self.num_heads, self.height * self.width, self.head_dim)
            )

        query = ops.transpose(query, axes=(0, 2, 1, 3))
        key = ops.transpose(key, axes=(0, 2, 1, 3))

        original_dtype = backend.standardize_dtype(query.dtype)
        query_embed = ops.cast(query, dtype="float32")
        query_embed = ops.add(
            ops.multiply(query_embed, cos),
            ops.multiply(rotate_pairwise(query_embed), sin),
        )
        key_embed = ops.cast(key, dtype="float32")
        key_embed = ops.add(
            ops.multiply(key_embed, cos),
            ops.multiply(rotate_pairwise(key_embed), sin),
        )
        query_embed = ops.cast(query_embed, dtype=original_dtype)
        key_embed = ops.cast(key_embed, dtype=original_dtype)

        query_embed = ops.transpose(query_embed, axes=(0, 2, 1, 3))
        key_embed = ops.transpose(key_embed, axes=(0, 2, 1, 3))
        return query_embed, key_embed

    def call(self, hidden_states, position_embeddings, training=None):
        new_shape = (
            -1,
            self.height * self.width,
            self.num_heads,
            self.head_dim,
        )

        query = self.q_proj(hidden_states, training=training)
        query = ops.reshape(query, new_shape)
        key = self.k_proj(hidden_states, training=training)
        key = ops.reshape(key, new_shape)
        value = self.v_proj(hidden_states, training=training)
        value = ops.reshape(value, new_shape)
        cos, sin = position_embeddings
        query, key = self.apply_rotary_pos_emb_2d(query, key, cos=cos, sin=sin)

        attention_output = ops.dot_product_attention(
            query, key, value, scale=self.scale, is_causal=False
        )
        attention_output = ops.reshape(
            attention_output, (-1, self.height, self.width, self.hidden_dim)
        )
        attention_output = self.o_proj(attention_output, training=training)
        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "attention_dropout_rate": self.attention_dropout_rate,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class SAM3PatchEmbedding(layers.Layer):
    def __init__(self, hidden_dim, patch_size, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.data_format = standardize_data_format(data_format)

        self.projection = layers.Conv2D(
            self.hidden_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=False,
            dtype=self.dtype_policy,
            name="projection",
        )

    def build(self, input_shape):
        self.projection.build(input_shape)
        output_shape = self.projection.compute_output_shape(input_shape)
        if self.data_format == "channels_last":
            self.seq_len = int(output_shape[1]) * int(output_shape[2])
        else:
            self.seq_len = int(output_shape[2]) * int(output_shape[3])

    def call(self, inputs, training=None):
        embeddings = self.projection(inputs, training=training)
        if self.data_format == "channels_last":
            embeddings = ops.reshape(
                embeddings, (-1, self.seq_len, self.hidden_dim)
            )
        else:
            embeddings = ops.reshape(
                embeddings, (-1, self.hidden_dim, self.seq_len)
            )
            embeddings = ops.transpose(embeddings, (0, 2, 1))
        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], None, self.hidden_dim]
        if self.data_format == "channels_last":
            if input_shape[1] is not None and input_shape[2] is not None:
                patch_num = input_shape[1] // self.patch_size
                output_shape[1] = patch_num**2
        else:
            if input_shape[2] is not None and input_shape[3] is not None:
                patch_num = input_shape[2] // self.patch_size
                output_shape[1] = patch_num**2
        return output_shape


class SAM3Embedding(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        patch_size,
        image_shape,
        dropout_rate=0.0,
        pretrain_image_shape=(336, 336, 3),
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.image_shape = (
            int(image_shape[0]),
            int(image_shape[1]),
            int(image_shape[2]),
        )
        self.dropout_rate = float(dropout_rate)
        self.pretrain_image_shape = (
            int(pretrain_image_shape[0]),
            int(pretrain_image_shape[1]),
            int(pretrain_image_shape[2]),
        )
        self.data_format = standardize_data_format(data_format)
        self.num_patches = (self.pretrain_image_shape[0] // self.patch_size) * (
            self.pretrain_image_shape[1] // self.patch_size
        )
        self.tiled_num_patches = (self.image_shape[0] // self.patch_size) * (
            self.image_shape[1] // self.patch_size
        )

        self.patch_embeddings = SAM3PatchEmbedding(
            hidden_dim=self.hidden_dim,
            patch_size=self.patch_size,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="patch_embeddings",
        )
        self.dropout = layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout"
        )

    def build(self, input_shape):
        self.patch_embeddings.build(input_shape)
        embedding_shape = self.patch_embeddings.compute_output_shape(
            input_shape
        )
        self.dropout.build(embedding_shape)

        # Note that there are two position embeddings:
        # `self.tiled_position_embeddings` is used for the image inputs during
        # both training and inference.
        # `self.position_embeddings` is used to load pretrained weights and
        # remains unchanged during training and inference. It will be updated
        # during saving once `self.tiled_position_embeddings` is modified.
        self.position_embeddings = self.add_weight(
            shape=(1, self.num_patches, self.hidden_dim),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="position_embeddings",
        )
        self.tiled_position_embeddings = self.add_weight(
            shape=(1, self.tiled_num_patches, self.hidden_dim),
            initializer="zeros",  # Will be initialized by tiling.
            trainable=True,
            name="tiled_position_embeddings",
        )

        # Initialize the interpolated position embeddings.
        self.tiled_position_embeddings.assign(
            self._tile_position_embeddings(
                self.position_embeddings,
                patch_size=self.patch_size,
                source_shape=self.pretrain_image_shape,
                target_shape=self.image_shape,
            )
        )

    def call(self, inputs, training=None):
        x = inputs
        patch_embeddings = self.patch_embeddings(x, training=training)
        if self.data_format == "channels_last":
            patch_embeddings = ops.reshape(
                patch_embeddings,
                (-1, self.patch_embeddings.seq_len, self.hidden_dim),
            )
        else:
            patch_embeddings = ops.reshape(
                patch_embeddings,
                (-1, self.hidden_dim, self.patch_embeddings.seq_len),
            )
            patch_embeddings = ops.transpose(patch_embeddings, (0, 2, 1))
        embeddings = ops.add(patch_embeddings, self.tiled_position_embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
                "image_shape": self.image_shape,
                "dropout_rate": self.dropout_rate,
                "pretrain_image_shape": self.pretrain_image_shape,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        if input_shape is None:
            input_shape = [None, None, None, None]
        output_shape = [input_shape[0], None, self.hidden_dim]
        if self.data_format == "channels_last":
            if input_shape[1] is not None and input_shape[2] is not None:
                patch_num = input_shape[1] // self.patch_size
                output_shape[1] = patch_num**2
        else:
            if input_shape[2] is not None and input_shape[3] is not None:
                patch_num = input_shape[2] // self.patch_size
                output_shape[1] = patch_num**2
        return output_shape

    @staticmethod
    def _tile_position_embeddings(
        position_embeddings, patch_size, source_shape, target_shape
    ):
        """Tile position embeddings to match the target image shape.

        Reference:
            - https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam3/modeling_sam3.py
        """
        position_embeddings = ops.convert_to_tensor(position_embeddings)
        patch_size = int(patch_size)
        source_shape = (int(source_shape[0]), int(source_shape[1]))
        target_shape = (int(target_shape[0]), int(target_shape[1]))
        hidden_dim = int(position_embeddings.shape[-1])

        if (
            source_shape[0] == target_shape[0]
            and source_shape[1] == target_shape[1]
        ):
            # No need to tile if the image size is the same as the
            # position embedding image size.
            return ops.copy(position_embeddings)

        # Tile position embeddings to match target image size.
        source_embedding_shape = (
            source_shape[0] // patch_size,
            source_shape[1] // patch_size,
        )
        target_embedding_shape = (
            target_shape[0] // patch_size,
            target_shape[1] // patch_size,
        )
        position_embeddings = ops.reshape(
            position_embeddings,
            (
                1,
                source_embedding_shape[0],
                source_embedding_shape[1],
                hidden_dim,
            ),
        )
        repeat_h = target_embedding_shape[0] // source_embedding_shape[0] + 1
        repeat_w = target_embedding_shape[1] // source_embedding_shape[1] + 1
        position_embeddings = ops.tile(
            position_embeddings, (1, repeat_h, repeat_w, 1)
        )
        position_embeddings = position_embeddings[
            :, : target_embedding_shape[0], : target_embedding_shape[1], :
        ]
        return ops.reshape(position_embeddings, (1, -1, hidden_dim))

    def _is_tiled_position_embeddings_updated(self):
        """Check if the tiled position embeddings are updated."""
        original_tiled_position_embeddings = self._tile_position_embeddings(
            self.position_embeddings,
            patch_size=self.patch_size,
            source_shape=self.pretrain_image_shape,
            target_shape=self.image_shape,
        )
        diff = ops.sum(
            ops.subtract(
                original_tiled_position_embeddings,
                self.tiled_position_embeddings,
            )
        )
        return ops.cond(
            ops.greater(diff, config.epsilon()), lambda: True, lambda: False
        )

    def save_own_variables(self, store):
        if self._is_tiled_position_embeddings_updated():
            self.position_embeddings.assign(
                self._tile_position_embeddings(
                    self.tiled_position_embeddings,
                    patch_size=self.patch_size,
                    source_shape=self.image_shape,
                    target_shape=self.pretrain_image_shape,
                )
            )
        super().save_own_variables(store)

    def load_own_variables(self, store):
        all_vars = self._trainable_variables + self._non_trainable_variables
        for i, v in enumerate(all_vars):
            if v is self.tiled_position_embeddings:
                continue
            v.assign(store[f"{i}"])
        self.tiled_position_embeddings.assign(
            self._tile_position_embeddings(
                self.position_embeddings,
                patch_size=self.patch_size,
                source_shape=self.pretrain_image_shape,
                target_shape=self.image_shape,
            )
        )


class SAM3SinePositionEmbedding(layers.Layer):
    def __init__(
        self,
        num_pos_feats=64,
        temperature=10000,
        normalize=False,
        scale=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def build(self, input_shape=None):
        if self.built:
            return

    def encode_1d_positions(self, x, y):
        x_embed = ops.multiply(x, self.scale)
        y_embed = ops.multiply(y, self.scale)
        dim_t = ops.cast(ops.arange(self.num_pos_feats), dtype=x.dtype)
        dim_t = ops.power(
            self.temperature,
            ops.divide(
                ops.multiply(2, ops.floor_divide(dim_t, 2)), self.num_pos_feats
            ),
        )
        pos_x = ops.divide(ops.expand_dims(x_embed, -1), dim_t)
        pos_y = ops.divide(ops.expand_dims(y_embed, -1), dim_t)
        pos_x = ops.stack(
            (ops.sin(pos_x[:, 0::2]), ops.cos(pos_x[:, 1::2])), axis=2
        )
        pos_x = ops.reshape(pos_x, (-1, self.num_pos_feats))
        pos_y = ops.stack(
            (ops.sin(pos_y[:, 0::2]), ops.cos(pos_y[:, 1::2])), axis=2
        )
        pos_y = ops.reshape(pos_y, (-1, self.num_pos_feats))
        return pos_x, pos_y

    def encode_boxes(self, boxes):
        dim_t = ops.cast(ops.arange(self.num_pos_feats), dtype=boxes.dtype)
        dim_t = ops.power(
            self.temperature,
            ops.divide(
                ops.multiply(2, ops.floor_divide(dim_t, 2)), self.num_pos_feats
            ),
        )

        x_embed = ops.multiply(boxes[..., 0], self.scale)
        y_embed = ops.multiply(boxes[..., 1], self.scale)
        w_embed = ops.multiply(boxes[..., 2], self.scale)
        h_embed = ops.multiply(boxes[..., 3], self.scale)
        pos_x = ops.divide(ops.expand_dims(x_embed, -1), dim_t)
        pos_y = ops.divide(ops.expand_dims(y_embed, -1), dim_t)
        pos_w = ops.divide(ops.expand_dims(w_embed, -1), dim_t)
        pos_h = ops.divide(ops.expand_dims(h_embed, -1), dim_t)
        pos_x_shape = ops.shape(pos_x)
        newshape = (pos_x_shape[0], pos_x_shape[1], self.num_pos_feats)
        pos_x = ops.stack(
            (ops.sin(pos_x[..., 0::2]), ops.cos(pos_x[..., 1::2])), axis=3
        )
        pos_x = ops.reshape(pos_x, newshape)
        pos_y = ops.stack(
            (ops.sin(pos_y[..., 0::2]), ops.cos(pos_y[..., 1::2])), axis=3
        )
        pos_y = ops.reshape(pos_y, newshape)
        pos_w = ops.stack(
            (ops.sin(pos_w[..., 0::2]), ops.cos(pos_w[..., 1::2])), axis=3
        )
        pos_w = ops.reshape(pos_w, newshape)
        pos_h = ops.stack(
            (ops.sin(pos_h[..., 0::2]), ops.cos(pos_h[..., 1::2])), axis=3
        )
        pos_h = ops.reshape(pos_h, newshape)
        return ops.concatenate([pos_y, pos_x, pos_w, pos_h], axis=2)

    def call(self, inputs, height, width, training=None):
        not_mask = ops.ones((1, height, width), dtype=self.compute_dtype)
        y_embed = ops.cumsum(not_mask, axis=1)
        x_embed = ops.cumsum(not_mask, axis=2)
        if self.normalize:
            eps = 1e-6
            y_embed = ops.multiply(
                ops.divide(y_embed, ops.add(y_embed[:, -1:, :], eps)),
                self.scale,
            )
            x_embed = ops.multiply(
                ops.divide(x_embed, ops.add(x_embed[:, :, -1:], eps)),
                self.scale,
            )
        dim_t = ops.cast(
            ops.arange(self.num_pos_feats), dtype=self.compute_dtype
        )
        dim_t = ops.power(
            self.temperature,
            ops.divide(
                ops.multiply(2, ops.floor_divide(dim_t, 2)), self.num_pos_feats
            ),
        )

        pos_x = ops.divide(ops.expand_dims(x_embed, -1), dim_t)
        pos_y = ops.divide(ops.expand_dims(y_embed, -1), dim_t)
        newshape = (1, height, width, self.num_pos_feats)
        pos_x = ops.stack(
            (ops.sin(pos_x[..., 0::2]), ops.cos(pos_x[..., 1::2])), axis=4
        )
        pos_x = ops.reshape(pos_x, newshape)
        pos_y = ops.stack(
            (ops.sin(pos_y[..., 0::2]), ops.cos(pos_y[..., 1::2])), axis=4
        )
        pos_y = ops.reshape(pos_y, newshape)
        pos = ops.concatenate([pos_y, pos_x], axis=3)
        pos = ops.tile(pos, (ops.shape(inputs)[0], 1, 1, 1))
        return pos

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_pos_feats": self.num_pos_feats,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "scale": self.scale,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] = self.num_pos_feats * 2
        return output_shape


class SAM3DecoderMLP(layers.Layer):
    def __init__(self, num_layers, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        if self.num_layers == 2:
            self.layer1 = layers.Dense(
                hidden_dim, dtype=self.dtype_policy, name="layer1"
            )
            self.layer2 = layers.Dense(
                output_dim, dtype=self.dtype_policy, name="layer2"
            )
        elif num_layers == 3:
            self.layer1 = layers.Dense(
                hidden_dim, dtype=self.dtype_policy, name="layer1"
            )
            self.layer2 = layers.Dense(
                hidden_dim, dtype=self.dtype_policy, name="layer2"
            )
            self.layer3 = layers.Dense(
                output_dim, dtype=self.dtype_policy, name="layer3"
            )
        else:
            raise ValueError("num_layers should be 2 or 3.")

    def build(self, input_shape):
        self.layer1.build(input_shape)
        input_shape = self.layer1.compute_output_shape(input_shape)
        self.layer2.build(input_shape)
        if self.num_layers == 3:
            input_shape = self.layer2.compute_output_shape(input_shape)
            self.layer3.build(input_shape)

    def call(self, inputs, training=None):
        x = ops.relu(self.layer1(inputs, training=training))
        if self.num_layers == 2:
            return self.layer2(x, training=training)
        else:
            x = ops.relu(self.layer2(x, training=training))
            return self.layer3(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return output_shape


class SAM3BoxDecoder(layers.Layer):
    def build(
        self,
        box_offsets_shape,
        reference_boxes_shape,
        pred_logits_shape,
        presence_logits_shape,
    ):
        pass

    def call(
        self,
        box_offsets,
        reference_boxes,
        pred_logits,
        presence_logits,
        training=None,
    ):
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        pred_boxes_cxcywh = ops.nn.sigmoid(
            ops.add(reference_boxes_inv_sig, box_offsets)
        )
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes_cxcywh)
        return (
            pred_boxes[:, -1],
            pred_logits[:, -1, :, 0],
            presence_logits[:, -1],
        )

    def compute_output_shape(
        self,
        box_offsets_shape,
        reference_boxes_shape,
        pred_logits_shape,
        presence_logits_shape,
    ):
        pred_boxes_shape = [
            box_offsets_shape[0],
            box_offsets_shape[-2],
            box_offsets_shape[-1],
        ]
        pred_logits_shape = [
            pred_logits_shape[0],
            pred_logits_shape[-2],
        ]
        presence_logits_shape = [
            presence_logits_shape[0],
            presence_logits_shape[-2],
            presence_logits_shape[-1],
        ]
        return pred_boxes_shape, pred_logits_shape, presence_logits_shape
