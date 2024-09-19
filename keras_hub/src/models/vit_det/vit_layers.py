# Copyright 2024 The KerasCV Authors
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


import keras
from keras import ops


class MLP(keras.layers.Layer):
    """A MLP block with architecture.

    The MLP block implements `input_dim -> [intermediate_dim] ->
    hidden_dim`. The code has been adapted from [Segment Anything paper](
    https://arxiv.org/abs/2304.02643), [Segment Anything GitHub](
    https://github.com/facebookresearch/segment-anything) and [Detectron2](
    https://github.com/facebookresearch/detectron2).

    Args:
        intermediate_dim (int): The number of units in the hidden layers.
        hidden_dim (int): The number of units in the output layer.
        activation (str): Activation to use in the hidden layers.
            Default is `"relu"`.
    """

    def __init__(
        self, intermediate_dim, hidden_dim, activation="relu", **kwargs
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        h = [intermediate_dim]
        self.dense_net = []
        for intermediate_dim in h:
            self.dense_net.append(keras.layers.Dense(intermediate_dim))
            self.dense_net.append(keras.layers.Activation(activation))
        self.dense_net.append(keras.layers.Dense(hidden_dim))
        self.dense_net = keras.models.Sequential(self.dense_net)

    def build(self, input_shape):
        self.dense_net.build(input_shape)
        self.built = True

    def call(self, x):
        return self.dense_net(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "hidden_dim": self.hidden_dim,
                "activation": self.activation,
            }
        )
        return config


class AddRelativePositionalEmbedding(keras.layers.Layer):
    def __init__(self, input_size, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.key_dim = key_dim
        self.rel_pos_h = self.add_weight(
            name="rel_pos_h",
            shape=(2 * self.input_size[0] - 1, self.key_dim),
            initializer="zeros",
        )
        self.rel_pos_w = self.add_weight(
            name="rel_pos_w",
            shape=(2 * self.input_size[1] - 1, self.key_dim),
            initializer="zeros",
        )
        self.built = True

    def _get_rel_pos(self, query_size, key_size, rel_pos):
        """Get relative positional embeddings.

        Get relative positional embeddings according to the relative positions
        of query and key sizes.

        Args:
            query_size (int): The number of features of the queries.
            key_size (int): The number of features of the keys.
            rel_pos (tensor): Relative positional embedding tensor.

        Returns:
            tensor: Extracted positional embeddings according to relative
                positions.
        """
        max_rel_dist = 2 * max(query_size, key_size) - 1
        if ops.shape(rel_pos)[0] != max_rel_dist:
            rel_pos_resized = ops.image.resize(
                image=ops.reshape(
                    rel_pos,
                    (1, ops.shape(rel_pos)[0], ops.shape(rel_pos)[1], 1),
                ),
                size=(max_rel_dist, ops.shape(rel_pos)[1]),
                interpolation="bilinear",
            )
            rel_pos_resized = ops.squeeze(rel_pos_resized, axis=(0, -1))
            return rel_pos_resized
        else:
            rel_pos_resized = rel_pos
        # Query coordinates
        query_coordinates = ops.cast(
            ops.arange(query_size), dtype=self.compute_dtype
        )[:, None] * (max(key_size / query_size, 1.0))
        # Key coordinates
        key_coordinates = ops.cast(
            ops.arange(key_size), dtype=self.compute_dtype
        )[None, :] * (max(query_size / key_size, 1.0))
        # Relative coordinates
        relative_coordinates = (query_coordinates - key_coordinates) + (
            key_size - 1
        ) * max(query_size / key_size, 1.0)
        relative_coordinates = ops.cast(relative_coordinates, dtype="int32")
        return ops.take(rel_pos_resized, relative_coordinates, 0)

    def call(self, attention_map, queries, query_size, key_size):
        """Calculate decomposed Relative Positional Embeddings

        The code has been adapted based on
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py  # noqa: E501

        Args:
            attention_map (tensor): Attention map.
            queries (tensor): Queries in the attention layer with shape
                `(batch, query_height * query_width, channels)`.
            query_size (tuple[int, int]): Spatial sequence size of queries with
                `(query_height, query_width)`.
            key_size (tuple[int, int]): Spatial sequence size of keys with
                `(key_height, key_width)`.

        Returns:
            tensor: attention map with added relative positional embeddings.
        """
        query_height, query_width = query_size[0], query_size[1]
        key_height, key_width = key_size[0], key_size[1]
        rel_heights = self._get_rel_pos(
            query_height, key_height, self.rel_pos_h
        )
        rel_widths = self._get_rel_pos(query_width, key_width, self.rel_pos_w)
        shape = ops.shape(queries)
        batch, channels = shape[0], shape[2]
        rel_queries = ops.reshape(
            queries, (batch, query_height, query_width, channels)
        )
        rel_heights = ops.einsum("bhwc,hkc->bhwk", rel_queries, rel_heights)
        rel_widths = ops.einsum("bhwc,wkc->bhwk", rel_queries, rel_widths)
        attention_map = ops.reshape(
            attention_map,
            (batch, query_height, query_width, key_height, key_width),
        )
        attention_map = attention_map + rel_heights[..., :, None]
        attention_map = attention_map + rel_widths[..., None, :]
        attention_map = ops.reshape(
            attention_map,
            (batch, query_height * query_width, key_height * key_width),
        )
        return attention_map

    def get_config(self):
        config = super().get_config()
        config.update({"input_size": self.input_size, "key_dim": self.key_dim})
        return config


class MultiHeadAttentionWithRelativePE(keras.layers.Layer):
    """Multi-head Attention block with relative position embeddings.

    The code has been adapted from [Segment Anything paper](
    https://arxiv.org/abs/2304.02643), [Segment Anything GitHub](
    https://github.com/facebookresearch/segment-anything) and [Detectron2](
    https://github.com/facebookresearch/detectron2).

    Args:
        num_heads (int): Number of attention heads.
        key_dim (int): Size of each attention head for query, key, and
            value.
        use_bias (bool, optional): Whether to use bias when projecting
            the queries, keys, and values. Defaults to `True`.
        use_rel_pos (bool, optional): Whether to use relative positional
            embeddings or not. Defaults to `False`.
        input_size (tuple[int, int], optional): Size of the input image.
            Must be provided when using relative positional embeddings.
            Defaults to `None`.

    Raises:
        ValueError: When `input_size = None` with `use_rel_pos = True`.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        use_bias=True,
        use_rel_pos=False,
        input_size=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = self.key_dim**-0.5
        self.use_bias = use_bias
        self.input_size = input_size
        self.use_rel_pos = use_rel_pos
        self.qkv = keras.layers.Dense(
            key_dim * self.num_heads * 3, use_bias=self.use_bias
        )
        self.projection = keras.layers.Dense(key_dim * self.num_heads)
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError(
                    "Input size must be provided if using relative "
                    "positional encoding."
                )
            self.add_decomposed_reative_pe = AddRelativePositionalEmbedding(
                self.input_size, self.key_dim
            )

    def build(self, input_shape=None):
        self.qkv.build([self.key_dim * self.num_heads])
        self.projection.build([self.key_dim * self.num_heads])
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        batch, height, width, channels = ops.shape(x)
        qkv = ops.transpose(
            ops.reshape(
                self.qkv(x),
                (batch, height * width, 3, self.num_heads, self.key_dim),
            ),
            axes=(2, 0, 3, 1, 4),
        )
        qkv = ops.reshape(
            qkv, (3, batch * self.num_heads, height * width, self.key_dim)
        )
        queries, keys, values = ops.unstack(qkv, axis=0)
        attention_map = (queries * self.scale) @ ops.transpose(
            keys, axes=(0, 2, 1)
        )
        if self.use_rel_pos:
            attention_map = self.add_decomposed_reative_pe(
                attention_map,
                queries=queries,
                query_size=(height, width),
                key_size=(height, width),
            )
        attention_map = ops.softmax(attention_map, axis=-1)
        x = ops.reshape(
            attention_map @ values,
            (batch, self.num_heads, height, width, self.key_dim),
        )
        x = ops.transpose(x, axes=(0, 2, 3, 1, 4))
        x = ops.reshape(x, (batch, height, width, channels))
        x = self.projection(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "use_bias": self.use_bias,
                "use_rel_pos": self.use_rel_pos,
                "input_size": self.input_size,
            }
        )
        return config


class WindowPartitioning(keras.layers.Layer):
    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.built = True

    def partition(self, x):
        batch, height, width, channels = ops.shape(x)
        pad_height = (
            self.window_size - height % self.window_size
        ) % self.window_size
        pad_width = (
            self.window_size - width % self.window_size
        ) % self.window_size
        if pad_height > 0 or pad_width > 0:
            x = ops.pad(x, ((0, 0), (0, pad_height), (0, pad_width), (0, 0)))
        height_padded, width_padded = height + pad_height, width + pad_width
        x = ops.reshape(
            x,
            (
                batch,
                height_padded // self.window_size,
                self.window_size,
                width_padded // self.window_size,
                self.window_size,
                channels,
            ),
        )
        windows = ops.reshape(
            ops.transpose(x, axes=(0, 1, 3, 2, 4, 5)),
            (-1, self.window_size, self.window_size, channels),
        )
        return windows, (height_padded, width_padded)

    def unpartition(self, windows, height_width_padded, height_width):
        height_padded, width_padded = height_width_padded
        height, width = height_width
        batch = ops.shape(windows)[0] // (
            (height_padded // self.window_size)
            * (width_padded // self.window_size)
        )
        x = ops.reshape(
            windows,
            (
                batch,
                height_padded // self.window_size,
                width_padded // self.window_size,
                self.window_size,
                self.window_size,
                -1,
            ),
        )
        x = ops.reshape(
            ops.transpose(x, axes=(0, 1, 3, 2, 4, 5)),
            (batch, height_padded, width_padded, -1),
        )
        return x[:, :height, :width, :]

    def get_config(self):
        config = super().get_config()
        config.update({"window_size": self.window_size})
        return config


class WindowedTransformerEncoder(keras.layers.Layer):
    """Implements windowed transformer encoder.

    Transformer blocks with support of window attention and residual
    propagation blocks. The code has been adapted from [Segment Anything paper](
    https://arxiv.org/abs/2304.02643), [Segment Anything GitHub](
    https://github.com/facebookresearch/segment-anything) and [Detectron2](
    https://github.com/facebookresearch/detectron2).

    Args:
        project_dim (int): the dimensionality of the projection of the
            encoder, and output of the `MultiHeadAttention`.
        intermediate_dim (int): the intermediate dimensionality of the MLP head
            before projecting to `project_dim`.
        num_heads (int): the number of heads for the `MultiHeadAttention`
            layer.
        use_bias (bool, optional): Whether to use bias to project the keys,
            queries, and values in the attention layer. Defaults to `True`.
        use_rel_pos (bool, optional): Whether to use relative positional
            emcodings in the attention layer. Defaults to `False`.
        window_size (int, optional): Window size for windowed attention.
            Defaults to `0`.
        input_size (tuple[int, int], optional): Height and width of the input
            image as a tuple of integers. Must be provided when using relative
            positional embeddings. Defaults to `None`.
        activation (str, optional): the activation function to apply in the
            MLP head - should be a function. Defaults to `"gelu"`.
        layer_norm_epsilon (float, optional): The epsilon to use in the layer
            normalization layers. Defaults to `1e-6`.
    """

    def __init__(
        self,
        project_dim,
        intermediate_dim,
        num_heads,
        use_bias=True,
        use_rel_pos=False,
        window_size=0,
        input_size=None,
        activation="gelu",
        layer_norm_epsilon=1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.input_size = input_size
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.window_size = window_size
        self.use_rel_pos = use_rel_pos

        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.attention = MultiHeadAttentionWithRelativePE(
            num_heads=self.num_heads,
            key_dim=self.project_dim // self.num_heads,
            use_bias=use_bias,
            use_rel_pos=use_rel_pos,
            input_size=(
                input_size if window_size == 0 else (window_size, window_size)
            ),
        )
        self.mlp_block = MLP(
            intermediate_dim,
            project_dim,
            activation="gelu",
        )
        self.window_partitioning = WindowPartitioning(window_size)

    def build(self, input_shape=None):
        self.layer_norm1.build([None, None, None, self.project_dim])
        self.layer_norm2.build([None, None, None, self.project_dim])
        self.attention.build()
        self.mlp_block.build([None, None, None, self.project_dim])
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        # Window Partition
        if self.window_size > 0:
            height, width = ops.shape(x)[1], ops.shape(x)[2]
            x, height_width_padded = self.window_partitioning.partition(x)

        x = self.attention(x)
        # Reverse Window Partition
        if self.window_size > 0:
            x = self.window_partitioning.unpartition(
                x,
                height_width_padded=height_width_padded,
                height_width=(height, width),
            )
        x = shortcut + x
        x = x + self.mlp_block(self.layer_norm2(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "use_bias": self.use_bias,
                "use_rel_pos": self.use_rel_pos,
                "window_size": self.window_size,
                "input_size": self.input_size,
                "activation": self.activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class ViTDetPatchingAndEmbedding(keras.layers.Layer):
    """
    Implements a image patch and embedding layer.

    Image to Patch Embedding using only a conv layer (without
    layer normalization).The code has been adapted from [Segment Anything
    paper](https://arxiv.org/abs/2304.02643), [Segment Anything GitHub](
    https://github.com/facebookresearch/segment-anything) and [Detectron2](
    https://github.com/facebookresearch/detectron2).

    Args:
        kernel_size (tuple[int, int], optional): Kernel size of the
            projection layer. Defaults to `(16, 16)`.
        strides (tuple, optional): Strides of the projection layer.
            Defaults to `(16, 16)`.
        embed_dim (int, optional): Number of filters to use in the
            projection layer i.e. projection size. Defaults to `768`.
    """

    def __init__(
        self, kernel_size=(16, 16), strides=(16, 16), embed_dim=768, **kwargs
    ):
        super().__init__(**kwargs)

        self.projection = keras.layers.Conv2D(
            embed_dim, kernel_size=kernel_size, strides=strides
        )
        self.kernel_size = kernel_size
        self.strides = strides
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.projection.build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        return self.projection.compute_output_shape(input_shape)

    def call(self, x):
        x = self.projection(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class AddPositionalEmbedding(keras.layers.Layer):
    def __init__(self, img_size, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(
                1,
                img_size // patch_size,
                img_size // patch_size,
                embed_dim,
            ),
            initializer="zeros",
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        return x + self.pos_embed

    def get_confg(self):
        config = super().get_config()
        config.update(
            {
                "img_size": self.img_size,
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config
