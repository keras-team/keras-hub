import functools
import math

import keras
import numpy as np
from keras import ops

from keras_hub.src.models.qwen_moe.qwen_moe_decoder import QwenMoeExperts
from keras_hub.src.models.qwen_moe.qwen_moe_decoder import QwenMoeMLP
from keras_hub.src.utils.keras_utils import clone_initializer


def _kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


def _cubic_kernel(x):
    """The cubic convolution kernel with `a=-0.5`."""
    x = np.abs(x)
    near = ((-0.5 + 2) * x - (-0.5 + 3)) * x * x + 1
    far = (((x - 5) * x + 8) * x - 4) * -0.5
    return np.where(x < 1, near, np.where(x < 2, far, 0.0))


@functools.lru_cache(maxsize=None)
def _bicubic_antialias_matrix(size_in, size_out):
    """The weights of an antialiased bicubic resize along one axis.

    The weights follow `F.interpolate(mode="bicubic", antialias=True,
    align_corners=False)` of PyTorch. A strided window scales the kernel
    when the axis shrinks. The weights of every output index sum to one.
    The sizes are static, so the weights are constants. The resize then runs
    as a matrix product. TensorFlow uses this path, because its antialiased
    resize op (`ScaleAndTranslate`) has no XLA kernel.

    Args:
        size_in: int. The input size of the axis.
        size_out: int. The output size of the axis.

    Returns:
        A `(size_out, size_in)` float64 NumPy array.
    """
    scale = size_in / size_out
    filter_scale = max(scale, 1.0)
    support = 2.0 * filter_scale
    center = (np.arange(size_out) + 0.5) * scale
    start = np.maximum(np.floor(center - support + 0.5), 0)
    stop = np.minimum(np.floor(center + support + 0.5), size_in)
    index = np.arange(size_in)[None, :]
    inside = (index >= start[:, None]) & (index < stop[:, None])
    weights = _cubic_kernel((index + 0.5 - center[:, None]) / filter_scale)
    weights = weights * inside
    return weights / weights.sum(axis=1, keepdims=True)


def _resize_bicubic(x, size):
    """Resize a `(1, height, width, channels)` tensor with bicubic weights.

    The resize runs in float32, like the Hugging Face source
    (modeling_unlimited_ocr.py:638).
    """
    dtype = x.dtype
    x = ops.cast(x, "float32")
    if keras.config.backend() == "tensorflow":
        # The antialiased resize op of TensorFlow does not compile with XLA.
        height_weights = ops.convert_to_tensor(
            _bicubic_antialias_matrix(x.shape[1], size[0]), dtype="float32"
        )
        width_weights = ops.convert_to_tensor(
            _bicubic_antialias_matrix(x.shape[2], size[1]), dtype="float32"
        )
        x = ops.einsum("oh,bhwc->bowc", height_weights, x)
        x = ops.einsum("pw,bowc->bopc", width_weights, x)
        return ops.cast(x, dtype)
    x = ops.image.resize(
        x,
        size,
        interpolation="bicubic",
        antialias=True,
        data_format="channels_last",
    )
    return ops.cast(x, dtype)


class UnlimitedOCRSamPositionEmbedding(keras.layers.Layer):
    """Learned absolute position embedding that resizes to the input grid.

    The layer holds a `(1, grid_size, grid_size, hidden_dim)` table. When the
    input grid differs from the table, the layer resizes the table with
    antialiased bicubic interpolation.

    Args:
        grid_size: int. The side of the position table.
        hidden_dim: int. The number of channels.
    """

    def __init__(self, grid_size, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.grid_size, self.grid_size, self.hidden_dim),
            initializer="zeros",
        )
        self.built = True

    def call(self, x):
        height, width = x.shape[1], x.shape[2]
        pos_embed = self.pos_embed
        if height != self.grid_size or width != self.grid_size:
            pos_embed = _resize_bicubic(pos_embed, (height, width))
        return x + pos_embed

    def compute_output_spec(self, x):
        return keras.KerasTensor(x.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"grid_size": self.grid_size, "hidden_dim": self.hidden_dim}
        )
        return config


class UnlimitedOCRCLIPEmbedding(keras.layers.Layer):
    """CLIP embedding that takes a feature map as patch tokens.

    The layer flattens the feature map, prepends a class embedding, and adds
    the position table. When the feature grid differs from the table grid,
    the layer resizes the patch part of the table with antialiased bicubic
    interpolation (modeling_unlimited_ocr.py:795-851). The CLIP convolution
    for patches does not exist here, because the SAM features act as patches.

    Args:
        hidden_dim: int. The number of channels.
        grid_size: int. The side of the patch grid of the position table.
    """

    def __init__(self, hidden_dim, grid_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

    def build(self, input_shape):
        self.class_embedding = self.add_weight(
            name="class_embedding",
            shape=(self.hidden_dim,),
            initializer=_kernel_initializer(self.hidden_dim**-0.5),
        )
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.grid_size**2 + 1, self.hidden_dim),
            initializer=_kernel_initializer(),
        )
        self.built = True

    def call(self, x):
        batch = ops.shape(x)[0]
        height, width = x.shape[1], x.shape[2]
        x = ops.reshape(x, (batch, height * width, self.hidden_dim))
        class_embedding = ops.broadcast_to(
            ops.reshape(self.class_embedding, (1, 1, self.hidden_dim)),
            (batch, 1, self.hidden_dim),
        )
        x = ops.concatenate([class_embedding, x], axis=1)

        position_embedding = self.position_embedding
        if height != self.grid_size or width != self.grid_size:
            patch_positions = ops.reshape(
                position_embedding[1:],
                (1, self.grid_size, self.grid_size, self.hidden_dim),
            )
            patch_positions = _resize_bicubic(patch_positions, (height, width))
            patch_positions = ops.reshape(
                patch_positions, (height * width, self.hidden_dim)
            )
            position_embedding = ops.concatenate(
                [position_embedding[:1], patch_positions], axis=0
            )
        return x + position_embedding[None, ...]

    def compute_output_spec(self, x):
        shape = (x.shape[0], x.shape[1] * x.shape[2] + 1, self.hidden_dim)
        return keras.KerasTensor(shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_dim": self.hidden_dim, "grid_size": self.grid_size}
        )
        return config


class UnlimitedOCRSparseMoeBlock(keras.layers.Layer):
    """Mixture of experts block with always-on shared experts.

    The router computes a softmax over all experts in float32. The block
    picks the `top_k` experts for each token. The block does not renormalize
    the selected weights. The output is the weighted sum of the routed expert
    outputs plus the output of the shared experts. No gate scales the shared
    experts.

    Args:
        hidden_dim: int. The size of the hidden state.
        moe_intermediate_dim: int. The intermediate size of one routed
            expert.
        num_shared_experts: int. The number of shared experts. The shared
            experts form one feedforward block with an intermediate size of
            `moe_intermediate_dim * num_shared_experts`.
        num_experts: int. The number of routed experts.
        top_k: int. The number of routed experts for each token.
        routed_scaling_factor: float. The factor that scales the routing
            weights. Defaults to `1.0`.
        kernel_initializer: string or `keras.initializers` initializer. The
            initializer of the kernels. Defaults to `"glorot_uniform"`.
    """

    def __init__(
        self,
        hidden_dim,
        moe_intermediate_dim,
        num_shared_experts,
        num_experts,
        top_k,
        routed_scaling_factor=1.0,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.moe_intermediate_dim = moe_intermediate_dim
        self.num_shared_experts = num_shared_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.kernel_initializer = kernel_initializer

    def build(self, decoder_sequence_shape):
        # The router runs in float32 for every compute dtype
        # (modeling_unlimited_ocr.py:1161). `autocast=False` keeps the kernel
        # at its stored precision when the layer reads it.
        self.router_kernel = self.add_weight(
            name="router_kernel",
            shape=(self.hidden_dim, self.num_experts),
            initializer=clone_initializer(self.kernel_initializer),
            dtype=self.variable_dtype,
            autocast=False,
        )

        self.expert_bank = QwenMoeExperts(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.moe_intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            name="experts",
            dtype=self.dtype_policy,
        )
        self.expert_bank.build(decoder_sequence_shape)

        # The shared experts add to the output with no gate
        # (`modeling_unlimited_ocr.py:1194-1197`).
        self.shared_expert_dense = QwenMoeMLP(
            intermediate_dim=self.moe_intermediate_dim
            * self.num_shared_experts,
            hidden_dim=self.hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            name="shared_expert_dense",
            dtype=self.dtype_policy,
        )
        self.shared_expert_dense.build(decoder_sequence_shape)
        self.built = True

    def call(self, hidden_states):
        hidden_states = ops.cast(hidden_states, self.compute_dtype)
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]
        flat_states = ops.reshape(hidden_states, (-1, self.hidden_dim))

        router_logits = ops.matmul(
            ops.cast(flat_states, "float32"),
            ops.cast(self.router_kernel, "float32"),
        )
        router_probs = ops.softmax(router_logits, axis=-1)
        top_p, top_i = ops.top_k(router_probs, k=self.top_k)
        top_p = top_p * self.routed_scaling_factor

        one_hot = ops.cast(ops.one_hot(top_i, self.num_experts), top_p.dtype)
        routing_full = ops.sum(one_hot * top_p[..., None], axis=1)
        routing_full = ops.cast(
            ops.transpose(routing_full, (1, 0)), flat_states.dtype
        )

        expert_out = self.expert_bank(flat_states)
        routed_out = ops.einsum("eth,et->th", expert_out, routing_full)
        out = routed_out + self.shared_expert_dense(flat_states)
        out = ops.cast(out, self.compute_dtype)
        return ops.reshape(out, (batch_size, seq_len, self.hidden_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "moe_intermediate_dim": self.moe_intermediate_dim,
                "num_shared_experts": self.num_shared_experts,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "routed_scaling_factor": self.routed_scaling_factor,
                "kernel_initializer": keras.initializers.serialize(
                    keras.initializers.get(self.kernel_initializer)
                ),
            }
        )
        return config


class UnlimitedOCRImageFeatureExtractor(keras.layers.Layer):
    """Computes projected features for the global views and the local tiles.

    The layer runs the vision tower and the projector on both inputs. Each
    input has a sample axis and an image axis, or a tile axis. The layer
    flattens the two axes before the vision tower. A batch with no images,
    or no tiles, skips the vision tower. Some backends cannot run a
    convolution with a batch size of `0`.

    Args:
        vision_encoder: A `keras_hub.models.UnlimitedOCRVisionEncoder`
            instance.
        multimodal_projector: A `keras.layers.Dense` instance. The layer
            maps the vision features to the text hidden size.
    """

    def __init__(self, vision_encoder, multimodal_projector, **kwargs):
        super().__init__(**kwargs)
        self.vision_encoder = vision_encoder
        self.multimodal_projector = multimodal_projector

    def build(self, pixel_values_shape, pixel_values_local_shape):
        if not self.vision_encoder.built:
            self.vision_encoder.build(pixel_values_shape)
        if not self.multimodal_projector.built:
            self.multimodal_projector.build(
                (None, None, self.vision_encoder.output_dim)
            )
        self.built = True

    def _num_tokens(self, image_shape):
        # Two strided convolutions with a kernel of 3 and a padding of 1.
        patch_size = self.vision_encoder.sam_patch_size
        grid = [size // patch_size for size in image_shape]
        for _ in range(2):
            grid = [(size + 1) // 2 for size in grid]
        return grid[0] * grid[1]

    def _encode(self, pixel_values, image_shape):
        def encode():
            features = self.vision_encoder(pixel_values)
            return self.multimodal_projector(features)

        def skip():
            return ops.zeros(
                (
                    0,
                    self._num_tokens(image_shape),
                    self.multimodal_projector.units,
                ),
                dtype=self.compute_dtype,
            )

        batch_size = pixel_values.shape[0]
        if batch_size == 0:
            return skip()
        if batch_size is not None:
            return encode()
        # The batch size is unknown while TensorFlow traces a graph. A text
        # batch then reaches the encoder with no image, so check at run time.
        return ops.cond(ops.shape(pixel_values)[0] > 0, encode, skip)

    def _flatten(self, images):
        return ops.reshape(images, (-1, *images.shape[2:]))

    def call(self, pixel_values, pixel_values_local):
        global_features = self._encode(
            self._flatten(pixel_values), pixel_values.shape[2:4]
        )
        local_features = self._encode(
            self._flatten(pixel_values_local), pixel_values_local.shape[2:4]
        )
        return global_features, local_features

    def compute_output_spec(self, pixel_values, pixel_values_local):
        units = self.multimodal_projector.units

        def spec(images):
            batch = images.shape[:2]
            batch = None if None in batch else batch[0] * batch[1]
            return keras.KerasTensor(
                (batch, self._num_tokens(images.shape[2:4]), units),
                dtype=self.compute_dtype,
            )

        return spec(pixel_values), spec(pixel_values_local)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder": keras.layers.serialize(self.vision_encoder),
                "multimodal_projector": keras.layers.serialize(
                    self.multimodal_projector
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["vision_encoder"] = keras.layers.deserialize(
            config["vision_encoder"]
        )
        config["multimodal_projector"] = keras.layers.deserialize(
            config["multimodal_projector"]
        )
        return super().from_config(config)


class UnlimitedOCRImageTextEmbeddingMerger(keras.layers.Layer):
    """Writes the image features into the image placeholder positions.

    The layer rebuilds the feature sequence of every image, in the order
    that the Hugging Face model uses (modeling_unlimited_ocr.py:1877-1900).
    For an image with local tiles, the sequence has the local rows, the
    global rows, and one view separator. Each row ends with one newline
    embedding. An image with no tiles has the global rows and the view
    separator. The layer then replaces the embedding of every placeholder
    token of a sample with the next feature of the concatenated sequence of
    the images of that sample. The images of a sample follow the order of
    its placeholders. This matches `masked_scatter` on each sample.

    A sample with fewer images than the other samples has padding images
    and padding tiles. They have a grid of `(0, 0)` and zero pixels. No
    placeholder selects them, because the padding follows the real images.

    The layer needs no data-dependent shapes. It computes, for every
    placeholder, the index of its feature in a bank that holds the local
    tile features, the global features, the newline and the separator.

    Args:
        hidden_dim: int. The text hidden size.
        image_token_id: int. The id of the placeholder token.

    Call arguments:
        token_ids: int tensor of shape `(batch, length)`.
        token_embeddings: float tensor of shape `(batch, length, hidden_dim)`.
        global_features: float tensor of shape
            `(batch * max_images, global_tokens, hidden_dim)`. The images
            of sample 0 come first. The tokens form a square grid.
        local_features: float tensor of shape
            `(batch * max_tiles, local_tokens, hidden_dim)`. The tiles of
            sample 0 come first. The tokens of one tile form a square grid.
            The tiles of a sample follow its images. The tiles of one image
            are in row-major order.
        patches_grid: int tensor of shape `(batch, max_images, 2)`. The
            number of tile columns and tile rows of each image. An image
            with no tiles, and a padding image, has `(0, 0)`.
    """

    def __init__(self, hidden_dim, image_token_id, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.image_token_id = image_token_id

    def build(
        self,
        token_ids_shape,
        token_embeddings_shape,
        global_features_shape,
        local_features_shape,
        patches_grid_shape,
    ):
        initializer = keras.initializers.RandomNormal(
            stddev=self.hidden_dim**-0.5
        )
        self.image_newline = self.add_weight(
            name="image_newline",
            shape=(self.hidden_dim,),
            initializer=initializer,
        )
        self.view_separator = self.add_weight(
            name="view_separator",
            shape=(self.hidden_dim,),
            initializer=initializer,
        )
        self.built = True

    def call(
        self,
        token_ids,
        token_embeddings,
        global_features,
        local_features,
        patches_grid,
    ):
        hidden_dim = self.hidden_dim
        batch_size = ops.shape(token_embeddings)[0]
        global_side = math.isqrt(global_features.shape[1])
        local_side = math.isqrt(local_features.shape[1])
        max_images = ops.shape(patches_grid)[1]
        max_tiles = ops.floor_divide(ops.shape(local_features)[0], batch_size)

        # One extra image with no tiles keeps every gather valid when a
        # sample has no image. No placeholder selects it.
        grid = ops.cast(patches_grid, "int32")
        padding = ops.zeros((batch_size, 1, 2), dtype="int32")
        grid = ops.concatenate([grid, padding], axis=1)
        columns, rows = grid[..., 0], grid[..., 1]
        tiles = columns * rows
        local_tokens = rows * local_side * (columns * local_side + 1)
        image_tokens = local_tokens + global_side * (global_side + 1) + 1
        image_offset = ops.cumsum(image_tokens, axis=1) - image_tokens
        tile_offset = ops.cumsum(tiles, axis=1) - tiles

        # The ordinal of a placeholder is its rank among the placeholders of
        # its sample.
        is_image = ops.equal(token_ids, self.image_token_id)
        ordinal = ops.cumsum(ops.cast(is_image, "int32"), axis=1) - 1
        ordinal = ops.maximum(ordinal, 0)
        slot = (
            ops.sum(
                ops.cast(
                    ordinal[:, :, None] >= image_offset[:, None, :], "int32"
                ),
                axis=2,
            )
            - 1
        )
        slot = ops.minimum(ops.maximum(slot, 0), max_images)
        position = ordinal - ops.take_along_axis(image_offset, slot, axis=1)
        image_columns = ops.take_along_axis(columns, slot, axis=1)
        image_local_tokens = ops.take_along_axis(local_tokens, slot, axis=1)
        sample = ops.arange(batch_size, dtype="int32")[:, None]

        # Local part: rows of `columns * local_side` tokens and a newline.
        row_width = image_columns * local_side + 1
        local_row = ops.floor_divide(position, row_width)
        local_column = ops.mod(position, row_width)
        local_is_newline = local_column == image_columns * local_side
        tile = (
            sample * max_tiles
            + ops.take_along_axis(tile_offset, slot, axis=1)
            + ops.floor_divide(local_row, local_side) * image_columns
            + ops.floor_divide(local_column, local_side)
        )
        local_bank_index = (
            tile * local_side * local_side
            + ops.mod(local_row, local_side) * local_side
            + ops.mod(local_column, local_side)
        )

        # Global part: rows of `global_side` tokens and a newline. The view
        # separator follows.
        global_position = position - image_local_tokens
        global_row = ops.floor_divide(global_position, global_side + 1)
        global_column = ops.mod(global_position, global_side + 1)
        global_is_newline = global_column == global_side
        global_is_separator = global_position >= global_side * (global_side + 1)
        num_local = batch_size * max_tiles * local_side * local_side
        global_bank_index = (
            num_local
            + (sample * max_images + slot) * global_side * global_side
            + global_row * global_side
            + global_column
        )

        newline_index = (
            num_local + batch_size * max_images * global_side * global_side
        )
        bank_index = ops.where(
            position < image_local_tokens,
            ops.where(local_is_newline, newline_index, local_bank_index),
            ops.where(
                global_is_separator,
                newline_index + 1,
                ops.where(global_is_newline, newline_index, global_bank_index),
            ),
        )
        bank_size = newline_index + 2
        bank_index = ops.minimum(ops.maximum(bank_index, 0), bank_size - 1)

        dtype = token_embeddings.dtype
        bank = ops.concatenate(
            [
                ops.cast(ops.reshape(local_features, (-1, hidden_dim)), dtype),
                ops.cast(ops.reshape(global_features, (-1, hidden_dim)), dtype),
                ops.cast(self.image_newline[None, :], dtype),
                ops.cast(self.view_separator[None, :], dtype),
            ],
            axis=0,
        )
        features = ops.take(bank, bank_index, axis=0)
        return ops.where(is_image[..., None], features, token_embeddings)

    def compute_output_spec(
        self,
        token_ids,
        token_embeddings,
        global_features,
        local_features,
        patches_grid,
    ):
        return keras.KerasTensor(
            token_embeddings.shape, dtype=self.compute_dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "image_token_id": self.image_token_id,
            }
        )
        return config
