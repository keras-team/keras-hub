import keras
import numpy as np

from keras_hub.src.models.d_fine.d_fine_utils import inverse_sigmoid


class DFineGate(keras.layers.Layer):
    """Gating layer for combining two input tensors using learnable gates.

    This layer is used within the `DFineDecoderLayer` to merge the output of
    the self-attention mechanism (residual) with the output of the
    cross-attention mechanism (`hidden_states`). It computes a weighted sum of
    the two inputs, where the weights are learned gates. The result is
    normalized using layer normalization.

    Args:
        hidden_dim: int, The hidden dimension size for the gate computation.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(self, hidden_dim, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.norm = keras.layers.LayerNormalization(
            epsilon=1e-5, name="norm", dtype=self.dtype_policy
        )
        self.gate = keras.layers.Dense(
            2 * self.hidden_dim,
            name="gate",
            dtype=self.dtype_policy,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

    def build(self, input_shape):
        batch_dim, seq_len_dim = None, None
        if input_shape and len(input_shape) == 3:
            batch_dim = input_shape[0]
            seq_len_dim = input_shape[1]
        gate_build_shape = (batch_dim, seq_len_dim, 2 * self.hidden_dim)
        self.gate.build(gate_build_shape)
        norm_build_shape = (batch_dim, seq_len_dim, self.hidden_dim)
        self.norm.build(norm_build_shape)
        super().build(input_shape)

    def call(self, second_residual, hidden_states, training=None):
        gate_input = keras.ops.concatenate(
            [second_residual, hidden_states], axis=-1
        )
        gates_linear_output = self.gate(gate_input)
        gates = keras.ops.sigmoid(gates_linear_output)
        gate_chunks = keras.ops.split(gates, 2, axis=-1)
        gate1 = gate_chunks[0]
        gate2 = gate_chunks[1]
        gated_sum = gate1 * second_residual + gate2 * hidden_states
        hidden_states = self.norm(gated_sum, training=training)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


class DFineMLP(keras.layers.Layer):
    """Multi-layer perceptron (MLP) layer.

    This layer implements a standard MLP. It is used in several places within
    the D-FINE model, such as the `reg_conf` head inside `DFineLQE` for
    predicting quality scores and the `pre_bbox_head` in `DFineDecoder` for
    initial bounding box predictions.

    Args:
        input_dim: int, The input dimension.
        hidden_dim: int, The hidden dimension for intermediate layers.
        output_dim: int, The output dimension.
        num_layers: int, The number of layers in the MLP.
        activation_function: str, The activation function to use between layers.
        kernel_initializer: str or Initializer, optional, Initializer for
            the kernel weights. Defaults to `"glorot_uniform"`.
        bias_initializer: str or Initializer, optional, Initializer for
            the bias weights. Defaults to `"zeros"`.
        last_layer_initializer: str or Initializer, optional, Special
            initializer for the final layer's weights and biases. If `None`,
            uses `kernel_initializer` and `bias_initializer`. Defaults to
            `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        activation_function="relu",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        last_layer_initializer=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        # NOTE: In the original code, this is done by searching the modules for
        # specific last layers, instead, we find the last layer in each of the
        # specific modules with `num_layers - 1`.
        self.last_layer_initializer = keras.initializers.get(
            last_layer_initializer
        )
        h = [hidden_dim] * (num_layers - 1)
        input_dims = [input_dim] + h
        output_dims = h + [output_dim]
        self.dense_layers = []
        for i, (_, out_dim) in enumerate(zip(input_dims, output_dims)):
            # NOTE: Req. for handling the case of initializing the final layers'
            # weights and biases to zero when required (for ex: `bbox_embed` or
            # `reg_conf`).
            is_last_layer = i == num_layers - 1
            current_kernel_init = self.kernel_initializer
            current_bias_init = self.bias_initializer
            if is_last_layer and self.last_layer_initializer is not None:
                current_kernel_init = self.last_layer_initializer
                current_bias_init = self.last_layer_initializer
            self.dense_layers.append(
                keras.layers.Dense(
                    units=out_dim,
                    name=f"mlp_dense_layer_{i}",
                    dtype=self.dtype_policy,
                    kernel_initializer=current_kernel_init,
                    bias_initializer=current_bias_init,
                )
            )
        self.activation_layer = keras.layers.Activation(
            activation_function,
            name="mlp_activation_layer",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        if self.dense_layers:
            current_build_shape = input_shape
            for i, dense_layer in enumerate(self.dense_layers):
                dense_layer.build(current_build_shape)
                current_build_shape = dense_layer.compute_output_shape(
                    current_build_shape
                )
        super().build(input_shape)

    def call(self, stat_features, training=None):
        x = stat_features
        for i in range(self.num_layers):
            dense_layer = self.dense_layers[i]
            x = dense_layer(x)
            if i < self.num_layers - 1:
                x = self.activation_layer(x)
        return x

    def compute_output_spec(self, stat_features_spec):
        output_shape = list(stat_features_spec.shape)
        output_shape[-1] = self.output_dim
        return keras.KerasTensor(
            shape=tuple(output_shape), dtype=self.compute_dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "activation_function": self.activation_function,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "last_layer_initializer": keras.initializers.serialize(
                    self.last_layer_initializer
                ),
            }
        )
        return config


class DFineSourceFlattener(keras.layers.Layer):
    """Layer to flatten and concatenate a list of source tensors.

    This layer is used in `DFineBackbone` to process feature maps from the
    `DFineHybridEncoder`. It takes a list of multi-scale feature maps,
    flattens each along its spatial dimensions, and concatenates them
    along the sequence dimension.

    Args:
        channel_axis: int, optional, The channel axis. Defaults to `None`.
        data_format: str, optional, The data format. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self, channel_axis=None, data_format=None, dtype=None, **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.channel_axis = channel_axis
        self.data_format = data_format

    def call(self, sources, training=None):
        source_flatten = []
        for i, source_item in enumerate(sources):
            if self.data_format == "channels_first":
                source_item = keras.ops.transpose(source_item, [0, 2, 3, 1])
            batch_size = keras.ops.shape(source_item)[0]
            channels = keras.ops.shape(source_item)[-1]
            source_reshaped = keras.ops.reshape(
                source_item, (batch_size, -1, channels)
            )
            source_flatten.append(source_reshaped)
        source_flatten_concatenated = keras.ops.concatenate(
            source_flatten, axis=1
        )
        return source_flatten_concatenated

    def compute_output_shape(self, sources_shape):
        if not sources_shape or not isinstance(sources_shape, list):
            return tuple()
        if not all(isinstance(s, tuple) and len(s) == 4 for s in sources_shape):
            return tuple()
        batch_size = sources_shape[0][0]
        if self.data_format == "channels_first":
            channels = sources_shape[0][1]
        else:
            channels = sources_shape[0][-1]
        calculated_spatial_elements = []
        for s_shape in sources_shape:
            if self.data_format == "channels_first":
                h, w = s_shape[2], s_shape[3]
            else:
                h, w = s_shape[1], s_shape[2]
            if h is None or w is None:
                calculated_spatial_elements.append(None)
            else:
                calculated_spatial_elements.append(h * w)
        if any(elem is None for elem in calculated_spatial_elements):
            total_spatial_elements = None
        else:
            total_spatial_elements = sum(calculated_spatial_elements)
        return (batch_size, total_spatial_elements, channels)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channel_axis": self.channel_axis,
                "data_format": self.data_format,
            }
        )
        return config


class DFineContrastiveDenoisingGroupGenerator(keras.layers.Layer):
    """Layer to generate denoising groups for contrastive learning.

    This layer, used in `DFineBackbone`, implements the core logic for
    contrastive denoising, a key training strategy in D-FINE. It takes ground
    truth `targets`, adds controlled noise to labels and boxes, and generates
    the necessary attention masks, queries, and reference points for the
    decoder. Due to functional model constraints, noise is generated once at
    model initialization.

    Args:
        num_labels: int, The number of object classes.
        num_denoising: int, The number of denoising queries.
        label_noise_ratio: float, The ratio of label noise to apply.
        box_noise_scale: float, The scale of box noise to apply.
        seed: int, optional, The random seed for noise generation.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        num_labels,
        num_denoising,
        label_noise_ratio,
        box_noise_scale,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_labels = num_labels
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, targets, num_queries):
        if self.num_denoising <= 0:
            return None, None, None, None
        num_ground_truths = [len(t["labels"]) for t in targets]
        max_gt_num = 0
        if num_ground_truths:
            max_gt_num = max(num_ground_truths)
        if max_gt_num == 0:
            return None, None, None, None
        num_groups_denoising_queries = self.num_denoising // max_gt_num
        num_groups_denoising_queries = (
            1
            if num_groups_denoising_queries == 0
            else num_groups_denoising_queries
        )
        batch_size = len(num_ground_truths)
        input_query_class = []
        input_query_bbox = []
        pad_gt_mask = []
        for i in range(batch_size):
            num_gt = num_ground_truths[i]
            if num_gt > 0:
                labels = targets[i]["labels"]
                boxes = targets[i]["boxes"]
                padded_class_labels = keras.ops.pad(
                    labels,
                    [[0, max_gt_num - num_gt]],
                    constant_values=self.num_labels,
                )
                padded_boxes = keras.ops.pad(
                    keras.ops.cast(boxes, dtype=self.compute_dtype),
                    [[0, max_gt_num - num_gt], [0, 0]],
                    constant_values=0.0,
                )
                mask = keras.ops.concatenate(
                    [
                        keras.ops.ones([num_gt], dtype="bool"),
                        keras.ops.zeros([max_gt_num - num_gt], dtype="bool"),
                    ]
                )
            else:
                padded_class_labels = keras.ops.full(
                    [max_gt_num], self.num_labels, dtype="int32"
                )
                padded_boxes = keras.ops.zeros(
                    [max_gt_num, 4], dtype=self.compute_dtype
                )
                mask = keras.ops.zeros([max_gt_num], dtype="bool")
            input_query_class.append(padded_class_labels)
            input_query_bbox.append(padded_boxes)
            pad_gt_mask.append(mask)
        input_query_class = keras.ops.stack(input_query_class, axis=0)
        input_query_bbox = keras.ops.stack(input_query_bbox, axis=0)
        pad_gt_mask = keras.ops.stack(pad_gt_mask, axis=0)
        input_query_class = keras.ops.tile(
            input_query_class, [1, 2 * num_groups_denoising_queries]
        )
        input_query_bbox = keras.ops.tile(
            input_query_bbox, [1, 2 * num_groups_denoising_queries, 1]
        )
        pad_gt_mask = keras.ops.tile(
            pad_gt_mask, [1, 2 * num_groups_denoising_queries]
        )
        negative_gt_mask = keras.ops.zeros(
            [batch_size, max_gt_num * 2, 1], dtype=self.compute_dtype
        )
        updates_neg = keras.ops.ones(
            [batch_size, max_gt_num, 1], dtype=negative_gt_mask.dtype
        )
        negative_gt_mask = keras.ops.slice_update(
            negative_gt_mask, [0, max_gt_num, 0], updates_neg
        )
        negative_gt_mask = keras.ops.tile(
            negative_gt_mask, [1, num_groups_denoising_queries, 1]
        )
        positive_gt_mask_float = 1.0 - negative_gt_mask
        squeezed_positive_gt_mask = keras.ops.squeeze(
            positive_gt_mask_float, axis=-1
        )
        positive_gt_mask = squeezed_positive_gt_mask * keras.ops.cast(
            pad_gt_mask, dtype=squeezed_positive_gt_mask.dtype
        )
        denoise_positive_idx = []
        for i in range(batch_size):
            mask_i = positive_gt_mask[i]
            idx = keras.ops.nonzero(mask_i)[0]
            denoise_positive_idx.append(idx)
        if self.label_noise_ratio > 0:
            noise_mask = keras.random.uniform(
                keras.ops.shape(input_query_class),
                dtype=self.compute_dtype,
                seed=self.seed_generator,
            ) < (self.label_noise_ratio * 0.5)
        max_len = 0
        for idx in denoise_positive_idx:
            current_len = keras.ops.shape(idx)[0]
            if current_len > max_len:
                max_len = current_len
        padded_indices = []
        for idx in denoise_positive_idx:
            current_len = keras.ops.shape(idx)[0]
            pad_len = max_len - current_len
            padded = keras.ops.pad(idx, [[0, pad_len]], constant_values=-1)
            padded_indices.append(padded)
        dn_positive_idx = (
            keras.ops.stack(padded_indices, axis=0) if padded_indices else None
        )
        if self.label_noise_ratio > 0:
            noise_mask = keras.ops.cast(noise_mask, "bool")
            new_label = keras.random.randint(
                keras.ops.shape(input_query_class),
                0,
                self.num_labels,
                seed=self.seed_generator,
                dtype="int32",
            )
            input_query_class = keras.ops.where(
                noise_mask & pad_gt_mask,
                new_label,
                input_query_class,
            )
        if self.box_noise_scale > 0:
            known_bbox = keras.utils.bounding_boxes.convert_format(
                input_query_bbox,
                source="center_xywh",
                target="xyxy",
                dtype=self.compute_dtype,
            )
            width_height = input_query_bbox[..., 2:]
            diff = (
                keras.ops.tile(width_height, [1, 1, 2])
                * 0.5
                * self.box_noise_scale
            )
            rand_int_sign = keras.random.randint(
                keras.ops.shape(input_query_bbox),
                0,
                2,
                seed=self.seed_generator,
            )
            rand_sign = (
                keras.ops.cast(rand_int_sign, dtype=diff.dtype) * 2.0 - 1.0
            )
            rand_part = keras.random.uniform(
                keras.ops.shape(input_query_bbox),
                seed=self.seed_generator,
                dtype=self.compute_dtype,
            )
            rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (
                1 - negative_gt_mask
            )
            rand_part = rand_part * rand_sign
            known_bbox = known_bbox + rand_part * diff
            known_bbox = keras.ops.clip(known_bbox, 0.0, 1.0)
            input_query_bbox = keras.utils.bounding_boxes.convert_format(
                known_bbox,
                source="xyxy",
                target="center_xywh",
                dtype=self.compute_dtype,
            )
        input_query_bbox = inverse_sigmoid(input_query_bbox)
        num_denoising_total = max_gt_num * 2 * num_groups_denoising_queries
        target_size = num_denoising_total + num_queries
        attn_mask = keras.ops.zeros(
            [target_size, target_size], dtype=self.compute_dtype
        )
        updates_attn1 = keras.ops.ones(
            [
                target_size - num_denoising_total,
                num_denoising_total,
            ],
            dtype=attn_mask.dtype,
        )
        attn_mask = keras.ops.slice_update(
            attn_mask, [num_denoising_total, 0], updates_attn1
        )
        for i in range(num_groups_denoising_queries):
            start = max_gt_num * 2 * i
            end = max_gt_num * 2 * (i + 1)
            updates_attn2 = keras.ops.ones(
                [end - start, start], dtype=attn_mask.dtype
            )
            attn_mask = keras.ops.slice_update(
                attn_mask, [start, 0], updates_attn2
            )
            updates_attn3 = keras.ops.ones(
                [end - start, num_denoising_total - end],
                dtype=attn_mask.dtype,
            )
            attn_mask = keras.ops.slice_update(
                attn_mask, [start, end], updates_attn3
            )
        if dn_positive_idx is not None:
            denoising_meta_values = {
                "dn_positive_idx": dn_positive_idx,
                "dn_num_group": keras.ops.convert_to_tensor(
                    num_groups_denoising_queries, dtype="int32"
                ),
                "dn_num_split": keras.ops.convert_to_tensor(
                    [num_denoising_total, num_queries], dtype="int32"
                ),
            }
        return (
            input_query_class,
            input_query_bbox,
            attn_mask,
            denoising_meta_values,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_labels": self.num_labels,
                "num_denoising": self.num_denoising,
                "label_noise_ratio": self.label_noise_ratio,
                "box_noise_scale": self.box_noise_scale,
                "seed": self.seed,
            }
        )
        return config


class DFineAnchorGenerator(keras.layers.Layer):
    """Layer to generate anchor boxes for object detection.

    This layer is used in `DFineBackbone` to generate anchor proposals. These
    anchors are combined with the output of the encoder's bounding box head
    (`enc_bbox_head`) to create initial reference points for the decoder's
    queries.

    Args:
        anchor_image_size: tuple, The size of the input image `(height, width)`.
        feat_strides: list, The strides of the feature maps.
        data_format: str, The data format of the image channels. Can be either
            `"channels_first"` or `"channels_last"`. If `None` is specified,
            it will use the `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json`. Defaults to `None`.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        anchor_image_size,
        feat_strides,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.anchor_image_size = anchor_image_size
        self.feat_strides = feat_strides
        self.data_format = data_format

    def call(self, sources_for_shape_derivation=None, grid_size=0.05):
        spatial_shapes = None
        if sources_for_shape_derivation is not None:
            if self.data_format == "channels_first":
                spatial_shapes = [
                    (keras.ops.shape(s)[2], keras.ops.shape(s)[3])
                    for s in sources_for_shape_derivation
                ]
            else:
                spatial_shapes = [
                    (keras.ops.shape(s)[1], keras.ops.shape(s)[2])
                    for s in sources_for_shape_derivation
                ]

        if spatial_shapes is None:
            spatial_shapes = [
                (
                    keras.ops.cast(self.anchor_image_size[0] / s, "int32"),
                    keras.ops.cast(self.anchor_image_size[1] / s, "int32"),
                )
                for s in self.feat_strides
            ]

        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = keras.ops.meshgrid(
                keras.ops.arange(height, dtype=self.compute_dtype),
                keras.ops.arange(width, dtype=self.compute_dtype),
                indexing="ij",
            )
            grid_xy = keras.ops.stack([grid_x, grid_y], axis=-1)
            grid_xy = keras.ops.expand_dims(grid_xy, axis=0) + 0.5
            grid_xy = grid_xy / keras.ops.array(
                [width, height], dtype=self.compute_dtype
            )
            wh = keras.ops.ones_like(grid_xy) * grid_size * (2.0**level)
            level_anchors = keras.ops.concatenate([grid_xy, wh], axis=-1)
            level_anchors = keras.ops.reshape(
                level_anchors, (-1, height * width, 4)
            )
            anchors.append(level_anchors)

        eps = 1e-2
        anchors = keras.ops.concatenate(anchors, axis=1)
        valid_mask = keras.ops.all(
            (anchors > eps) & (anchors < 1 - eps), axis=-1, keepdims=True
        )
        anchors_transformed = keras.ops.log(anchors / (1 - anchors))
        dtype_name = keras.backend.standardize_dtype(self.compute_dtype)
        if dtype_name == "float16":
            finfo_dtype = np.float16
        else:
            finfo_dtype = np.float32
        max_float = keras.ops.array(
            np.finfo(finfo_dtype).max, dtype=self.compute_dtype
        )
        anchors = keras.ops.where(valid_mask, anchors_transformed, max_float)

        return anchors, valid_mask

    def compute_output_shape(
        self, sources_for_shape_derivation_shape=None, grid_size_shape=None
    ):
        num_total_anchors_dim = None

        if sources_for_shape_derivation_shape is None:
            num_total_anchors_calc = 0
            for s_stride in self.feat_strides:
                h = self.anchor_image_size[0] // s_stride
                w = self.anchor_image_size[1] // s_stride
                num_total_anchors_calc += h * w
            num_total_anchors_dim = num_total_anchors_calc
        else:
            calculated_spatial_elements = []
            for s_shape in sources_for_shape_derivation_shape:
                if self.data_format == "channels_first":
                    h, w = s_shape[2], s_shape[3]
                else:
                    h, w = s_shape[1], s_shape[2]
                if h is None or w is None:
                    calculated_spatial_elements.append(None)
                else:
                    calculated_spatial_elements.append(h * w)
            if any(elem is None for elem in calculated_spatial_elements):
                num_total_anchors_dim = None
            else:
                num_total_anchors_dim = sum(calculated_spatial_elements)

        anchors_shape = (1, num_total_anchors_dim, 4)
        valid_mask_shape = (1, num_total_anchors_dim, 1)
        return anchors_shape, valid_mask_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "anchor_image_size": self.anchor_image_size,
                "feat_strides": self.feat_strides,
                "data_format": self.data_format,
            }
        )
        return config


class DFineSpatialShapesExtractor(keras.layers.Layer):
    """Layer to extract spatial shapes from input tensors.

    This layer is used in `DFineBackbone` to extract the spatial dimensions
    (height, width) from the multi-scale feature maps. The resulting shape
    tensor is passed to the `DFineDecoder` for use in deformable attention.

    Args:
        data_format: str, optional, The data format of the input tensors.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(self, data_format=None, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.data_format = data_format

    def call(self, sources):
        if self.data_format == "channels_first":
            spatial_shapes = [
                (keras.ops.shape(s)[2], keras.ops.shape(s)[3]) for s in sources
            ]
        else:
            spatial_shapes = [
                (keras.ops.shape(s)[1], keras.ops.shape(s)[2]) for s in sources
            ]
        spatial_shapes_tensor = keras.ops.array(spatial_shapes, dtype="int32")
        return spatial_shapes_tensor

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError("Expected a list of shape tuples")
        num_sources = len(input_shape)
        return (num_sources, 2)

    def get_config(self):
        config = super().get_config()
        config.update({"data_format": self.data_format})
        return config


class DFineInitialQueryAndReferenceGenerator(keras.layers.Layer):
    """Layer to generate initial queries and reference points for the decoder.

    This layer is a crucial component in `DFineBackbone` that bridges the
    encoder and decoder. It selects the top-k predictions from the encoder's
    output heads and uses them to generate the initial `target` (queries) and
    `reference_points` that are fed into the `DFineDecoder`.

    Args:
        num_queries: int, The number of queries to generate.
        hidden_dim: int, The hidden dimension of the model.
        learn_initial_query: bool, Whether to learn the initial query
            embeddings.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        num_queries,
        hidden_dim,
        learn_initial_query,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.learn_initial_query = learn_initial_query
        if self.learn_initial_query:
            self.query_indices_base = keras.ops.expand_dims(
                keras.ops.arange(self.num_queries, dtype="int32"), axis=0
            )
            self.weight_embedding = keras.layers.Embedding(
                input_dim=num_queries,
                output_dim=hidden_dim,
                name="weight_embedding",
                dtype=self.dtype_policy,
                embeddings_initializer="glorot_uniform",
            )
        else:
            self.weight_embedding = None

    def call(
        self,
        inputs,
        denoising_bbox_unact=None,
        denoising_class=None,
        training=None,
    ):
        (
            enc_outputs_class,
            enc_outputs_coord_logits_plus_anchors,
            output_memory,
            sources_last_element,
        ) = inputs
        enc_outputs_class_max = keras.ops.max(enc_outputs_class, axis=-1)
        topk_ind = keras.ops.top_k(
            enc_outputs_class_max, k=self.num_queries, sorted=True
        )[1]

        def gather_batch(elems):
            data, indices = elems
            return keras.ops.take(data, indices, axis=0)

        reference_points_unact = keras.ops.map(
            gather_batch, (enc_outputs_coord_logits_plus_anchors, topk_ind)
        )
        enc_topk_logits = keras.ops.map(
            gather_batch, (enc_outputs_class, topk_ind)
        )
        enc_topk_bboxes = keras.ops.sigmoid(reference_points_unact)

        if denoising_bbox_unact is not None:
            current_batch_size = keras.ops.shape(reference_points_unact)[0]
            denoising_bbox_unact = denoising_bbox_unact[:current_batch_size]
            if denoising_class is not None:
                denoising_class = denoising_class[:current_batch_size]
            reference_points_unact = keras.ops.concatenate(
                [denoising_bbox_unact, reference_points_unact], axis=1
            )
        if self.learn_initial_query:
            query_indices = self.query_indices_base
            target_embedding_val = self.weight_embedding(
                query_indices, training=training
            )
            batch_size = keras.ops.shape(sources_last_element)[0]
            target = keras.ops.tile(target_embedding_val, [batch_size, 1, 1])
        else:
            target = keras.ops.map(gather_batch, (output_memory, topk_ind))
            target = keras.ops.stop_gradient(target)

        if denoising_class is not None:
            target = keras.ops.concatenate([denoising_class, target], axis=1)
        init_reference_points = keras.ops.stop_gradient(reference_points_unact)
        return init_reference_points, target, enc_topk_logits, enc_topk_bboxes

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_queries": self.num_queries,
                "hidden_dim": self.hidden_dim,
                "learn_initial_query": self.learn_initial_query,
            }
        )
        return config

    def compute_output_spec(
        self,
        inputs,
        denoising_bbox_unact=None,
        denoising_class=None,
        training=None,
    ):
        (
            enc_outputs_class_spec,
            _,
            output_memory_spec,
            _,
        ) = inputs
        batch_size = enc_outputs_class_spec.shape[0]
        d_model_dim = output_memory_spec.shape[-1]
        num_labels_dim = enc_outputs_class_spec.shape[-1]
        num_queries_for_ref_points = self.num_queries
        if denoising_bbox_unact is not None:
            if len(denoising_bbox_unact.shape) > 1:
                if denoising_bbox_unact.shape[1] is not None:
                    num_queries_for_ref_points = (
                        denoising_bbox_unact.shape[1] + self.num_queries
                    )
                else:
                    num_queries_for_ref_points = None
        num_queries_for_target = self.num_queries
        if denoising_class is not None:
            if len(denoising_class.shape) > 1:
                if denoising_class.shape[1] is not None:
                    num_queries_for_target = (
                        denoising_class.shape[1] + self.num_queries
                    )
                else:
                    num_queries_for_target = None
        init_reference_points_spec = keras.KerasTensor(
            shape=(batch_size, num_queries_for_ref_points, 4),
            dtype=self.compute_dtype,
        )
        target_spec = keras.KerasTensor(
            shape=(batch_size, num_queries_for_target, d_model_dim),
            dtype=self.compute_dtype,
        )
        enc_topk_logits_spec = keras.KerasTensor(
            shape=(batch_size, self.num_queries, num_labels_dim),
            dtype=self.compute_dtype,
        )
        enc_topk_bboxes_spec = keras.KerasTensor(
            shape=(batch_size, self.num_queries, 4), dtype=self.compute_dtype
        )

        return (
            init_reference_points_spec,
            target_spec,
            enc_topk_logits_spec,
            enc_topk_bboxes_spec,
        )


class DFineIntegral(keras.layers.Layer):
    """Layer to compute integrated values from predicted corner probabilities.

    This layer implements the integral regression technique for bounding box
    prediction. It is used in `DFineDecoder` to transform the predicted
    distribution over bins (from `bbox_embed`) into continuous distance values,
    which are then used to calculate the final box coordinates.

    Args:
        max_num_bins: int, The maximum number of bins for the predictions.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(self, max_num_bins, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.max_num_bins = max_num_bins

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, pred_corners, project, training=None):
        original_shape = keras.ops.shape(pred_corners)
        batch_size = original_shape[0]
        num_queries = original_shape[1]
        reshaped_pred_corners = keras.ops.reshape(
            pred_corners, (-1, self.max_num_bins + 1)
        )
        softmax_output = keras.ops.softmax(reshaped_pred_corners, axis=1)
        linear_output = keras.ops.matmul(
            softmax_output, keras.ops.transpose(project)
        )
        squeezed_output = keras.ops.squeeze(linear_output, axis=-1)
        output_grouped_by_4 = keras.ops.reshape(squeezed_output, (-1, 4))
        final_output = keras.ops.reshape(
            output_grouped_by_4, (batch_size, num_queries, -1)
        )
        return final_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_num_bins": self.max_num_bins,
            }
        )
        return config


class DFineLQE(keras.layers.Layer):
    """Layer to compute quality scores for predictions.

    This layer, used within `DFineDecoder`, implements the Localization Quality
    Estimation (LQE) head. It computes a quality score from the distribution of
    predicted bounding box corners and adds this score to the classification
    logits, enhancing prediction confidence.

    Args:
        top_prob_values: int, The number of top probabilities to consider.
        max_num_bins: int, The maximum number of bins for the predictions.
        lqe_hidden_dim: int, The hidden dimension for the MLP.
        num_lqe_layers: int, The number of layers in the MLP.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        top_prob_values,
        max_num_bins,
        lqe_hidden_dim,
        num_lqe_layers,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.top_prob_values = top_prob_values
        self.max_num_bins = max_num_bins
        self.reg_conf = DFineMLP(
            input_dim=4 * (self.top_prob_values + 1),
            hidden_dim=lqe_hidden_dim,
            output_dim=1,
            num_layers=num_lqe_layers,
            dtype=self.dtype_policy,
            last_layer_initializer="zeros",
            name="reg_conf",
        )

    def build(self, input_shape):
        reg_conf_input_shape = (
            input_shape[0][0],
            input_shape[0][1],
            4 * (self.top_prob_values + 1),
        )
        self.reg_conf.build(reg_conf_input_shape)
        super().build(input_shape)

    def call(self, scores, pred_corners, training=None):
        original_shape = keras.ops.shape(pred_corners)
        batch_size = original_shape[0]
        length = original_shape[1]
        reshaped_pred_corners = keras.ops.reshape(
            pred_corners, (batch_size, length, 4, self.max_num_bins + 1)
        )
        prob = keras.ops.softmax(reshaped_pred_corners, axis=-1)
        prob_topk, _ = keras.ops.top_k(
            prob, k=self.top_prob_values, sorted=True
        )
        stat = keras.ops.concatenate(
            [prob_topk, keras.ops.mean(prob_topk, axis=-1, keepdims=True)],
            axis=-1,
        )
        reshaped_stat = keras.ops.reshape(stat, (batch_size, length, -1))
        quality_score = self.reg_conf(reshaped_stat, training=training)
        return scores + quality_score

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "top_prob_values": self.top_prob_values,
                "max_num_bins": self.max_num_bins,
                "lqe_hidden_dim": self.reg_conf.hidden_dim,
                "num_lqe_layers": self.reg_conf.num_layers,
            }
        )
        return config


class DFineConvNormLayer(keras.layers.Layer):
    """Convolutional layer with normalization and optional activation.

    This is a fundamental building block used in the CNN parts of D-FINE. It
    combines a `Conv2D` layer with `BatchNormalization` and an optional
    activation. It is used extensively in layers like `DFineRepVggBlock`,
    `DFineCSPRepLayer`, and within the `DFineHybridEncoder`.

    Args:
        filters: int, The number of output channels.
        kernel_size: int, The size of the convolutional kernel.
        batch_norm_eps: float, The epsilon value for batch normalization.
        stride: int, The stride of the convolution.
        groups: int, The number of groups for grouped convolution.
        padding: int or None, The padding to apply.
        activation_function: str or None, The activation function to use.
        kernel_initializer: str or Initializer, optional, Initializer for
            the kernel weights. Defaults to `"glorot_uniform"`.
        bias_initializer: str or Initializer, optional, Initializer for
            the bias weights. Defaults to `"zeros"`.
        channel_axis: int, optional, The channel axis. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        batch_norm_eps,
        stride,
        groups,
        padding,
        activation_function,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        channel_axis=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.batch_norm_eps = batch_norm_eps
        self.stride = stride
        self.groups = groups
        self.padding_arg = padding
        self.activation_function = activation_function
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.channel_axis = channel_axis
        if self.padding_arg is None:
            keras_conv_padding_mode = "same"
            self.explicit_padding_layer = None
        else:
            keras_conv_padding_mode = "valid"
            self.explicit_padding_layer = keras.layers.ZeroPadding2D(
                padding=self.padding_arg,
                name=f"{self.name}_explicit_padding",
                dtype=self.dtype_policy,
            )

        self.convolution = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=keras_conv_padding_mode,
            groups=self.groups,
            use_bias=False,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name=f"{self.name}_convolution",
        )
        self.normalization = keras.layers.BatchNormalization(
            epsilon=self.batch_norm_eps,
            name=f"{self.name}_normalization",
            axis=self.channel_axis,
            dtype=self.dtype_policy,
        )
        self.activation_layer = (
            keras.layers.Activation(
                self.activation_function,
                name=f"{self.name}_activation",
                dtype=self.dtype_policy,
            )
            if self.activation_function
            else keras.layers.Identity(
                name=f"{self.name}_identity_activation", dtype=self.dtype_policy
            )
        )

    def build(self, input_shape):
        if self.explicit_padding_layer:
            self.explicit_padding_layer.build(input_shape)
            shape = self.explicit_padding_layer.compute_output_shape(
                input_shape
            )
        else:
            shape = input_shape
        self.convolution.build(shape)
        conv_output_shape = self.convolution.compute_output_shape(shape)
        self.normalization.build(conv_output_shape)
        self.activation_layer.build(conv_output_shape)
        super().build(input_shape)

    def call(self, hidden_state, training=None):
        if self.explicit_padding_layer:
            hidden_state = self.explicit_padding_layer(hidden_state)
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state, training=training)
        hidden_state = self.activation_layer(hidden_state)
        return hidden_state

    def compute_output_shape(self, input_shape):
        shape = input_shape
        if self.explicit_padding_layer:
            shape = self.explicit_padding_layer.compute_output_shape(shape)
        return self.convolution.compute_output_shape(shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "batch_norm_eps": self.batch_norm_eps,
                "stride": self.stride,
                "groups": self.groups,
                "padding": self.padding_arg,
                "activation_function": self.activation_function,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "channel_axis": self.channel_axis,
            }
        )
        return config


class DFineRepVggBlock(keras.layers.Layer):
    """RepVGG-style block with two parallel convolutional paths.

    This layer implements a block inspired by the RepVGG architecture, featuring
    two parallel convolutional paths (3x3 and 1x1) that are summed. It serves
    as the core bottleneck block within the `DFineCSPRepLayer`.

    Args:
        activation_function: str, The activation function to use.
        filters: int, The number of output channels.
        batch_norm_eps: float, The epsilon value for batch normalization.
        kernel_initializer: str or Initializer, optional, Initializer for
            the kernel weights. Defaults to `"glorot_uniform"`.
        bias_initializer: str or Initializer, optional, Initializer for
            the bias weights. Defaults to `"zeros"`.
        channel_axis: int, optional, The channel axis. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        activation_function,
        filters,
        batch_norm_eps=1e-5,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        channel_axis=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.activation_function = activation_function
        self.filters = filters
        self.batch_norm_eps = batch_norm_eps
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.channel_axis = channel_axis
        self.conv1_layer = DFineConvNormLayer(
            filters=self.filters,
            kernel_size=3,
            batch_norm_eps=self.batch_norm_eps,
            stride=1,
            groups=1,
            padding=1,
            activation_function=None,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            channel_axis=self.channel_axis,
            name="conv1",
        )
        self.conv2_layer = DFineConvNormLayer(
            filters=self.filters,
            kernel_size=1,
            batch_norm_eps=self.batch_norm_eps,
            stride=1,
            groups=1,
            padding=0,
            activation_function=None,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            channel_axis=self.channel_axis,
            name="conv2",
        )
        self.activation_layer = (
            keras.layers.Activation(
                self.activation_function,
                name="block_activation",
                dtype=self.dtype_policy,
            )
            if self.activation_function
            else keras.layers.Identity(
                name="identity_activation", dtype=self.dtype_policy
            )
        )

    def build(self, input_shape):
        self.conv1_layer.build(input_shape)
        self.conv2_layer.build(input_shape)
        self.activation_layer.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        y1 = self.conv1_layer(x, training=training)
        y2 = self.conv2_layer(x, training=training)
        y = y1 + y2
        return self.activation_layer(y)

    def compute_output_shape(self, input_shape):
        return self.conv1_layer.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation_function": self.activation_function,
                "filters": self.filters,
                "batch_norm_eps": self.batch_norm_eps,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "channel_axis": self.channel_axis,
            }
        )
        return config


class DFineCSPRepLayer(keras.layers.Layer):
    """CSP (Cross Stage Partial) layer with repeated bottleneck blocks.

    This layer implements a Cross Stage Partial (CSP) block using
    `DFineRepVggBlock` as its bottleneck. It is a key component of the
    `DFineFeatureAggregationBlock` block, which forms the FPN/PAN structure in
    the `DFineHybridEncoder`.

    Args:
        activation_function: str, The activation function to use.
        batch_norm_eps: float, The epsilon value for batch normalization.
        filters: int, The number of output channels.
        num_blocks: int, The number of bottleneck blocks.
        expansion: float, The expansion factor for hidden channels. Defaults to
            `1.0`.
        kernel_initializer: str or Initializer, optional, Initializer for
            the kernel weights. Defaults to `"glorot_uniform"`.
        bias_initializer: str or Initializer, optional, Initializer for
            the bias weights. Defaults to `"zeros"`.
        channel_axis: int, optional, The channel axis. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        activation_function,
        batch_norm_eps,
        filters,
        num_blocks,
        expansion=1.0,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        channel_axis=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.activation_function = activation_function
        self.batch_norm_eps = batch_norm_eps
        self.filters = filters
        self.num_blocks = num_blocks
        self.expansion = expansion
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.channel_axis = channel_axis
        hidden_channels = int(self.filters * self.expansion)
        self.conv1 = DFineConvNormLayer(
            filters=hidden_channels,
            kernel_size=1,
            batch_norm_eps=self.batch_norm_eps,
            stride=1,
            groups=1,
            padding=0,
            activation_function=self.activation_function,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            channel_axis=self.channel_axis,
            name="conv1",
        )
        self.conv2 = DFineConvNormLayer(
            filters=hidden_channels,
            kernel_size=1,
            batch_norm_eps=self.batch_norm_eps,
            stride=1,
            groups=1,
            padding=0,
            activation_function=self.activation_function,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            channel_axis=self.channel_axis,
            name="conv2",
        )
        self.bottleneck_layers = [
            DFineRepVggBlock(
                activation_function=self.activation_function,
                filters=hidden_channels,
                batch_norm_eps=self.batch_norm_eps,
                dtype=self.dtype_policy,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                channel_axis=self.channel_axis,
                name=f"bottleneck_{i}",
            )
            for i in range(self.num_blocks)
        ]
        if hidden_channels != self.filters:
            self.conv3 = DFineConvNormLayer(
                filters=self.filters,
                kernel_size=1,
                batch_norm_eps=self.batch_norm_eps,
                stride=1,
                groups=1,
                padding=0,
                activation_function=self.activation_function,
                dtype=self.dtype_policy,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                channel_axis=self.channel_axis,
                name="conv3",
            )
        else:
            self.conv3 = keras.layers.Identity(
                name="conv3_identity", dtype=self.dtype_policy
            )

    def build(self, input_shape):
        self.conv1.build(input_shape)
        self.conv2.build(input_shape)
        bottleneck_input_shape = self.conv1.compute_output_shape(input_shape)
        for bottleneck_layer in self.bottleneck_layers:
            bottleneck_layer.build(bottleneck_input_shape)
        self.conv3.build(bottleneck_input_shape)
        super().build(input_shape)

    def call(self, hidden_state, training=None):
        hidden_state_1 = self.conv1(hidden_state, training=training)
        for bottleneck_layer in self.bottleneck_layers:
            hidden_state_1 = bottleneck_layer(hidden_state_1, training=training)
        hidden_state_2 = self.conv2(hidden_state, training=training)
        summed_hidden_states = hidden_state_1 + hidden_state_2
        if isinstance(self.conv3, keras.layers.Identity):
            hidden_state_3 = self.conv3(summed_hidden_states)
        else:
            hidden_state_3 = self.conv3(summed_hidden_states, training=training)
        return hidden_state_3

    def compute_output_shape(self, input_shape):
        shape_after_conv1 = self.conv1.compute_output_shape(input_shape)
        return self.conv3.compute_output_shape(shape_after_conv1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation_function": self.activation_function,
                "batch_norm_eps": self.batch_norm_eps,
                "filters": self.filters,
                "num_blocks": self.num_blocks,
                "expansion": self.expansion,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "channel_axis": self.channel_axis,
            }
        )
        return config


class DFineFeatureAggregationBlock(keras.layers.Layer):
    """Complex block combining convolutional and CSP layers.

    This layer implements a complex feature extraction block combining multiple
    convolutional and `DFineCSPRepLayer` layers. It is the main building block
    for the Feature Pyramid Network (FPN) and Path Aggregation Network (PAN)
    pathways within the `DFineHybridEncoder`.

    Args:
        encoder_hidden_dim: int, The hidden dimension of the encoder.
        hidden_expansion: float, The expansion factor for hidden channels.
        batch_norm_eps: float, The epsilon value for batch normalization.
        activation_function: str, The activation function to use.
        num_blocks: int, The number of blocks in the CSP layers.
        kernel_initializer: str or Initializer, optional, Initializer for
            the kernel weights. Defaults to `"glorot_uniform"`.
        bias_initializer: str or Initializer, optional, Initializer for
            the bias weights. Defaults to `"zeros"`.
        channel_axis: int, optional, The channel axis. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        encoder_hidden_dim,
        hidden_expansion,
        batch_norm_eps,
        activation_function,
        num_blocks,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        channel_axis=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_expansion = hidden_expansion
        self.batch_norm_eps = batch_norm_eps
        self.activation_function = activation_function
        self.num_blocks = num_blocks
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.channel_axis = channel_axis

        conv3_dim = self.encoder_hidden_dim * 2
        self.conv4_dim = int(
            self.hidden_expansion * self.encoder_hidden_dim / 2
        )
        self.conv_dim = conv3_dim // 2
        self.conv1 = DFineConvNormLayer(
            filters=conv3_dim,
            kernel_size=1,
            batch_norm_eps=self.batch_norm_eps,
            stride=1,
            groups=1,
            padding=0,
            activation_function=self.activation_function,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            channel_axis=self.channel_axis,
            name="conv1",
        )
        self.csp_rep1 = DFineCSPRepLayer(
            activation_function=self.activation_function,
            batch_norm_eps=self.batch_norm_eps,
            filters=self.conv4_dim,
            num_blocks=self.num_blocks,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            channel_axis=self.channel_axis,
            name="csp_rep1",
        )
        self.conv2 = DFineConvNormLayer(
            filters=self.conv4_dim,
            kernel_size=3,
            batch_norm_eps=self.batch_norm_eps,
            stride=1,
            groups=1,
            padding=1,
            activation_function=self.activation_function,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            channel_axis=self.channel_axis,
            name="conv2",
        )
        self.csp_rep2 = DFineCSPRepLayer(
            activation_function=self.activation_function,
            batch_norm_eps=self.batch_norm_eps,
            filters=self.conv4_dim,
            num_blocks=self.num_blocks,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            channel_axis=self.channel_axis,
            name="csp_rep2",
        )
        self.conv3 = DFineConvNormLayer(
            filters=self.conv4_dim,
            kernel_size=3,
            batch_norm_eps=self.batch_norm_eps,
            stride=1,
            groups=1,
            padding=1,
            activation_function=self.activation_function,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            channel_axis=self.channel_axis,
            name="conv3",
        )
        self.conv4 = DFineConvNormLayer(
            filters=self.encoder_hidden_dim,
            kernel_size=1,
            batch_norm_eps=self.batch_norm_eps,
            stride=1,
            groups=1,
            padding=0,
            activation_function=self.activation_function,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            channel_axis=self.channel_axis,
            name="conv4",
        )

    def build(self, input_shape):
        self.conv1.build(input_shape)
        shape_after_conv1 = self.conv1.compute_output_shape(input_shape)
        csp_rep_input_shape_list = list(shape_after_conv1)
        csp_rep_input_shape_list[self.channel_axis] = self.conv_dim
        csp_rep_input_shape = tuple(csp_rep_input_shape_list)
        self.csp_rep1.build(csp_rep_input_shape)
        shape_after_csp_rep1 = self.csp_rep1.compute_output_shape(
            csp_rep_input_shape
        )
        self.conv2.build(shape_after_csp_rep1)
        shape_after_conv2 = self.conv2.compute_output_shape(
            shape_after_csp_rep1
        )
        self.csp_rep2.build(shape_after_conv2)
        shape_after_csp_rep2 = self.csp_rep2.compute_output_shape(
            shape_after_conv2
        )
        self.conv3.build(shape_after_csp_rep2)
        shape_for_concat_list = list(shape_after_conv1)
        shape_for_concat_list[self.channel_axis] = (
            self.conv_dim * 2 + self.conv4_dim * 2
        )
        shape_for_concat = tuple(shape_for_concat_list)
        self.conv4.build(shape_for_concat)
        super().build(input_shape)

    def call(self, input_features, training=None):
        conv1_out = self.conv1(input_features, training=training)
        split_features_tensor = keras.ops.split(
            conv1_out, [self.conv_dim, self.conv_dim], axis=self.channel_axis
        )
        split_features = list(split_features_tensor)
        branch1 = self.csp_rep1(split_features[-1], training=training)
        branch1 = self.conv2(branch1, training=training)
        branch2 = self.csp_rep2(branch1, training=training)
        branch2 = self.conv3(branch2, training=training)
        split_features.extend([branch1, branch2])
        merged_features = keras.ops.concatenate(
            split_features, axis=self.channel_axis
        )
        merged_features = self.conv4(merged_features, training=training)
        return merged_features

    def compute_output_shape(self, input_shape):
        shape_after_conv1 = self.conv1.compute_output_shape(input_shape)
        shape_for_concat_list = list(shape_after_conv1)
        shape_for_concat_list[self.channel_axis] = (
            self.conv_dim * 2 + self.conv4_dim * 2
        )
        shape_for_concat = tuple(shape_for_concat_list)
        return self.conv4.compute_output_shape(shape_for_concat)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "hidden_expansion": self.hidden_expansion,
                "batch_norm_eps": self.batch_norm_eps,
                "activation_function": self.activation_function,
                "num_blocks": self.num_blocks,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "channel_axis": self.channel_axis,
            }
        )
        return config


class DFineSCDown(keras.layers.Layer):
    """Downsampling layer using convolutions.

    This layer is used in the `DFineHybridEncoder` to perform downsampling.
    Specifically, it is part of the Path Aggregation Network (PAN) bottom-up
    pathway, reducing the spatial resolution of feature maps.

    Args:
        encoder_hidden_dim: int, The hidden dimension of the encoder.
        batch_norm_eps: float, The epsilon value for batch normalization.
        kernel_size: int, The kernel size for the second convolution.
        stride: int, The stride for the second convolution.
        kernel_initializer: str or Initializer, optional, Initializer for
            the kernel weights. Defaults to `"glorot_uniform"`.
        bias_initializer: str or Initializer, optional, Initializer for
            the bias weights. Defaults to `"zeros"`.
        channel_axis: int, optional, The channel axis. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        encoder_hidden_dim,
        batch_norm_eps,
        kernel_size,
        stride,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        channel_axis=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.batch_norm_eps = batch_norm_eps
        self.conv2_kernel_size = kernel_size
        self.conv2_stride = stride
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.channel_axis = channel_axis
        self.conv1 = DFineConvNormLayer(
            filters=self.encoder_hidden_dim,
            kernel_size=1,
            batch_norm_eps=self.batch_norm_eps,
            stride=1,
            groups=1,
            padding=0,
            activation_function=None,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            channel_axis=self.channel_axis,
            name="conv1",
        )
        self.conv2 = DFineConvNormLayer(
            filters=self.encoder_hidden_dim,
            kernel_size=self.conv2_kernel_size,
            batch_norm_eps=self.batch_norm_eps,
            stride=self.conv2_stride,
            groups=self.encoder_hidden_dim,
            padding=(self.conv2_kernel_size - 1) // 2,
            activation_function=None,
            dtype=self.dtype_policy,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            channel_axis=self.channel_axis,
            name="conv2",
        )

    def build(self, input_shape):
        self.conv1.build(input_shape)
        shape_after_conv1 = self.conv1.compute_output_shape(input_shape)
        self.conv2.build(shape_after_conv1)
        super().build(input_shape)

    def call(self, input_features, training=None):
        x = self.conv1(input_features, training=training)
        x = self.conv2(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        shape_after_conv1 = self.conv1.compute_output_shape(input_shape)
        return self.conv2.compute_output_shape(shape_after_conv1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "batch_norm_eps": self.batch_norm_eps,
                "kernel_size": self.conv2_kernel_size,
                "stride": self.conv2_stride,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "channel_axis": self.channel_axis,
            }
        )
        return config


class DFineMLPPredictionHead(keras.layers.Layer):
    """MLP head for making predictions from feature vectors.

    This layer is a generic MLP used for various prediction tasks in D-FINE.
    It is used for the encoder's bounding box head (`enc_bbox_head` in
    `DFineBackbone`), the decoder's bounding box embedding (`bbox_embed` in
    `DFineDecoder`), and the query position head (`query_pos_head` in
    `DFineDecoder`).

    Args:
        input_dim: int, The input dimension.
        hidden_dim: int, The hidden dimension for intermediate layers.
        output_dim: int, The output dimension.
        num_layers: int, The number of layers in the MLP.
        kernel_initializer: str or Initializer, optional, Initializer for
            the kernel weights. Defaults to `"glorot_uniform"`.
        bias_initializer: str or Initializer, optional, Initializer for
            the bias weights. Defaults to `"zeros"`.
        last_layer_initializer: str or Initializer, optional, Special
            initializer for the final layer's weights and biases. If `None`,
            uses `kernel_initializer` and `bias_initializer`. Defaults to
            `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        last_layer_initializer=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.last_layer_initializer = keras.initializers.get(
            last_layer_initializer
        )

        h = [self.hidden_dim] * (self.num_layers - 1)
        input_dims = [self.input_dim] + h
        output_dims = h + [self.output_dim]

        self.dense_layers = []
        for i, (_, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last_layer = i == self.num_layers - 1
            current_kernel_init = self.kernel_initializer
            current_bias_init = self.bias_initializer
            if is_last_layer and self.last_layer_initializer is not None:
                current_kernel_init = self.last_layer_initializer
                current_bias_init = self.last_layer_initializer
            self.dense_layers.append(
                keras.layers.Dense(
                    units=out_dim,
                    name=f"linear_{i}",
                    dtype=self.dtype_policy,
                    kernel_initializer=current_kernel_init,
                    bias_initializer=current_bias_init,
                )
            )

    def build(self, input_shape):
        if self.dense_layers:
            current_build_shape = input_shape
            for i, dense_layer in enumerate(self.dense_layers):
                dense_layer.build(current_build_shape)
                current_build_shape = dense_layer.compute_output_shape(
                    current_build_shape
                )
        super().build(input_shape)

    def call(self, x, training=None):
        current_x = x
        for i, layer in enumerate(self.dense_layers):
            current_x = layer(current_x)
            if i < self.num_layers - 1:
                current_x = keras.ops.relu(current_x)
        return current_x

    def compute_output_spec(self, x_spec):
        output_shape = list(x_spec.shape)
        output_shape[-1] = self.output_dim
        return keras.KerasTensor(
            shape=tuple(output_shape), dtype=self.compute_dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "last_layer_initializer": keras.initializers.serialize(
                    self.last_layer_initializer
                ),
            }
        )
        return config
