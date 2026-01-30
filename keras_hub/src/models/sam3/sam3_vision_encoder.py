import numpy as np
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.sam3.sam3_layers import SAM3MLP
from keras_hub.src.models.sam3.sam3_layers import SAM3Embedding
from keras_hub.src.models.sam3.sam3_layers import SAM3RoPEAttention
from keras_hub.src.models.sam3.sam3_layers import SAM3SinePositionEmbedding
from keras_hub.src.models.sam3.sam3_utils import window_partition
from keras_hub.src.models.sam3.sam3_utils import window_unpartition


class SAM3ViTRotaryEmbedding(layers.Layer):
    def __init__(self, rope_theta, head_dim, end_x, end_y, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.rope_theta = float(rope_theta)
        self.head_dim = int(head_dim)
        self.end_x = int(end_x)
        self.end_y = int(end_y)
        self.scale = float(scale)

        # Ensure even dimension for proper axial splitting.
        if self.head_dim % 4 != 0:
            raise ValueError("Dimension must be divisible by 4 for axial RoPE")

    def build(self, input_shape):
        freqs = 1.0 / (
            self.rope_theta
            ** (
                np.arange(0, self.head_dim, 4)[: (self.head_dim // 4)]
                / self.head_dim
            )
        )
        flattened_indices = np.arange(self.end_x * self.end_y, dtype=np.int64)
        x_positions = (flattened_indices % self.end_x) * self.scale
        y_positions = (
            np.floor_divide(flattened_indices, self.end_x) * self.scale
        )
        freqs_x = np.outer(x_positions, freqs).astype(np.float32)
        freqs_y = np.outer(y_positions, freqs).astype(np.float32)
        inv_freq = np.concatenate([freqs_x, freqs_y], axis=-1)
        inv_freq = np.repeat(inv_freq, repeats=2, axis=-1)
        rope_embeddings_cos = np.cos(inv_freq)
        rope_embeddings_sin = np.sin(inv_freq)
        self.rope_embeddings_cos = self.add_weight(
            name="rope_embeddings_cos",
            shape=rope_embeddings_cos.shape,
            dtype=self.variable_dtype,
            trainable=False,
            initializer=rope_embeddings_cos,
        )
        self.rope_embeddings_sin = self.add_weight(
            name="rope_embeddings_sin",
            shape=rope_embeddings_sin.shape,
            dtype=self.variable_dtype,
            trainable=False,
            initializer=rope_embeddings_sin,
        )

    def call(self, inputs):
        return self.rope_embeddings_cos, self.rope_embeddings_sin

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rope_theta": self.rope_theta,
                "head_dim": self.head_dim,
                "end_x": self.end_x,
                "end_y": self.end_y,
                "scale": self.scale,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        embedding_shape = (self.end_x * self.end_y, self.head_dim)
        return (embedding_shape, embedding_shape)

    def load_own_variables(self, store):
        try:
            return super().load_own_variables(store)
        except ValueError:
            # `SAM3ViTRotaryEmbedding` has precomputed weights only. The issue
            # of the loading logic could be ignored.
            pass


class SAM3ViTLayer(layers.Layer):
    def __init__(
        self,
        image_shape,
        patch_size,
        hidden_dim,
        intermediate_dim,
        num_heads,
        hidden_activation="gelu",
        rope_theta=10000.0,
        window_size=0,
        rotary_scale=1.0,
        attention_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
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
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.num_heads = int(num_heads)
        self.hidden_activation = hidden_activation
        self.rope_theta = float(rope_theta)
        self.window_size = int(window_size)
        self.rotary_scale = float(rotary_scale)
        self.hidden_dropout_rate = float(hidden_dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.head_dim = self.hidden_dim // self.num_heads
        input_size = (
            self.image_shape[0] // self.patch_size,
            self.image_shape[1] // self.patch_size,
        )
        if self.window_size > 0 and (
            input_size[0] % self.window_size != 0
            or input_size[1] % self.window_size != 0
        ):
            raise ValueError(
                "Image size must be divisible by `patch_size` and "
                "`window_size` for windowed attention. "
                f"Received image size: {image_shape}, "
                f"patch_size: {patch_size}, window_size: {window_size}"
            )
        rotary_input_size = (
            input_size if window_size == 0 else (window_size, window_size)
        )

        self.layer_norm1 = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm1",
        )
        self.rotary_emb = SAM3ViTRotaryEmbedding(
            rope_theta=rope_theta,
            head_dim=self.head_dim,
            end_x=rotary_input_size[0],
            end_y=rotary_input_size[1],
            scale=self.rotary_scale,
            dtype=self.dtype_policy,
            name="rotary_emb",
        )
        self.attention = SAM3RoPEAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            attention_dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm2",
        )
        self.mlp = SAM3MLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            activation=self.hidden_activation,
            dropout_rate=self.hidden_dropout_rate,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.dropout = layers.Dropout(
            self.hidden_dropout_rate, dtype=self.dtype_policy, name="dropout"
        )

    def build(self, input_shape):
        self.input_hidden_dim = int(input_shape[-1])
        self.layer_norm1.build(input_shape)
        input_shape = self.layer_norm1.compute_output_shape(input_shape)
        self.rotary_emb.build(input_shape)
        input_shape_before_attention = input_shape
        if self.window_size > 0:
            input_shape = list(input_shape)
            input_shape = (
                None,
                self.window_size,
                self.window_size,
                input_shape[-1],
            )
        self.attention.build(input_shape)
        input_shape = self.attention.compute_output_shape(input_shape)
        if self.window_size > 0:
            input_shape = input_shape_before_attention
        self.layer_norm2.build(input_shape)
        self.mlp.build(input_shape)
        self.dropout.build(input_shape)

    def call(self, hidden_states, training=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states, training=training)
        if self.window_size > 0:
            height, width = (
                self.image_shape[0] // self.patch_size,
                self.image_shape[1] // self.patch_size,
            )
            # Partition into non-overlapping windows for efficient attention.
            hidden_states = window_partition(
                hidden_states,
                height,
                width,
                self.window_size,
                self.input_hidden_dim,
            )

        position_embeddings = self.rotary_emb(hidden_states, training=training)
        hidden_states = self.attention(
            hidden_states, position_embeddings, training=training
        )
        if self.window_size > 0:
            # Reverse window partition to restore original spatial layout.
            hidden_states = window_unpartition(
                hidden_states, height, width, self.window_size, self.hidden_dim
            )
        hidden_states = ops.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states, training=training)
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = ops.add(
            residual, self.dropout(hidden_states, training=training)
        )
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape,
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "hidden_activation": self.hidden_activation,
                "rope_theta": self.rope_theta,
                "window_size": self.window_size,
                "rotary_scale": self.rotary_scale,
                "attention_dropout_rate": self.attention_dropout_rate,
                "hidden_dropout_rate": self.hidden_dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class SAM3ViTEncoder(layers.Layer):
    def __init__(
        self,
        image_shape,
        patch_size,
        num_layers,
        hidden_dim,
        intermediate_dim,
        num_heads,
        pretrain_image_shape=(336, 336, 3),
        hidden_activation="gelu",
        rope_theta=100000.0,
        window_size=0,
        global_attn_indexes=None,
        attention_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
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
        self.hidden_activation = hidden_activation
        self.rope_theta = float(rope_theta)
        self.window_size = int(window_size)
        if global_attn_indexes is not None:
            self.global_attn_indexes = [int(i) for i in global_attn_indexes]
        else:
            self.global_attn_indexes = None
        self.pretrain_image_shape = (
            int(pretrain_image_shape[0]),
            int(pretrain_image_shape[1]),
            int(pretrain_image_shape[2]),
        )
        self.hidden_dropout_rate = float(hidden_dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        height = self.image_shape[0] // self.patch_size

        self.embeddings = SAM3Embedding(
            hidden_dim=self.hidden_dim,
            patch_size=self.patch_size,
            image_shape=self.image_shape,
            dropout_rate=self.hidden_dropout_rate,
            pretrain_image_shape=self.pretrain_image_shape,
            dtype=self.dtype_policy,
            name="embeddings",
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm",
        )
        self.layers = [
            SAM3ViTLayer(
                image_shape=self.image_shape,
                patch_size=self.patch_size,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                num_heads=self.num_heads,
                hidden_activation=self.hidden_activation,
                rope_theta=self.rope_theta,
                window_size=(
                    self.window_size if i not in self.global_attn_indexes else 0
                ),
                rotary_scale=(
                    1.0
                    if i not in self.global_attn_indexes
                    else float(self.window_size) / height
                ),
                attention_dropout_rate=self.attention_dropout_rate,
                hidden_dropout_rate=self.hidden_dropout_rate,
                layer_norm_epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name=f"layer_{i}",
            )
            for i in range(self.num_layers)
        ]

    def build(self, input_shape):
        self.embeddings.build(input_shape)
        input_shape = self.embeddings.compute_output_shape(input_shape)
        input_shape = list(input_shape)
        height = self.image_shape[0] // self.patch_size
        width = self.image_shape[1] // self.patch_size
        input_shape = [input_shape[0], height, width, self.hidden_dim]
        self.layer_norm.build(input_shape)
        for layer in self.layers:
            layer.build(input_shape)

    def call(self, pixel_values, training=None):
        hidden_states = self.embeddings(pixel_values, training=training)
        height = self.image_shape[0] // self.patch_size
        width = self.image_shape[1] // self.patch_size
        # Reshape to spatial format for windowed attention:
        # [batch_size, height, width, hidden_size]
        hidden_states = ops.reshape(
            hidden_states, (-1, height, width, self.hidden_dim)
        )
        hidden_states = self.layer_norm(hidden_states, training=training)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, training=training)

        # Reshape back to sequence format:
        # [batch_size, height*width, hidden_size]
        return ops.reshape(hidden_states, (-1, height * width, self.hidden_dim))

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
                "pretrain_image_shape": self.pretrain_image_shape,
                "hidden_activation": self.hidden_activation,
                "rope_theta": self.rope_theta,
                "window_size": self.window_size,
                "global_attn_indexes": self.global_attn_indexes,
                "attention_dropout_rate": self.attention_dropout_rate,
                "hidden_dropout_rate": self.hidden_dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        input_shape = self.embeddings.compute_output_shape(input_shape)
        return input_shape


class SAM3FPNLayer(layers.Layer):
    def __init__(self, input_dim, fpn_dim, scale_factor, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.fpn_dim = int(fpn_dim)
        self.scale_factor = float(scale_factor)

        # Build the upsampling/downsampling layers based on scale factor.
        if self.scale_factor == 4.0:
            self.scale_layers = [
                layers.Conv2DTranspose(
                    self.input_dim // 2,
                    kernel_size=2,
                    strides=2,
                    dtype=self.dtype_policy,
                    name="scale_layers_0",
                ),
                layers.Activation(
                    "gelu", dtype=self.dtype_policy, name="scale_layers_1"
                ),
                layers.Conv2DTranspose(
                    self.input_dim // 4,
                    kernel_size=2,
                    strides=2,
                    dtype=self.dtype_policy,
                    name="scale_layers_2",
                ),
            ]
        elif self.scale_factor == 2.0:
            self.scale_layers = [
                layers.Conv2DTranspose(
                    self.input_dim // 2,
                    kernel_size=2,
                    strides=2,
                    dtype=self.dtype_policy,
                    name="scale_layers_0",
                )
            ]
        elif self.scale_factor == 1.0:
            self.scale_layers = []
        elif self.scale_factor == 0.5:
            self.scale_layers = [
                layers.MaxPooling2D(
                    pool_size=2,
                    strides=2,
                    dtype=self.dtype_policy,
                    name="scale_layers_0",
                )
            ]
        else:
            raise ValueError(
                f"Unsupported scale factor: {self.scale_factor}. "
                "Supported scale factors are 4.0, 2.0, 1.0, and 0.5."
            )
        self.proj1 = layers.Conv2D(
            self.fpn_dim, kernel_size=1, dtype=self.dtype_policy, name="proj1"
        )
        self.pad = layers.ZeroPadding2D(
            padding=1, dtype=self.dtype_policy, name="pad"
        )
        self.proj2 = layers.Conv2D(
            self.fpn_dim, kernel_size=3, dtype=self.dtype_policy, name="proj2"
        )

    def build(self, input_shape):
        for layer in self.scale_layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self.proj1.build(input_shape)
        input_shape = self.proj1.compute_output_shape(input_shape)
        self.pad.build(input_shape)
        input_shape = self.pad.compute_output_shape(input_shape)
        self.proj2.build(input_shape)

    def call(self, inputs, training=None):
        hidden_states = inputs
        for layer in self.scale_layers:
            hidden_states = layer(hidden_states, training=training)
        hidden_states = self.proj1(hidden_states, training=training)
        hidden_states = self.pad(hidden_states, training=training)
        return self.proj2(hidden_states, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "fpn_dim": self.fpn_dim,
                "scale_factor": self.scale_factor,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for layer in self.scale_layers:
            output_shape = layer.compute_output_shape(output_shape)
        output_shape = self.proj1.compute_output_shape(output_shape)
        output_shape = self.pad.compute_output_shape(output_shape)
        return self.proj2.compute_output_shape(output_shape)


class SAM3VisionNeck(layers.Layer):
    def __init__(self, hidden_dim, fpn_hidden_dim, scale_factors, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.fpn_hidden_dim = int(fpn_hidden_dim)
        self.scale_factors = scale_factors

        self.position_encoding = SAM3SinePositionEmbedding(
            num_pos_feats=self.fpn_hidden_dim // 2,
            normalize=True,
            dtype=self.dtype_policy,
            name="position_encoding",
        )
        self.fpn_layers = [
            SAM3FPNLayer(
                input_dim=self.hidden_dim,
                fpn_dim=self.fpn_hidden_dim,
                scale_factor=scale,
                dtype=self.dtype_policy,
                name=f"fpn_layer_{i}",
            )
            for i, scale in enumerate(self.scale_factors)
        ]

    def build(self, input_shape):
        self.position_encoding.build()
        self.fpn_image_shapes = []
        for layer in self.fpn_layers:
            layer.build(input_shape)
            fpn_shape = layer.compute_output_shape(input_shape)
            self.fpn_image_shapes.append([int(fpn_shape[1]), int(fpn_shape[2])])

    def call(self, hidden_states, training=None):
        fpn_hidden_states = []
        fpn_position_encodings = []
        for i, layer in enumerate(self.fpn_layers):
            fpn_output = layer(hidden_states, training=training)
            fpn_hidden_states.append(fpn_output)
            height, width = self.fpn_image_shapes[i]
            pos_enc = self.position_encoding(
                fpn_output, height=height, width=width, training=training
            )
            fpn_position_encodings.append(pos_enc)
        return fpn_hidden_states, fpn_position_encodings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "fpn_hidden_dim": self.fpn_hidden_dim,
                "scale_factors": self.scale_factors,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        fpn_hidden_state_shapes = []
        for layer in self.fpn_layers:
            fpn_hidden_state_shapes.append(
                layer.compute_output_shape(input_shape)
            )
        # fpn_hidden_states and fpn_position_encodings have the same shapes.
        return fpn_hidden_state_shapes, fpn_hidden_state_shapes


@keras_hub_export("keras_hub.layers.SAM3VisionEncoder")
class SAM3VisionEncoder(layers.Layer):
    """A vision encoder for the Segment Anything Model 3 (SAM3).

    This layer implements a Vision Transformer (ViT) backbone followed by a
    Feature Pyramid Network (FPN) neck. It processes input images and produces
    multi-scale feature maps and their corresponding position encodings.

    Args:
        image_shape: tuple. The shape of the input image
            (height, width, channels).
        patch_size: int. The size of the patches to be extracted from the image.
        num_layers: int. The number of transformer layers in the ViT backbone.
        hidden_dim: int. The hidden dimension of the transformer layers.
        intermediate_dim: int. The dimension of the intermediate layer in the
            transformer's MLP.
        num_heads: int. The number of attention heads.
        fpn_hidden_dim: int. The hidden dimension of the FPN.
        fpn_scale_factors: list of floats. The scale factors for each level of
            the feature pyramid.
        pretrain_image_shape: tuple. The shape of the image used during
            pretraining, for position embedding interpolation. Defaults to
            `(336, 336, 3)`.
        hidden_activation: str. The activation function for the transformer
            layers. Defaults to `"gelu"`.
        rope_theta: float. The theta value for rotary position embeddings.
            Defaults to `10000.0`.
        window_size: int. The size of the window for windowed attention.
            Defaults to `0`.
        global_attn_indexes: list of ints. The indices of the layers that use
            global attention instead of windowed attention.
        attention_dropout_rate: float. The dropout rate for attention. Defaults
            to `0`.
        hidden_dropout_rate: float. The dropout rate for the MLP. Defaults to
            `0.0`.
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
        fpn_hidden_dim,
        fpn_scale_factors,
        pretrain_image_shape=(336, 336, 3),
        hidden_activation="gelu",
        rope_theta=10000.0,
        window_size=0,
        global_attn_indexes=None,
        attention_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
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
        self.fpn_hidden_dim = int(fpn_hidden_dim)
        self.fpn_scale_factors = fpn_scale_factors
        self.hidden_activation = hidden_activation
        self.rope_theta = float(rope_theta)
        self.window_size = int(window_size)
        if global_attn_indexes is not None:
            self.global_attn_indexes = [int(i) for i in global_attn_indexes]
        else:
            self.global_attn_indexes = None
        self.pretrain_image_shape = (
            int(pretrain_image_shape[0]),
            int(pretrain_image_shape[1]),
            int(pretrain_image_shape[2]),
        )
        self.hidden_dropout_rate = float(hidden_dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        self.backbone = SAM3ViTEncoder(
            image_shape=self.image_shape,
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_heads=self.num_heads,
            pretrain_image_shape=self.pretrain_image_shape,
            hidden_activation=self.hidden_activation,
            rope_theta=self.rope_theta,
            window_size=self.window_size,
            global_attn_indexes=self.global_attn_indexes,
            attention_dropout_rate=self.attention_dropout_rate,
            hidden_dropout_rate=self.hidden_dropout_rate,
            layer_norm_epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="backbone",
        )
        self.vision_neck = SAM3VisionNeck(
            hidden_dim=self.hidden_dim,
            fpn_hidden_dim=self.fpn_hidden_dim,
            scale_factors=self.fpn_scale_factors,
            dtype=self.dtype_policy,
            name="vision_neck",
        )

    def build(self, input_shape):
        self.backbone.build(input_shape)
        input_shape = self.backbone.compute_output_shape(input_shape)
        height = self.image_shape[0] // self.patch_size
        width = self.image_shape[1] // self.patch_size
        input_shape = (input_shape[0], height, width, input_shape[-1])
        self.vision_neck.build(input_shape)

    def call(self, pixel_values, training=None):
        hidden_states = self.backbone(pixel_values, training=training)
        height = self.image_shape[0] // self.patch_size
        width = self.image_shape[1] // self.patch_size
        spatial_hidden_states = ops.reshape(
            hidden_states, (-1, height, width, self.hidden_dim)
        )
        fpn_hidden_states, fpn_position_encodings = self.vision_neck(
            spatial_hidden_states, training=training
        )
        return fpn_hidden_states, fpn_position_encodings

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
                "fpn_hidden_dim": self.fpn_hidden_dim,
                "fpn_scale_factors": self.fpn_scale_factors,
                "pretrain_image_shape": self.pretrain_image_shape,
                "hidden_activation": self.hidden_activation,
                "rope_theta": self.rope_theta,
                "window_size": self.window_size,
                "global_attn_indexes": self.global_attn_indexes,
                "attention_dropout_rate": self.attention_dropout_rate,
                "hidden_dropout_rate": self.hidden_dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        input_shape = self.backbone.compute_output_shape(input_shape)
        height = self.image_shape[0] // self.patch_size
        width = self.image_shape[1] // self.patch_size
        input_shape = (input_shape[0], height, width, input_shape[-1])
        fpn_hidden_state_shapes, fpn_position_encoding_shapes = (
            self.vision_neck.compute_output_shape(input_shape)
        )
        return fpn_hidden_state_shapes, fpn_position_encoding_shapes
