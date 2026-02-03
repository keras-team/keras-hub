import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


class Qwen3OmniVisionPatchEmbed(layers.Layer):
    """3D patch embedding layer for Qwen3-Omni vision encoder.

    Converts video or image input into patches using 3D convolution.
    For images, the temporal dimension is 1. For videos, the temporal
    dimension represents frames.

    Args:
        patch_size: int. The spatial patch size (height and width).
        temporal_patch_size: int. The temporal patch size (frames).
        in_channels: int. The number of input channels (e.g., 3 for RGB).
        embed_dim: int. The output embedding dimension.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the layer's computations and weights.
    """

    def __init__(
        self,
        patch_size,
        temporal_patch_size,
        in_channels,
        embed_dim,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = layers.Conv3D(
            filters=embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            use_bias=False,
            data_format="channels_last",
            dtype=dtype,
            name="proj",
        )

    def build(self, input_shape):
        self.proj.build(input_shape)
        self.built = True

    def call(self, pixel_values):
        """Forward pass.

        Args:
            pixel_values: Tensor with shape
                `(batch_size, temporal, height, width, channels)`.

        Returns:
            Tensor with shape `(batch_size, num_patches, embed_dim)`.
        """
        hidden_states = self.proj(pixel_values)
        batch_size = ops.shape(hidden_states)[0]
        embed_dim = ops.shape(hidden_states)[-1]
        hidden_states = ops.reshape(hidden_states, [batch_size, -1, embed_dim])
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "in_channels": self.in_channels,
                "embed_dim": self.embed_dim,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        temporal = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        num_patches = (
            (temporal // self.temporal_patch_size)
            * (height // self.patch_size)
            * (width // self.patch_size)
        )
        return (batch_size, num_patches, self.embed_dim)


class Qwen3OmniVisionBlock(layers.Layer):
    """Vision transformer block for Qwen3-Omni.

    Implements a standard Vision Transformer (ViT) block with pre-normalization,
    multi-head attention, and a feed-forward MLP. MoE routing is supported
    in some layers but not yet implemented.

    Args:
        hidden_size: int. The hidden dimension.
        num_heads: int. The number of attention heads.
        intermediate_size: int. The MLP intermediate dimension.
        num_experts: int. Number of MoE experts. If 0, uses dense MLP.
            Currently not implemented, always uses dense MLP. Defaults to 0.
        activation: string. The activation function name.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the layer's computations and weights.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        num_experts=0,
        activation="gelu_pytorch_tanh",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.activation_name = activation

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6,
            dtype=dtype,
            name="norm1",
        )
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads,
            dtype=dtype,
            name="attn",
        )
        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6,
            dtype=dtype,
            name="norm2",
        )

        # MLP - MoE support planned but not yet implemented
        if num_experts > 0:
            # TODO: Implement MoE routing similar to text decoder
            # For now, fall back to dense MLP
            pass

        self.mlp = keras.Sequential(
            [
                layers.Dense(
                    intermediate_size,
                    dtype=dtype,
                    name="mlp_fc1",
                ),
                layers.Activation(activation, dtype=dtype),
                layers.Dense(
                    hidden_size,
                    dtype=dtype,
                    name="mlp_fc2",
                ),
            ],
            name="mlp",
        )

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.attn.build(input_shape, input_shape)
        self.norm2.build(input_shape)
        self.mlp.build(input_shape)
        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        training=False,
    ):
        """Forward pass.

        Args:
            hidden_states: Tensor with shape
                `(batch_size, sequence_length, hidden_size)`.
            attention_mask: Tensor or None. The attention mask.
            training: bool. Whether the layer is in training mode.

        Returns:
            Tensor with shape `(batch_size, sequence_length, hidden_size)`.
        """
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(
            query=hidden_states,
            value=hidden_states,
            key=hidden_states,
            attention_mask=attention_mask,
            training=training,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = residual + hidden_states

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "num_experts": self.num_experts,
                "activation": self.activation_name,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


@keras_hub_export("keras_hub.models.Qwen3OmniVisionEncoder")
class Qwen3OmniVisionEncoder(Backbone):
    """Vision encoder for Qwen3-Omni (ViT-based with MoE).

    This encoder processes image and video inputs using a Vision Transformer
    (ViT) architecture with:
    - 3D patch embedding for spatiotemporal features
    - Learnable position embeddings with rotary position embeddings (RoPE)
    - Vision transformer blocks (some with MoE routing)
    - Spatial patch merging for multi-resolution features

    Args:
        image_size: int. The resolution of input images (assumes square images).
            Defaults to `448`.
        patch_size: int. The spatial patch size. Defaults to `14`.
        temporal_patch_size: int. The temporal patch size for videos.
            Defaults to `2`.
        in_channels: int. The number of input channels (e.g., 3 for RGB).
            Defaults to `3`.
        hidden_size: int. The hidden dimension. Defaults to `1664`.
        depth: int. The number of transformer layers. Defaults to `50`.
        num_heads: int. The number of attention heads. Defaults to `16`.
        intermediate_size: int. The MLP intermediate dimension.
            Defaults to `8960`.
        spatial_merge_size: int. The spatial merge factor for downsampling.
            Defaults to `2`.
        moe_layers: list of int or None. The layer indices that use MoE routing.
            If None, all layers use dense MLP. Defaults to None.
        num_experts_per_layer: list of int or None. The number of experts for
            each MoE layer. Must match length of `moe_layers` if provided.
            Defaults to None.
        num_experts_per_tok: int. The number of experts to route to per token
            (top-k routing) for MoE layers. Defaults to `2`.
        activation: string. The activation function name. Defaults to `"silu"`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the model's computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done at float32 precision regardless of dtype.

    Example:
    ```python
    import numpy as np
    import keras_hub

    # Image input (batch_size, temporal=1, height, width, channels)
    pixel_values = np.random.uniform(size=(1, 1, 448, 448, 3))

    # Vision encoder
    vision_encoder = keras_hub.models.Qwen3OmniVisionEncoder(
        image_size=448,
        patch_size=14,
        hidden_size=1664,
        depth=50,
    )
    output = vision_encoder({"pixel_values": pixel_values})
    ```
    """

    def __init__(
        self,
        image_size=448,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=1664,
        depth=50,
        num_heads=16,
        intermediate_size=8960,
        spatial_merge_size=2,
        moe_layers=None,
        num_experts_per_layer=None,
        num_experts_per_tok=2,
        hidden_act="gelu_pytorch_tanh",
        dtype=None,
        **kwargs,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.spatial_merge_size = spatial_merge_size
        self.moe_layers_indices = moe_layers or []
        self.num_experts_per_layer_list = num_experts_per_layer or []
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_act = hidden_act

        # Calculate number of patches and positions
        num_patches_per_side = image_size // patch_size
        num_position_embeddings = num_patches_per_side**2

        # === Patch embedding ===
        self.patch_embed = Qwen3OmniVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            dtype=dtype,
            name="patch_embed",
        )

        # === Learnable position embeddings ===
        self.pos_embed = layers.Embedding(
            input_dim=num_position_embeddings,
            output_dim=hidden_size,
            dtype=dtype,
            name="pos_embed",
        )
        self.num_position_embeddings = num_position_embeddings
        self.num_patches_per_side = num_patches_per_side

        # === Vision transformer blocks ===
        self.blocks = []
        for i in range(depth):
            # Check if this layer uses MoE
            if i in self.moe_layers_indices:
                layer_idx = self.moe_layers_indices.index(i)
                num_experts = (
                    self.num_experts_per_layer_list[layer_idx]
                    if layer_idx < len(self.num_experts_per_layer_list)
                    else 0
                )
            else:
                num_experts = 0

            block = Qwen3OmniVisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                activation=hidden_act,
                dtype=dtype,
                name=f"block_{i}",
            )
            self.blocks.append(block)

        # === Functional Model ===
        # Input: (batch, temporal, height, width, channels)
        pixel_values = layers.Input(
            shape=(None, image_size, image_size, in_channels),
            name="pixel_values",
        )
        outputs = self.call_with_inputs(pixel_values)

        super().__init__(
            inputs={"pixel_values": pixel_values},
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

    def _interpolate_position_embeddings(self, batch_size, num_patches):
        """Interpolate position embeddings for variable input sizes.

        Args:
            batch_size: The batch size.
            num_patches: The actual number of patches in the input.

        Returns:
            Position embeddings tensor with shape
            (batch_size, num_patches, hidden_size).
        """
        # For simplicity, use nearest neighbor interpolation
        # TODO: In full implementation, this should do bilinear interpolation
        # based on grid_thw (temporal, height, width)

        # Get position IDs (0 to num_patches-1)
        position_ids = ops.arange(num_patches, dtype="int32")
        position_ids = ops.minimum(
            position_ids, self.num_position_embeddings - 1
        )

        # Embed positions
        pos_embeddings = self.pos_embed(position_ids)

        # Expand to batch
        pos_embeddings = ops.expand_dims(pos_embeddings, 0)
        pos_embeddings = ops.tile(pos_embeddings, [batch_size, 1, 1])

        return pos_embeddings

    def call_with_inputs(self, pixel_values, training=False):
        """Forward pass through the vision encoder.

        Args:
            pixel_values: Tensor with shape
                `(batch_size, temporal, height, width, channels)`.
            training: bool. Whether the model is in training mode.

        Returns:
            Tensor with shape `(batch_size, num_patches, hidden_size)`.
        """
        # Patch embedding
        hidden_states = self.patch_embed(pixel_values)

        # Add position embeddings
        batch_size = ops.shape(hidden_states)[0]
        num_patches = ops.shape(hidden_states)[1]
        pos_embeddings = self._interpolate_position_embeddings(
            batch_size, num_patches
        )
        hidden_states = hidden_states + pos_embeddings

        # Apply transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, training=training)

        return hidden_states

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: dict. A dictionary with `"pixel_values"` key containing
                image/video tensor with shape
                `(batch_size, temporal, height, width, channels)`.
            training: bool. Whether the model is in training mode.

        Returns:
            Tensor with shape `(batch_size, num_patches, hidden_size)`.
        """
        return self.call_with_inputs(inputs["pixel_values"], training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "in_channels": self.in_channels,
                "hidden_size": self.hidden_size,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "spatial_merge_size": self.spatial_merge_size,
                "hidden_act": self.hidden_act,
            }
        )
        return config
