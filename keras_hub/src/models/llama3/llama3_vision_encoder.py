import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.llama.llama_layernorm import LlamaLayerNorm


class GatedPositionalEmbedding(layers.Layer):
    """Gated positional embedding for Llama 3.2 Vision.

    Matches HuggingFace MllamaPrecomputedPositionEmbedding architecture.
    """

    def __init__(
        self,
        num_patches,
        hidden_dim,
        max_num_tiles=4,
        max_aspect_ratio_id=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.max_num_tiles = max_num_tiles
        self.max_aspect_ratio_id = max_aspect_ratio_id

        # Main position embedding (num_patches includes CLS token)
        self.embedding = self.add_weight(
            name="embedding",
            shape=(num_patches, hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        # Gate for positional embedding
        self.gate = self.add_weight(
            name="gate",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        # Tile position embedding - matches HuggingFace structure
        # Shape: (max_aspect_ratio_id+1, max_num_tiles*num_patches*hidden_dim)
        self.tile_embedding = layers.Embedding(
            input_dim=max_aspect_ratio_id + 1,
            output_dim=max_num_tiles * num_patches * hidden_dim,
            name="tile_embedding",
        )

    def build(self, input_shape):
        self.tile_embedding.build((None,))
        super().build(input_shape)

    def call(self, x, aspect_ratio_ids=None, tile_indices=None):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch*num_tiles, num_patches, hidden_dim)
            aspect_ratio_ids: Tensor of aspect ratio IDs (batch*num_tiles,)
            tile_indices: Tensor of tile indices (batch*num_tiles,)

        Returns:
            Tensor of same shape with position embeddings added.
        """
        # Apply gated position embedding (inverted gate like HuggingFace)
        gated_pos_embed = (1 - ops.tanh(self.gate)) * self.embedding
        x = x + gated_pos_embed

        # Apply tile position embeddings if aspect_ratio_ids provided
        if aspect_ratio_ids is not None:
            # Lookup tile embedding:
            # (batch*tiles,) -> (batch*tiles, max_tiles*num_patches*hidden)
            tile_embedding = self.tile_embedding(aspect_ratio_ids)

            # Reshape to (batch*tiles, max_tiles, num_patches, hidden_dim)
            batch_tiles = ops.shape(x)[0]
            tile_embedding = ops.reshape(
                tile_embedding,
                (
                    batch_tiles,
                    self.max_num_tiles,
                    self.num_patches,
                    self.hidden_dim,
                ),
            )

            # Select the correct tile embedding for each sample using
            # tile_indices
            if tile_indices is not None:
                # tile_indices: (batch*tiles,)
                # Use take_along_axis for efficiency instead of one_hot
                # Reshape indices for broadcasting: (batch*tiles, 1, 1, 1)
                indices = ops.reshape(tile_indices, (-1, 1, 1, 1))
                # tile_embedding: (batch*tiles, max_tiles, patches, hidden)
                # Select along max_tiles axis (axis=1)
                tile_embedding = ops.take_along_axis(
                    tile_embedding, indices, axis=1
                )
                # Result: (batch*tiles, 1, patches, hidden)
                # Squeeze to (batch*tiles, patches, hidden)
                tile_embedding = ops.squeeze(tile_embedding, axis=1)
            else:
                # Fallback to 0-th tile if no indices provided
                # (should not happen in correct usage)
                tile_embedding = tile_embedding[:, 0, :, :]

            # Apply gated tile embedding
            gated_tile_embed = ops.tanh(self.gate) * tile_embedding
            x = x + gated_tile_embed

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "hidden_dim": self.hidden_dim,
                "max_num_tiles": self.max_num_tiles,
                "max_aspect_ratio_id": self.max_aspect_ratio_id,
            }
        )
        return config


class AspectRatioEmbedding(layers.Layer):
    """Aspect ratio embedding for Llama 3.2 Vision.

    Matches HuggingFace MllamaPrecomputedAspectRatioEmbedding architecture.
    """

    def __init__(
        self,
        max_num_tiles,
        num_patches,
        hidden_dim,
        max_aspect_ratio_id=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_num_tiles = max_num_tiles
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.max_aspect_ratio_id = max_aspect_ratio_id

        # Embedding: (max_aspect_ratio_id + 1, max_num_tiles * hidden_dim)
        self.embedding = layers.Embedding(
            input_dim=max_aspect_ratio_id + 1,
            output_dim=max_num_tiles * hidden_dim,
            name="embedding",
        )
        self.gate = self.add_weight(
            name="gate",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )

    def build(self, input_shape):
        self.embedding.build((None,))
        super().build(input_shape)

    def call(self, x, aspect_ratio_ids=None, tile_indices=None):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch*num_tiles, num_patches, hidden_dim)
            aspect_ratio_ids: Tensor of aspect ratio IDs (batch*num_tiles,)
            tile_indices: Tensor of tile indices (batch*num_tiles,)

        Returns:
            Tensor of same shape with tile embeddings added.
        """
        if aspect_ratio_ids is None:
            return x

        # Lookup embeddings:
        # (batch*num_tiles,) -> (batch*num_tiles, max_num_tiles * hidden_dim)
        embeddings = self.embedding(aspect_ratio_ids)

        batch_tiles = ops.shape(x)[0]

        # Reshape to isolate max_tiles dimension
        # (batch*tiles, max_num_tiles, hidden_dim)
        embeddings = ops.reshape(
            embeddings, (batch_tiles, self.max_num_tiles, self.hidden_dim)
        )

        if tile_indices is not None:
            # Use take_along_axis for efficiency
            # indices: (batch*tiles, 1, 1) to match
            # (batch*tiles, max_tiles, hidden)
            indices = ops.reshape(tile_indices, (-1, 1, 1))

            # Select:
            # (batch*tiles, max_tiles, hidden) -> (batch*tiles, 1, hidden)
            embeddings = ops.take_along_axis(embeddings, indices, axis=1)
            embeddings = ops.squeeze(embeddings, axis=1)
        else:
            embeddings = embeddings[:, 0, :]

        # Reshape to (batch*tiles, 1, hidden_dim) for broadcasting
        embeddings = ops.reshape(embeddings, (batch_tiles, 1, self.hidden_dim))

        # Apply gate
        gated_embeddings = ops.tanh(self.gate) * embeddings

        return x + gated_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_num_tiles": self.max_num_tiles,
                "num_patches": self.num_patches,
                "hidden_dim": self.hidden_dim,
                "max_aspect_ratio_id": self.max_aspect_ratio_id,
            }
        )
        return config


class Llama3VisionTransformerLayer(layers.Layer):
    """Transformer layer with RMSNorm and SwiGLU for Llama 3.2 Vision."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        dropout=0.0,
        layer_norm_epsilon=1e-6,
        normalize_first=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.normalize_first = normalize_first

        # === Normalization (LlamaRMSNorm) ===
        self.norm1 = LlamaLayerNorm(
            epsilon=layer_norm_epsilon,
            name="layer_norm_1",
        )
        self.norm2 = LlamaLayerNorm(
            epsilon=layer_norm_epsilon,
            name="layer_norm_2",
        )

        # === Self Attention ===
        # We use the Keras MultiHeadAttention layer
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout,
            use_bias=False,  # Llama usually uses no bias in QKV
            name="self_attention",
        )
        self.dropout = layers.Dropout(dropout)

        # === MLP (SwiGLU) ===
        # SwiGLU uses 3 projections: gate, up, down
        self.gate_proj = layers.Dense(
            intermediate_dim,
            use_bias=False,
            name="mlp_gate_proj",
        )
        self.up_proj = layers.Dense(
            intermediate_dim,
            use_bias=False,
            name="mlp_up_proj",
        )
        self.down_proj = layers.Dense(
            hidden_dim,
            use_bias=False,
            name="mlp_down_proj",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, padding_mask=None):
        # Self Attention block
        residual = x
        if self.normalize_first:
            x = self.norm1(x)

        # Keras MultiHeadAttention returns (attention_output, attention_scores)
        # We need to handle the return signature of MultiHeadAttention
        # It returns just attention_output by default if
        # return_attention_scores is False (default)
        attn_output = self.attn(
            query=x,
            value=x,
            key=x,
            attention_mask=padding_mask,
        )
        attn_output = self.dropout(attn_output)
        x = residual + attn_output

        if not self.normalize_first:
            x = self.norm1(x)

        # MLP block (SwiGLU)
        residual = x
        if self.normalize_first:
            x = self.norm2(x)

        # SwiGLU: SiLU(gate) * up
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = ops.silu(gate) * up
        x = self.down_proj(x)

        x = residual + x

        if not self.normalize_first:
            x = self.norm2(x)

        return x


class Llama3VisionGlobalTransformerEncoder(Llama3VisionTransformerLayer):
    """Transformer encoder with gated residuals for Llama 3.2 Vision Global
    Layers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Gating parameters specific to Global layers
        self.gate_attn = self.add_weight(
            name="gate_attn",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        self.gate_ffn = self.add_weight(
            name="gate_ffn",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x, padding_mask=None):
        # === Self Attention Block ===
        residual = x
        if self.normalize_first:
            x = self.norm1(x)

        attn_output = self.attn(
            query=x,
            value=x,
            key=x,
            attention_mask=padding_mask,
        )
        attn_output = self.dropout(attn_output)

        # Apply gated residual connection (Global specific)
        attn_output = ops.tanh(self.gate_attn) * attn_output
        x = residual + attn_output

        if not self.normalize_first:
            x = self.norm1(x)

        # === Feedforward Block ===
        residual = x
        if self.normalize_first:
            x = self.norm2(x)

        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = ops.silu(gate) * up
        x = self.down_proj(x)

        # Apply gated residual connection (Global specific)
        x = ops.tanh(self.gate_ffn) * x
        x = residual + x

        if not self.normalize_first:
            x = self.norm2(x)

        return x


@keras_hub_export("keras_hub.models.Llama3VisionEncoder")
class Llama3VisionEncoder(keras.layers.Layer):
    """Vision encoder for the Llama 3.2 Vision model.

    This layer implements the MllamaVisionModel architecture with support for
    multi-tile images, gated positional embeddings, and two-stage encoding
    (local + global transformer layers).

    Args:
        hidden_dim: int. The size of the transformer hidden state.
        num_layers: int. The number of local transformer layers.
        num_heads: int. The number of attention heads.
        intermediate_dim: int. The output dimension of the feedforward network.
        patch_size: int. The size of each square image patch.
        image_size: int. The input image resolution. Defaults to `560`.
        num_channels: int. The number of input channels. Defaults to `3`.
        global_layers: int. Number of global encoder layers. Defaults to `8`.
        max_num_tiles: int. Maximum number of image tiles. Defaults to `4`.
        activation: str. The activation function. Defaults to `"gelu"`.
        dropout: float. Dropout rate. Defaults to `0.0`.
        layer_norm_epsilon: float. Layer norm epsilon. Defaults to `1e-6`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.

    Example:
    ```python
    encoder = keras_hub.models.Llama3VisionEncoder(
        hidden_dim=1280,
        num_layers=32,
        num_heads=16,
        intermediate_dim=5120,
        patch_size=14,
        image_size=560,
    )
    images = np.random.uniform(size=(1, 560, 560, 3))
    output = encoder(images)  # Shape: (1, num_patches, hidden_dim)
    ```
    """

    def __init__(
        self,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        patch_size,
        image_size=560,
        num_channels=3,
        global_layers=8,
        max_num_tiles=4,
        max_aspect_ratio_id=8,
        intermediate_layers_indices=None,
        activation="gelu",
        dropout=0.0,
        layer_norm_epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        # === Config ===
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.global_layers_count = global_layers
        self.max_num_tiles = max_num_tiles
        self.max_aspect_ratio_id = max_aspect_ratio_id
        # Default to empty list (no intermediate layer concatenation)
        # HF 11B uses [3, 7, 15, 23, 30], but must be explicitly specified
        if intermediate_layers_indices is None:
            intermediate_layers_indices = []
        self.intermediate_layers_indices = intermediate_layers_indices
        self.activation = activation
        self.dropout_rate = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        self.num_patches = (image_size // patch_size) ** 2

        # === Layers ===
        # Patch embedding (Conv2D)
        self.patch_embedding = layers.Conv2D(
            filters=hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            use_bias=False,
            name="patch_embedding",
        )

        # Class embedding (learnable CLS token)
        self.class_embedding = self.add_weight(
            name="class_embedding",
            shape=(hidden_dim,),
            initializer="zeros",
            trainable=True,
        )

        # Gated positional embedding with tile support
        self.gated_positional_embedding = GatedPositionalEmbedding(
            num_patches=self.num_patches + 1,  # +1 for CLS token
            hidden_dim=hidden_dim,
            max_num_tiles=max_num_tiles,
            max_aspect_ratio_id=max_aspect_ratio_id,
            name="gated_positional_embedding",
        )

        # Pre/Post tile positional embeddings
        self.pre_tile_positional_embedding = AspectRatioEmbedding(
            max_num_tiles=max_num_tiles,
            num_patches=self.num_patches + 1,
            hidden_dim=hidden_dim,
            max_aspect_ratio_id=max_aspect_ratio_id,
            name="pre_tile_positional_embedding",
        )
        self.post_tile_positional_embedding = AspectRatioEmbedding(
            max_num_tiles=max_num_tiles,
            num_patches=self.num_patches + 1,
            hidden_dim=hidden_dim,
            max_aspect_ratio_id=max_aspect_ratio_id,
            name="post_tile_positional_embedding",
        )

        # Local transformer layers
        self.transformer_layers = [
            Llama3VisionTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                normalize_first=True,
                name=f"transformer_layer_{i}",
            )
            for i in range(num_layers)
        ]

        # Global transformer layers
        self.global_transformer_layers = [
            Llama3VisionGlobalTransformerEncoder(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                normalize_first=True,
                name=f"global_transformer_layer_{i}",
            )
            for i in range(global_layers)
        ]

        # Layer norms (pre and post)
        self.layernorm_pre = LlamaLayerNorm(
            epsilon=layer_norm_epsilon,
            name="layernorm_pre",
        )
        self.layernorm_post = LlamaLayerNorm(
            epsilon=layer_norm_epsilon,
            name="layernorm_post",
        )

    def build(self, input_shape):
        # input_shape: (batch, tiles, H, W, C)
        # Flatten for patch embedding: (batch*tiles, H, W, C)
        if len(input_shape) == 5:
            patch_input_shape = (
                None,
                input_shape[-3],
                input_shape[-2],
                input_shape[-1],
            )
        else:
            patch_input_shape = input_shape

        self.patch_embedding.build(patch_input_shape)

        # Build transformers with CLS token included
        # Shape: (batch*tiles, patches, dim) or (batch, tiles*patches, dim)
        # Using (None, None, dim) handles both
        transformer_shape = (None, None, self.hidden_dim)

        for layer in self.transformer_layers:
            layer.build(transformer_shape)
        for layer in self.global_transformer_layers:
            layer.build(transformer_shape)

        self.layernorm_pre.build(transformer_shape)
        self.layernorm_post.build(transformer_shape)
        super().build(input_shape)

    def call(self, images, aspect_ratio_ids=None, aspect_ratio_mask=None):
        """Forward pass of the vision encoder.

        Args:
            images: Tensor of shape
                `(batch, num_tiles, height, width, channels)`.
            aspect_ratio_ids: Optional tensor of shape `(batch, num_tiles)`.
            aspect_ratio_mask: Optional tensor of shape `(batch, num_tiles)`.

        Returns:
            Tensor of shape `(batch, num_tiles * num_patches, hidden_dim)`.
        """
        shape = ops.shape(images)
        batch_size = shape[0]
        num_tiles = shape[1]

        # Flatten tiles for local processing: (B, T, H, W, C) -> (B*T, H, W, C)
        flattened_images = ops.reshape(
            images, (-1, self.image_size, self.image_size, self.num_channels)
        )

        # Flatten aspect_ratio_ids: (B, T) -> (B*T,)
        if aspect_ratio_ids is not None:
            aspect_ratio_ids = ops.reshape(aspect_ratio_ids, (-1,))

        # Create tile indices: (Batch, Tiles) -> (Batch*Tiles,)
        tile_indices = None
        if aspect_ratio_ids is not None:
            # Assuming linear 0..T-1 indexing for tiles
            tile_range = ops.arange(num_tiles, dtype="int32")
            # Clamp tile range to max_num_tiles - 1 to avoid index errors
            tile_range = ops.minimum(tile_range, self.max_num_tiles - 1)
            tile_indices = ops.tile(tile_range[None, :], (batch_size, 1))
            tile_indices = ops.reshape(tile_indices, (-1,))

        # Patch embedding: (B*T, H, W, C) -> (B*T, patches, hidden_dim)
        embeddings = self.patch_embedding(flattened_images)
        embeddings = ops.reshape(
            embeddings, (batch_size * num_tiles, -1, self.hidden_dim)
        )

        # Add CLS token
        # (B*T, 1, D)
        cls_token = ops.broadcast_to(
            self.class_embedding, (batch_size * num_tiles, 1, self.hidden_dim)
        )
        embeddings = ops.concatenate([cls_token, embeddings], axis=1)

        # Pre-tile positional embedding
        embeddings = self.pre_tile_positional_embedding(
            embeddings, aspect_ratio_ids, tile_indices
        )

        # Gated positional embedding
        embeddings = self.gated_positional_embedding(
            embeddings, aspect_ratio_ids, tile_indices
        )

        # Local transformer layers + collect intermediate outputs
        intermediate_outputs = []
        for i, layer in enumerate(self.transformer_layers):
            embeddings = layer(embeddings)
            # Collect outputs at specified indices
            if i in self.intermediate_layers_indices:
                intermediate_outputs.append(embeddings)

        # Post-tile positional embedding
        embeddings = self.post_tile_positional_embedding(
            embeddings, aspect_ratio_ids, tile_indices
        )

        # Pre layer norm for global layers
        embeddings = self.layernorm_pre(embeddings)

        # === Global Processing ===
        # Reshape to (B, T*P, D) for global attention
        # patches_per_tile includes CLS token
        patches_per_tile = ops.shape(embeddings)[1]
        embeddings = ops.reshape(
            embeddings,
            (batch_size, num_tiles * patches_per_tile, self.hidden_dim),
        )

        # Create global padding mask if aspect_ratio_mask is provided
        global_padding_mask = None
        if aspect_ratio_mask is not None:
            # mask: (B, T) -> (B, T, 1) -> (B, T, P) -> (B, T*P)
            mask = ops.expand_dims(aspect_ratio_mask, axis=-1)
            mask = ops.tile(mask, [1, 1, patches_per_tile])
            global_padding_mask = ops.reshape(
                mask, (batch_size, num_tiles * patches_per_tile)
            )
            # Expand to (B, 1, 1, T*P) for attention broadcasting (B, H, Q, K)
            global_padding_mask = ops.expand_dims(global_padding_mask, axis=1)
            global_padding_mask = ops.expand_dims(global_padding_mask, axis=1)
            # Keras attention expects bool or int. Mllama uses 0 for padding?
            # Usually Keras padding_mask: True/1 for valid, False/0 for pad?
            # Mllama uses 1 for valid, 0 for pad. KerasHub TransformerEncoder
            # supports this.

        # Global transformer layers
        for layer in self.global_transformer_layers:
            embeddings = layer(embeddings, padding_mask=global_padding_mask)

        # Post layer norm
        embeddings = self.layernorm_post(embeddings)

        # Concatenate intermediate outputs + final output
        # HF: [layer_3, layer_7, layer_15, layer_23, layer_30, final]
        # Intermediate outputs are (B*T, P, D). Need to flatten final to match?
        # OR reshape intermediates to (B, T*P, D)?
        # The output of the backbone expects combined features.
        # But wait, intermediate_outputs are "local" features.
        # HF Mllama "vision_output" concatenates [intermediate... + final].
        # Are intermediate features reshaped to global?
        # Yes, MllamaVisionModel returns specific hidden states.
        # We need to reshape intermediate_outputs to (B, T*P, D).

        if intermediate_outputs:
            reshaped_intermediates = []
            for out in intermediate_outputs:
                out = ops.reshape(
                    out,
                    (batch_size, num_tiles * patches_per_tile, self.hidden_dim),
                )
                reshaped_intermediates.append(out)

            reshaped_intermediates.append(embeddings)
            # Concat along feature dim: (batch, T*P, (N+1)*D)
            embeddings = ops.concatenate(reshaped_intermediates, axis=-1)

        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "num_channels": self.num_channels,
                "global_layers": self.global_layers_count,
                "max_num_tiles": self.max_num_tiles,
                "max_aspect_ratio_id": self.max_aspect_ratio_id,
                "intermediate_layers_indices": self.intermediate_layers_indices,
                "activation": self.activation,
                "dropout": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def freeze_local_encoder(self):
        """Freeze local encoder layers."""
        self.patch_embedding.trainable = False
        for layer in self.transformer_layers:
            layer.trainable = False

    def freeze_global_encoder(self):
        """Freeze global encoder layers."""
        for layer in self.global_transformer_layers:
            layer.trainable = False
        self.layernorm_pre.trainable = False
        self.layernorm_post.trainable = False

    def freeze_all(self):
        """Freeze all encoder weights."""
        self.trainable = False

    def unfreeze_all(self):
        """Unfreeze all encoder weights."""
        self.trainable = True
