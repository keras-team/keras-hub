import keras
from keras import layers
from keras import ops
import collections.abc

def window_partition(x, window_size):
    """Partition the input tensor into non-overlapping windows."""
    batch_size, height, width, channels = ops.shape(x)
    
    x = ops.reshape(
        x, 
        (
            batch_size, 
            height // window_size, 
            window_size, 
            width // window_size, 
            window_size, 
            channels
        )
    )
    
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = ops.reshape(
        x, (-1, window_size, window_size, channels)
    )
    return windows


def window_reverse(windows, window_size, height, width, channels):
    """Reverse window partitioning."""
    batch_size = ops.shape(windows)[0] // ((height // window_size) * (width // window_size))
    
    x = ops.reshape(
        windows,
        (
            batch_size,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            channels
        )
    )
    
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (batch_size, height, width, channels))
    return x


class DropPath(layers.Layer):
    """Drop paths (Stochastic Depth) per sample.
    
    This is an implementation of the paper "Deep Networks with Stochastic Depth",
    which randomly drops entire layers for regularization.
    
    Args:
        drop_prob: float, probability of dropping path.
    """

    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if self.drop_prob == 0.0 or not training:
            return x

        # Keep probability
        keep_prob = 1.0 - self.drop_prob
        
        # Create binary mask with shape [batch_size, 1, 1, 1]
        batch_size = ops.shape(x)[0]
        random_tensor = keep_prob + ops.random.uniform((batch_size, 1, 1, 1), dtype=x.dtype)
        binary_mask = ops.floor(random_tensor)
        
        # Scale output to preserve expected value
        output = x / keep_prob * binary_mask
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


class Mlp(layers.Layer):
    """MLP module for Transformer.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension.
        out_features: Output dimension.
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        
        self.fc1 = layers.Dense(hidden_features, name="fc1")
        self.act = keras.activations.gelu
        self.fc2 = layers.Dense(out_features, name="fc2")
        self.drop = layers.Dropout(dropout_rate) if dropout_rate > 0.0 else None

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop is not None:
            x = self.drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "out_features": self.out_features,
            "dropout_rate": self.dropout_rate,
        })
        return config


class WindowAttention(layers.Layer):
    """Window based multi-head self attention.
    
    Args:
        dim: Number of input channels
        window_size: Window size
        num_heads: Number of attention heads
        qkv_bias: Add bias to query, key, value projections
        attention_dropout: Attention dropout rate
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attention_dropout=0.,
        dropout=0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.dim = dim
        self.window_size = (
            window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear layers for Q, K, V
        self.qkv = layers.Dense(
            dim * 3, 
            use_bias=qkv_bias,
            name="qkv"
        )
        
        # Relative position encoding
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table"
        )
        
        # Get pair-wise relative position index
        coords = ops.stack(ops.meshgrid(
            ops.arange(self.window_size[0]),
            ops.arange(self.window_size[1])
        ))
        coords = ops.reshape(coords, [2, -1])
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = ops.transpose(relative_coords, [1, 2, 0])
        
        relative_coords = relative_coords + self.window_size[0] - 1
        relative_coords = relative_coords * (2 * self.window_size[0] - 1)
        relative_position_index = ops.sum(relative_coords, -1)
        
        self.relative_position_index = relative_position_index
        
        self.attn_drop = layers.Dropout(attention_dropout)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(dropout)

    def build(self, input_shape):
        self.num_windows = input_shape[0] // (
            self.window_size[0] * self.window_size[1]
        )
        super().build(input_shape)

    def call(self, x, mask=None):
        """Forward pass.
        
        Args:
            x: Input tensor with shape [batch*num_windows, window_size*window_size, dim].
            mask: Optional mask for shifted window attention.
            
        Returns:
            Output tensor with shape [batch*num_windows, window_size*window_size, dim].
        """
        B_, N, C = ops.shape(x)
        
        # QKV projection
        qkv = self.qkv(x)  # [B_, N, 3*C]
        
        # Calculate exact dimensions
        qkv_dim = ops.shape(qkv)[-1]
        dim_per_head = C // self.num_heads
        
        # Split QKV
        # This splits the last dimension into 3 equal parts
        chunk_size = qkv_dim // 3
        q = qkv[:, :, :chunk_size]
        k = qkv[:, :, chunk_size:2*chunk_size]
        v = qkv[:, :, 2*chunk_size:]
        
        # Reshape to separate heads
        q = ops.reshape(q, (B_, N, self.num_heads, dim_per_head))
        k = ops.reshape(k, (B_, N, self.num_heads, dim_per_head))
        v = ops.reshape(v, (B_, N, self.num_heads, dim_per_head))
        
        # Transpose to [B_, num_heads, N, head_dim]
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))
        
        # Scale query
        q = q * self.scale
        
        # Compute attention scores
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        
        # Add relative position bias
        relative_position_bias = ops.take(
            self.relative_position_bias_table,
            self.relative_position_index,
        )
        
        relative_position_bias = ops.reshape(
            relative_position_bias,
            (self.window_size[0] * self.window_size[1], 
             self.window_size[0] * self.window_size[1], 
             self.num_heads)
        )
        
        # Transpose to [num_heads, Wh*Ww, Wh*Ww]
        relative_position_bias = ops.transpose(relative_position_bias, (2, 0, 1))
        
        # Add to attention [B_, num_heads, N, N]
        attn = attn + ops.expand_dims(relative_position_bias, axis=0)
        
        # Apply attention mask if provided
        if mask is not None:
            nW = mask.shape[0]  # num_windows
            # attn: [B_/nW, nW, num_heads, N, N]
            # mask: [1, nW, 1, N, N]
            attn = ops.reshape(attn, (-1, nW, self.num_heads, N, N))
            mask = ops.expand_dims(mask, axis=1)  # [nW, 1, N, N] -> [1, nW, 1, N, N]
            attn = attn + ops.cast(mask, attn.dtype) * -100.0
            attn = ops.reshape(attn, (-1, self.num_heads, N, N))
        
        # Softmax normalization and dropout
        attn = ops.softmax(attn, axis=-1)
        if self.attn_drop is not None:
            attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = ops.matmul(attn, v)  # [B_, num_heads, N, head_dim]
        
        # Transpose back to [B_, N, C]
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B_, N, C))
        
        # Output projection and dropout
        x = self.proj(x)
        if self.proj_drop is not None:
            x = self.proj_drop(x)
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "qkv_bias": self.qkv_bias,
            "attention_dropout": self.attention_dropout,
            "dropout": self.dropout,
        })
        return config


class SwinTransformerBlock(layers.Layer):
    """Swin Transformer Block.
    
    Args:
        dim: Number of input channels.
        input_resolution: Input resolution (height, width).
        num_heads: Number of attention heads.
        window_size: Window size for attention.
        shift_size: Shift size for shifted window attention (0 or window_size//2).
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value.
        dropout_rate: Dropout rate.
        attention_dropout: Dropout rate for attention.
        path_dropout: Stochastic depth rate.
        norm_layer: Normalization layer class.
    """

    def __init__(
        self,
        dim,
        input_resolution=None,
        num_heads=1,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout_rate=0.0,
        attention_dropout=0.0,
        path_dropout=0.0,
        norm_layer=layers.LayerNormalization,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(epsilon=1e-5, name="norm1")
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attention_dropout=attention_dropout,
            dropout=dropout_rate,
            name="attn"
        )
        self.drop_path = DropPath(path_dropout) if path_dropout > 0. else None
        self.norm2 = norm_layer(epsilon=1e-5, name="norm2")
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            dropout_rate=dropout_rate,
            name="mlp"
        )
        
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = ops.zeros((1, H, W, 1))
            
            h_slices = [
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            ]
            w_slices = [
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            ]
            
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask_segment = ops.ones((1, H, W, 1))
                    img_mask_segment = ops.index_update(
                        img_mask_segment, (..., h, w, ...), ops.ones((1, h.stop - h.start if h.stop else H - h.start, 
                                                                   w.stop - w.start if w.stop else W - w.start, 1)) * cnt
                    )
                    img_mask = img_mask + img_mask_segment
                    cnt += 1
                    
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = ops.reshape(mask_windows, (-1, self.window_size * self.window_size))
            attn_mask = ops.expand_dims(mask_windows, axis=1) - ops.expand_dims(mask_windows, axis=2)
            attn_mask = ops.where(attn_mask != 0, -100.0, 0.0)
            self.attn_mask = attn_mask
        else:
            self.attn_mask = None

    def call(self, x):
        B, L, C = ops.shape(x)
        H, W = self.input_resolution
        
        window_size = self.window_size
        shift_size = self.shift_size
        
        if min(H, W) <= window_size:
            window_size = min(H, W)
            shift_size = 0
        
        x = ops.reshape(x, (B, H, W, C))
        
        if self.shift_size > 0:
            shifted_x = ops.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x
        
        x_windows = window_partition(shifted_x, self.window_size)  # [B*num_windows, window_size, window_size, C]
        x_windows = ops.reshape(x_windows, (-1, self.window_size * self.window_size, C))  # [B*num_windows, window_size*window_size, C]
        
        identity = x_windows
        
        x_windows = self.norm1(x_windows)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # [B*num_windows, window_size*window_size, C]
        
        if self.drop_path is not None:
            attn_windows = self.drop_path(attn_windows)
        
        x_windows = identity + attn_windows
        
        identity = x_windows
        x_windows = self.norm2(x_windows)
        x_windows = self.mlp(x_windows)
        
        if self.drop_path is not None:
            x_windows = self.drop_path(x_windows)
        
        x_windows = identity + x_windows
        
        x_windows = ops.reshape(x_windows, (-1, self.window_size, self.window_size, C))
        
        if self.shift_size > 0:
            x = window_reverse(x_windows, self.window_size, H, W, C)
            x = ops.roll(x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = window_reverse(x_windows, self.window_size, H, W, C)
        
        x = ops.reshape(x, (B, H * W, C))
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "input_resolution": self.input_resolution,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "mlp_ratio": self.mlp_ratio,
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class PatchMerging(layers.Layer):
    """Patch Merging Layer.
    
    This layer performs downsampling by concatenating patches and using a linear layer.
    
    Args:
        dim: Number of input channels.
    """

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False, name="reduction")
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")

    def call(self, x, H, W):
        """Forward pass.
        
        Args:
            x: Input tensor with shape [B, H*W, C].
            H: Height of feature map.
            W: Width of feature map.
            
        Returns:
            Downsampled feature map with shape [B, H/2*W/2, 2*C].
        """
        B, L, C = ops.shape(x)
        
        x = ops.reshape(x, (B, H, W, C))
        pad_values = ((0, 0), (0, H % 2), (0, W % 2), (0, 0))
        x = ops.pad(x, pad_values)
        
        # Reshape to group patches
        x0 = x[:, 0::2, 0::2, :]  
        x1 = x[:, 1::2, 0::2, :]  
        x2 = x[:, 0::2, 1::2, :]  
        x3 = x[:, 1::2, 1::2, :]  
        
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)
        x = self.norm(x)
        x = self.reduction(x)
        x = ops.reshape(x, (B, -1, 2 * C))
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

    def compute_output_shape(self, input_shape):
        batch_size, seq_len, channels = input_shape
        return (batch_size, seq_len // 4, channels * 2)


class PatchEmbedding(layers.Layer):
    """Image to Patch Embedding.
    
    Args:
        patch_size: Size of each patch.
        embed_dim: Embedding dimension.
        norm_layer: Normalization layer.
        data_format: Format of the input data, either "channels_last" or "channels_first".
        patch_norm: If True, add normalization after patch embedding.
    """

    def __init__(
        self,
        patch_size=4,
        embed_dim=96,
        norm_layer=None,
        data_format="channels_last",
        patch_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.data_format = data_format
        
        self.proj = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format=data_format,
            name="proj",
        )
        
        self.norm = norm_layer(epsilon=1e-5, name="norm") if patch_norm and norm_layer else None

    def call(self, x):
        """Forward pass.
        
        Args:
            x: Input images with shape [B, H, W, C] in channels_last format
               or [B, C, H, W] in channels_first format.
            
        Returns:
            Patch embeddings with shape [B, H//patch_size * W//patch_size, embed_dim].
        """
        B = ops.shape(x)[0]
        
        x = self.proj(x) 
        
        if self.data_format == "channels_last":
            _, H, W, C = ops.shape(x)
            x = ops.reshape(x, (B, H * W, C))
        else:
            _, C, H, W = ops.shape(x)
            x = ops.transpose(x, (0, 2, 3, 1))  # [B, H, W, C]
            x = ops.reshape(x, (B, H * W, C))
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "data_format": self.data_format,
        })
        return config


class SwinTransformerStage(layers.Layer):
    """Swin Transformer Stage.
    
    A stage consists of multiple Swin Transformer blocks with the same resolution,
    and an optional patch merging layer at the beginning.
    
    Args:
        dim: Number of input channels.
        depth: Number of blocks in this stage.
        num_heads: Number of attention heads.
        window_size: Local window size.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value.
        dropout_rate: Dropout rate.
        attention_dropout: Dropout rate for attention.
        path_dropout: Stochastic depth rate.
        downsample: Downsample layer at the end of the layer.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout_rate=0.0,
        attention_dropout=0.0,
        path_dropout=0.0,
        downsample=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.blocks = []
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=None,  
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate,
                    attention_dropout=attention_dropout,
                    path_dropout=path_dropout[i] if isinstance(path_dropout, list) else path_dropout,
                    name=f"blocks_{i}",
                )
            )
        
        self.downsample = downsample

    def call(self, x):
        """Forward pass.
        
        Args:
            x: Input feature with shape [B, H*W, C].
            
        Returns:
            Output feature with shape [B, H/2*W/2, 2*C] if downsample is applied,
            otherwise [B, H*W, C].
        """
        B, L, C = ops.shape(x)
        
        H_W = ops.cast(ops.sqrt(ops.cast(L, "float32")), "int32")
        
        for block in self.blocks:
            block.input_resolution = (H_W, H_W)
        
        for block in self.blocks:
            x = block(x)
        
        if self.downsample is not None:
            x = self.downsample(x, H_W, H_W)
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "depth": self.depth,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
        })
        return config

    def compute_output_shape(self, input_shape):
        batch_size, seq_len, channels = input_shape
        if self.downsample is not None:
            return (batch_size, seq_len // 4, channels * 2)
        return input_shape
