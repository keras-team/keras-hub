import keras
from keras import layers
from keras import ops
import collections.abc
from typing import Union, Tuple, Any
import numpy as np

def get_relative_position_index(win_h, win_w):
    """Get pair-wise relative position index for each token inside the window.
    
    Args:
        win_h: Height of the window.
        win_w: Width of the window.
        
    Returns:
        A tensor of shape (win_h*win_w, win_h*win_w) containing the relative
        position indices for each pair of tokens in the window.
    """
    xx, yy = ops.meshgrid(ops.arange(win_h), ops.arange(win_w), indexing="ij")
    coords = ops.stack([yy, xx], axis=0)  
    coords_flatten = ops.reshape(coords, (2, -1))    
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
    relative_coords = ops.transpose(relative_coords, (1, 2, 0))  
    xx = (relative_coords[:, :, 0] + win_h - 1) * (2 * win_w - 1)
    yy = relative_coords[:, :, 1] + win_w - 1
    relative_coords = ops.stack([xx, yy], axis=-1)
    relative_position_index = ops.sum(relative_coords, axis=-1) 
    return relative_position_index

def window_partition(x, window_size):
    """Partition the input tensor into non-overlapping windows.
    
    Args:
        x: Input tensor with shape [B, H, W, C]
        window_size: Size of the window
        
    Returns:
        Windows with shape [B*num_windows, window_size, window_size, C]
    """
    shape = ops.shape(x)
    if len(shape) != 4:
        raise ValueError(f"Expected input tensor to have 4 dimensions, got {len(shape)}")
    
    B = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        H = H + pad_h
        W = W + pad_w
    
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    
    # Reshape to windows
    x = ops.reshape(
        x,
        (
            B,
            num_windows_h,
            window_size,
            num_windows_w,
            window_size,
            C
        )
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5)) 
    windows = ops.reshape(
        x,
        (-1, window_size, window_size, C)
    )
    
    return windows, (H, W)


def window_reverse(windows, window_size, height, width, channels):
    """Reverse window partitioning.
    
    Args:
        windows: Windows with shape [B*num_windows, window_size, window_size, C]
        window_size: Size of the window
        height: Height of the feature map
        width: Width of the feature map
        channels: Number of channels
        
    Returns:
        Feature map with shape [B, H, W, C]
    """
    # Calculate number of windows
    num_windows_h = height // window_size
    num_windows_w = width // window_size
    batch_size = ops.shape(windows)[0] // (num_windows_h * num_windows_w)
    
    # Reshape windows to [B, num_windows_h, num_windows_w, window_size, window_size, C]
    x = ops.reshape(
        windows,
        (
            batch_size,
            num_windows_h,
            num_windows_w,
            window_size,
            window_size,
            channels
        )
    )
    
    # Permute dimensions to get [B, H, W, C]
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
        keep_prob = 1.0 - self.drop_prob
        
        batch_size = ops.shape(x)[0]
        random_tensor = keep_prob + ops.random.uniform((batch_size, 1, 1, 1))
        binary_mask = ops.floor(random_tensor)
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
        act_layer: Activation function to use (e.g., keras.activations.gelu).
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=keras.activations.relu,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.act_layer = act_layer
        self.dropout_rate = dropout_rate

        self.fc1 = layers.Dense(hidden_features, name="fc1")
        self.fc2 = layers.Dense(out_features, name="fc2")
        self.drop = layers.Dropout(dropout_rate) if dropout_rate > 0.0 else None

    def call(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
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
            "act_layer": keras.activations.serialize(self.act_layer),
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["act_layer"] = keras.activations.deserialize(config["act_layer"])
        return cls(**config)


class WindowAttention(keras.layers.Layer):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float, optional): Override default scaling factor for queries and keys (default: head_dim ** -0.5)
        attn_drop (float, optional): Dropout ratio of attention weights. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        num_heads,
        head_dim=None,
        window_size=7,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = (
            window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )
        self.win_h, self.win_w = self.window_size
        self.window_area = self.win_h * self.win_w
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.scale = qk_scale if qk_scale is not None else self.head_dim ** -0.5
        self.attn_dim = self.head_dim * self.num_heads
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop

        self.relative_position_index = get_relative_position_index(
            win_h=self.win_h,
            win_w=self.win_w
        )

    def build(self, input_shape):
        self.qkv = keras.layers.Dense(
            self.head_dim * self.num_heads * 3, use_bias=self.qkv_bias, name="attention_qkv"
        )
        self.attn_drop = keras.layers.Dropout(self.attn_drop_rate)
        self.proj = keras.layers.Dense(self.dim, name="attention_projection")
        self.proj_drop = keras.layers.Dropout(self.proj_drop_rate)

        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.win_h - 1) * (2 * self.win_w - 1), self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="relative_position_bias_table",
        )
        super().build(input_shape)

    def _get_rel_pos_bias(self) -> Any:
        relative_position_bias = ops.take(
            self.relative_position_bias_table,
            self.relative_position_index,
            axis=0,
        )
        return ops.transpose(relative_position_bias, (2, 0, 1))

    def call(
        self, x, mask=None, return_attns=False
    ) -> Union[Any, Tuple[Any, Any]]:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (B_, N, 3, self.num_heads, -1))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = ops.unstack(qkv, 3)

        scale = ops.cast(self.scale, dtype=qkv.dtype)
        q = q * scale
        attn = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2]))
        attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            num_win = ops.shape(mask)[0]
            attn = ops.reshape(
                attn, (B_ // num_win, num_win, self.num_heads, N, N)
            )
            attn = attn + ops.expand_dims(mask, 1)[None, ...]

            attn = ops.reshape(attn, (-1, self.num_heads, N, N))
            attn = ops.nn.softmax(attn, -1)
        else:
            attn = ops.nn.softmax(attn, -1)

        attn = self.attn_drop(attn)

        x = ops.matmul(attn, v)
        x = ops.transpose(x, axes=[0, 2, 1, 3])
        x = ops.reshape(x, (B_, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attns:
            return x, attn
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "win_h": self.win_h,
                "win_w": self.win_w,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "attn_dim": self.attn_dim,
                "scale": self.scale,
                "qkv_bias": self.qkv_bias,
                "attn_drop": self.attn_drop,
                "proj_drop": self.proj_drop,
            }
        )
        return config


class SwinTransformerBlock(keras.layers.Layer):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (keras.layers.Layer, optional): Activation layer. Default: keras.layers.Activation("gelu")
        norm_layer (keras.layers.Layer, optional): Normalization layer. Default: keras.layers.LayerNormalization(epsilon=1e-5)
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=keras.activations.gelu,
        norm_layer=keras.layers.LayerNormalization,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.act_layer = act_layer

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(epsilon=1e-5, name="norm1")
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            name="attn",
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else keras.layers.Identity()
        self.norm2 = norm_layer(epsilon=1e-5, name="norm2")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=self.act_layer,
            dropout_rate=drop,
            name="mlp",
        )

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = ops.shape(x)

        shortcut = x
        x = self.norm1(x)
        x = ops.reshape(x, (B, H, W, C))

        attn_mask = None
        if self.shift_size > 0:
            shifted_x = ops.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
            img_mask = np.zeros((1, H, W, 1), dtype=np.int32)
            cnt = 0
            h_slices = [
                (0, H // 2),
                (H // 2, H - self.shift_size),
                (H - self.shift_size, H),
            ]
            w_slices = [
                (0, W // 2),
                (W // 2, W - self.shift_size),
                (W - self.shift_size, W),
            ]
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h[0]:h[1], w[0]:w[1], :] = cnt
                    cnt += 1
            img_mask = ops.convert_to_tensor(img_mask)

            mask_windows = window_partition(img_mask, self.window_size)[0]
            mask_windows = ops.reshape(mask_windows, (-1, self.window_size * self.window_size))
            attn_mask = ops.expand_dims(mask_windows, 1) - ops.expand_dims(mask_windows, 2)
            attn_mask = ops.where(attn_mask != 0, -100.0, 0.0)
        else:
            shifted_x = x

        x_windows, (H_pad, W_pad) = window_partition(x=shifted_x, window_size=self.window_size)
        x_windows = ops.reshape(x_windows, (-1, self.window_size * self.window_size, C))
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = ops.reshape(attn_windows, (-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad, C)

        if self.shift_size > 0:
            x = ops.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x

        if H_pad > H or W_pad > W:
            x = x[:, :H, :W, :]

        x = ops.reshape(x, (B, H * W, C))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "input_resolution": self.input_resolution,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "shift_size": self.shift_size,
                "mlp_ratio": self.mlp_ratio,
            }
        )
        return config


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

class PatchEmbedding(layers.Layer):
    """Image to Patch Embedding layer for Swin Transformer.

    Args:
        patch_size: int. Patch size (usually 4).
        embed_dim: int. Output embedding dimension.
        norm_layer: Callable layer class for normalization (e.g., LayerNormalization).
        data_format: str. Either "channels_last" or "channels_first".
        patch_norm: bool. Whether to apply normalization.
    """

    def __init__(
        self,
        patch_size=4,
        embed_dim=96,
        norm_layer=None,
        data_format="channels_last",
        patch_norm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.data_format = data_format
        self.patch_norm = patch_norm
        self.norm_layer = norm_layer

        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format=data_format,
            name="proj",
        )

        if self.patch_norm and self.norm_layer is not None:
            self.norm = norm_layer(name="norm")
        else:
            self.norm = None

    def call(self, x):
        x = self.proj(x)  # shape: (B, H//P, W//P, C)
        if self.data_format == "channels_first":
            x = ops.transpose(x, [0, 2, 3, 1])
        x = ops.reshape(x, [ops.shape(x)[0], -1, self.embed_dim])
        if self.norm:
            x = self.norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "data_format": self.data_format,
            "patch_norm": self.patch_norm,
            "norm_layer": keras.saving.serialize_keras_object(self.norm_layer)
            if self.norm_layer else None,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["norm_layer"] = keras.saving.deserialize_keras_object(config["norm_layer"])
        return cls(**config)


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
        drop: Dropout rate.
        attn_drop: Dropout rate for attention.
        drop_path: Stochastic depth rate.
        downsample: Downsample layer at the end of the layer.
        input_resolution: Input resolution.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        downsample=None,
        input_resolution=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.blocks = []
        self.downsample = downsample
        self._drop_path = drop_path
        self._qkv_bias = qkv_bias
        self._drop = drop
        self._attn_drop = attn_drop
        self.input_resolution = input_resolution

    def build(self, input_shape):
        for i in range(self.depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=self.dim,
                    input_resolution=self.input_resolution,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self._qkv_bias,
                    drop=self._drop,
                    attn_drop=self._attn_drop,
                    drop_path=self._drop_path[i] if isinstance(self._drop_path, list) else self._drop_path,
                    name=f"blocks_{i}",
                )
            )
        
        if self.downsample is not None:
            self.downsample = self.downsample(
                dim=self.dim,
                name="downsample",
            )

        super().build(input_shape)

    def call(self, x):
        """Forward pass.
        
        Args:
            x: Input feature with shape [B, H*W, C].
            
        Returns:
            Output feature with shape [B, H/2*W/2, 2*C] if downsample is applied,
            otherwise [B, H*W, C].
        """
        for block in self.blocks:
            x = block(x)
        
        if self.downsample is not None:
            H, W = self.input_resolution
            x = self.downsample(x, H=H, W=W)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self._qkv_bias,
            "drop": self._drop,
            "attn_drop": self._attn_drop,
            "drop_path": self._drop_path,
            "downsample": keras.utils.serialize_keras_object(self.downsample) if self.downsample else None,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["downsample"] = keras.utils.deserialize_keras_object(config["downsample"]) if config["downsample"] else None
        return cls(**config)

