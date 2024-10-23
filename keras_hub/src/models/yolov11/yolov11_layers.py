from keras import Sequential
from keras import layers
from keras import ops


class ConvBNAct(layers.Layer):
    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        activation="swish",
        apply_act=True,
        groups=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(
            out_channels,
            kernel_size,
            strides=stride,
            padding=padding,
            use_bias=False,
            groups=groups,
        )
        self.bn = layers.BatchNormalization(epsilon=0.001, momentum=0.03)
        self.act = layers.Activation(activation)
        self.apply_act = apply_act

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.apply_act:
            return self.act(x)
        else:
            return x


class Bottleneck(layers.Layer):
    """Standard bottleneck layer with optional shortcut connection."""

    def __init__(self, c2, shortcut=True, groups=1, k=(3, 3), e=0.5, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        c_ = int(c2 * e)  # hidden channels
        # First convolution with customizable kernel size k[0]
        self.cv1 = ConvBNAct(c_, kernel_size=k[0], stride=1, activation="swish")
        # Second convolution with customizable kernel size k[1] and optional groups for grouped convolution
        self.cv2 = ConvBNAct(c2, kernel_size=k[1], stride=1, activation="swish")

        # Determine if a residual connection (shortcut) should be added
        self.shortcut = shortcut

    def call(self, x, training=False):
        out = self.cv1(x, training=training)
        out = self.cv2(out, training=training)

        # If shortcut is enabled, add input x to the output
        if self.shortcut and (out.shape[-1] == x.shape[-1]):
            return layers.Add()([x, out])
        return out


class CSPBottleneck3C(layers.Layer):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c2, n=1, shortcut=True, groups=1, e=0.5, **kwargs):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__(**kwargs)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBNAct(c_, 1, 1)
        self.cv2 = ConvBNAct(c_, 1, 1)
        self.cv3 = ConvBNAct(c2, 1)  # optional act=FReLU(c2)
        self.m = [
            Bottleneck(c_, shortcut, groups, k=((1, 1), (3, 3)), e=1.0)
            for _ in range(n)
        ]

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(
            ops.concatenate((self.m(self.cv1(x)), self.cv2(x)), axis=1)
        )


class CSPBottleneck3CFast(layers.Layer):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c2, n=1, shortcut=False, groups=1, e=0.5, **kwargs):
        super().__init__(**kwargs)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBNAct(c_, kernel_size=1)
        self.cv2 = ConvBNAct(c_, kernel_size=1)
        self.cv3 = ConvBNAct(c2, kernel_size=1)
        self.m = [Bottleneck(c_, shortcut, groups, e=1.0) for _ in range(n)]

    def call(self, x, training=False):
        y = [self.cv2(x, training=training), self.cv1(x, training=training)]
        for m in self.m:
            y.append(m(y[-1], training=training))
        return self.cv3(layers.Concatenate(axis=-1)(y), training=training)


class C3k(CSPBottleneck3C):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c2, n=1, shortcut=True, groups=1, e=0.5, k=3, **kwargs):
        super().__init__(
            c2=c2, n=1, shortcut=shortcut, groups=groups, e=e, **kwargs
        )
        c_ = int(c2 * e)  # hidden channels
        self.m = [Bottleneck(c_, shortcut, groups, e=1.0) for _ in range(n)]

    def call(self, x, training=False):
        for m in self.m:
            x = m(x, training=training)
        return x


class C3k2(CSPBottleneck3CFast):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c2, n=1, c3k=False, e=0.5, groups=1, shortcut=True, **kwargs
    ):
        super().__init__(c2, n, shortcut, groups, e, **kwargs)
        c_ = int(c2 * e)  # hidden channels
        self.m = [
            (
                C3k(c_, 2, shortcut, groups)
                if c3k
                else Bottleneck(c_, shortcut, groups)
            )
            for _ in range(n)
        ]


class SPPF(layers.Layer):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvBNAct(c_, 1, 1)
        self.cv2 = ConvBNAct(c2, 1, 1)
        self.m = layers.MaxPooling2D(
            pool_size=k, strides=1, padding="same"
        )  # padding should be k//2

    def call(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(ops.concatenate(y, axis=1))


class C2PSA(layers.Layer):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention
    mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules
    for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (ConvBNAct): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (ConvBNAct): 1x1 convolution layer to reduce the number of output channels to c.
        m (keras.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    """

    def __init__(self, c, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        self.c = int(c * e)
        self.cv1 = ConvBNAct(2 * self.c, 1, 1)
        self.cv2 = ConvBNAct(c, 1)

        self.m = Sequential(
            [
                PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64)
                for _ in range(n)
            ]
        )

    def call(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = ops.split(self.cv1(x), indices_or_sections=2, axis=-1)
        b = self.m(b)
        return self.cv2(ops.concatenate((a, b), axis=1))


class PSABlock(layers.Layer):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head
    attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = Sequential(
            [ConvBNAct(c * 2, 1), ConvBNAct(c, 1, apply_act=False)]
        )
        self.add = shortcut

    def call(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class Attention(layers.Layer):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (ConvBNAct): Convolutional layer for computing the query, key, and value.
        proj (ConvBNAct): Convolutional layer for projecting the attended values.
        pe (ConvBNAct): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = ConvBNAct(h, 1, apply_act=False)
        self.proj = ConvBNAct(dim, 1, apply_act=False)
        self.pe = ConvBNAct(dim, 3, 1, groups=dim, apply_act=False)

    def call(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x: KerasTensor. The input tensor.

        Returns:
            KerasTensor. The output tensor after self-attention.
        """
        B, C, H, W = ops.shape(x)
        N = H * W
        qkv = self.qkv(x)

        qkv = ops.reshape(
            qkv, (B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
        )
        q, k, v = ops.split(qkv, (self.key_dim, self.head_dim), axis=2)

        attn = (ops.transpose(q, (0, 1, 3, 2)) @ k) * self.scale
        attn = ops.softmax(attn, axis=-1)
        x = v @ ops.transpose(attn, (0, 1, 3, 2))
        x = ops.reshape(x, (B, C, H, W))
        x = x + self.pe(ops.reshape(v, (B, C, H, W)))
        x = self.proj(x)
        return x
