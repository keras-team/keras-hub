from keras import layers


class ConvBNAct(layers.Layer):
    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        activation="swish",
        **kwargs
    ):
        super(ConvBNAct, self).__init__(**kwargs)
        self.conv = layers.Conv2D(
            out_channels,
            kernel_size,
            strides=stride,
            padding=padding,
            use_bias=False,
        )
        self.bn = layers.BatchNormalization(epsilon=0.001, momentum=0.03)
        self.act = layers.Activation(activation)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return self.act(x)


class Bottleneck(layers.Layer):
    """Standard bottleneck layer with optional shortcut connection."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        c_ = int(c2 * e)  # hidden channels
        # First convolution with customizable kernel size k[0]
        self.cv1 = ConvBNAct(c_, kernel_size=k[0], stride=1, activation="swish")
        # Second convolution with customizable kernel size k[1] and optional groups for grouped convolution
        self.cv2 = ConvBNAct(c2, kernel_size=k[1], stride=1, activation="swish")

        # Determine if a residual connection (shortcut) should be added
        self.add = shortcut and (c1 == c2)

    def call(self, x, training=False):
        out = self.cv1(x, training=training)
        out = self.cv2(out, training=training)

        # If shortcut is enabled, add input x to the output
        if self.add:
            return layers.Add()([x, out])
        return out


class C3f(layers.Layer):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, **kwargs):
        super(C3f, self).__init__(**kwargs)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBNAct(c_, kernel_size=1)
        self.cv2 = ConvBNAct(c_, kernel_size=1)
        self.cv3 = ConvBNAct(c2, kernel_size=1)
        self.m = [Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]

    def call(self, x, training=False):
        y = [self.cv2(x, training=training), self.cv1(x, training=training)]
        for m in self.m:
            y.append(m(y[-1], training=training))
        return self.cv3(layers.Concatenate(axis=-1)(y), training=training)


class C3k2(C3f):
    def __init__(
        self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, **kwargs
    ):
        super(C3k2, self).__init__(c1, c2, n, shortcut, g, e, **kwargs)
        c_ = int(c2 * e)  # hidden channels
        self.m = [
            (
                C3k(c_, c_, 2, shortcut, g)
                if c3k
                else Bottleneck(c_, c_, shortcut, g)
            )
            for _ in range(n)
        ]


class C3k(layers.Layer):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, **kwargs):
        super(C3k, self).__init__(**kwargs)
        c_ = int(c2 * e)  # hidden channels
        self.m = [Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]

    def call(self, x, training=False):
        for m in self.m:
            x = m(x, training=training)
        return x
