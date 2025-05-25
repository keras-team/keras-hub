"""
Copyright 2022 The KerasCV Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

ControlNet Version

by AJ Young

"""

import keras

"""
Models
"""


class DiffusionModel(keras.Model):
    def __init__(
        self,
        img_height,
        img_width,
        max_text_length,
        name="LockedDiffusionModel",
    ):
        context = keras.layers.Input(
            (max_text_length, 768), name="Context_Input"
        )
        t_embed_input = keras.layers.Input((320,), name="TimeStepEmbed_Input")
        latent = keras.layers.Input(
            (img_height // 8, img_width // 8, 4), name="LatentImage_Input"
        )

        ### ControlNet Input ###

        controlNet1 = keras.layers.Input(shape=(1), name="ControlNet_Input001")
        controlNet2 = keras.layers.Input(shape=(1), name="ControlNet_Input002")
        controlNet3 = keras.layers.Input(shape=(1), name="ControlNet_Input003")
        controlNet4 = keras.layers.Input(shape=(1), name="ControlNet_Input004")
        controlNet5 = keras.layers.Input(shape=(1), name="ControlNet_Input005")
        controlNet6 = keras.layers.Input(shape=(1), name="ControlNet_Input006")
        controlNet7 = keras.layers.Input(shape=(1), name="ControlNet_Input007")
        controlNet8 = keras.layers.Input(shape=(1), name="ControlNet_Input008")
        controlNet9 = keras.layers.Input(shape=(1), name="ControlNet_Input009")
        controlNet10 = keras.layers.Input(shape=(1), name="ControlNet_Input010")
        controlNet11 = keras.layers.Input(shape=(1), name="ControlNet_Input011")
        controlNet12 = keras.layers.Input(shape=(1), name="ControlNet_Input012")
        controlNet13 = keras.layers.Input(shape=(1), name="ControlNet_Input013")

        controlNetResults = [
            controlNet1,
            controlNet2,
            controlNet3,
            controlNet4,
            controlNet5,
            controlNet6,
            controlNet7,
            controlNet8,
            controlNet9,
            controlNet10,
            controlNet11,
            controlNet12,
            controlNet13,
        ]

        t_emb = keras.layers.Dense(1280, name="TimeEmbed1")(t_embed_input)
        t_emb = keras.layers.Activation("swish", name="swishActivation")(t_emb)
        t_emb = keras.layers.Dense(1280, name="TimeEmbed2")(t_emb)

        # Downsampling flow, aka input_blocks

        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1, name="inputBlocks")(
            latent
        )
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(x)

        # Middle flow

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = ResBlock(1280)([x, t_emb])
        # controlNetResults, controlNetResult = tfPOP(controlNetResults)
        x = x + controlNetResults.pop()

        # Upsampling flow

        for _ in range(3):
            # controlNetResults, controlNetResult = tfPOP(controlNetResults)
            x = keras.layers.Concatenate()(
                [x, outputs.pop() + controlNetResults.pop()]
            )
            x = ResBlock(1280)([x, t_emb])
        x = Upsample(1280)(x)

        for _ in range(3):
            # controlNetResults, controlNetResult = tfPOP(controlNetResults)
            x = keras.layers.Concatenate()(
                [x, outputs.pop() + controlNetResults.pop()]
            )
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = Upsample(1280)(x)

        for _ in range(3):
            # controlNetResults, controlNetResult = tfPOP(controlNetResults)
            x = keras.layers.Concatenate()(
                [x, outputs.pop() + controlNetResults.pop()]
            )
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
        x = Upsample(640)(x)

        for _ in range(3):
            # controlNetResults, controlNetResult = tfPOP(controlNetResults)
            x = keras.layers.Concatenate()(
                [x, outputs.pop() + controlNetResults.pop()]
            )
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])

        # Exit flow

        x = GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        output = PaddedConv2D(4, kernel_size=3, padding=1)(x)

        super().__init__(
            inputs=[
                latent,
                t_embed_input,
                context,
                controlNet1,
                controlNet2,
                controlNet3,
                controlNet4,
                controlNet5,
                controlNet6,
                controlNet7,
                controlNet8,
                controlNet9,
                controlNet10,
                controlNet11,
                controlNet12,
                controlNet13,
            ],
            outputs=output,
            name=name,
        )


class ControlNetDiffusionModel(keras.Model):
    def __init__(
        self,
        img_height,
        img_width,
        max_text_length,
        name="ControlNetModel",
    ):
        context = keras.layers.Input(
            (max_text_length, 768), name="Context_Input"
        )
        inputHint = keras.layers.Input(
            (img_height, img_width, 3), name="Hint_Input"
        )
        t_embed_input = keras.layers.Input((320,), name="TimeStepEmbed_Input")
        latent = keras.layers.Input(
            (img_height // 8, img_width // 8, 4), name="LatentImage_Input"
        )

        t_emb = keras.layers.Dense(1280, name="ControlTimeEmbed1")(
            t_embed_input
        )
        t_emb = keras.layers.Activation("swish", name="swishActivation")(t_emb)
        t_emb = keras.layers.Dense(1280, name="ControlTimeEmbed2")(t_emb)

        # Input Hint Blocks

        guidedHint = HintBlocks()(inputHint)

        # Downsampling flow, aka input_blocks

        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1, name="inputBlocks")(
            latent
        )
        x = x + guidedHint
        outputs.append(zeroConv(x, 320, "zeroConv1"))

        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])
            outputs.append(zeroConv(x, 320))
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(zeroConv(x, 320, "zeroConv4"))

        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
            outputs.append(zeroConv(x, 640))
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(zeroConv(x, 640, "zeroConv7"))

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
            outputs.append(zeroConv(x, 1280))
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(zeroConv(x, 1280, "zeroConv10"))

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(zeroConv(x, 1280))

        # Middle flow

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = ResBlock(1280)([x, t_emb])
        outputs.append(zeroConv(x, 1280, "zeroConv13"))

        super().__init__(
            [latent, t_embed_input, context, inputHint], outputs, name=name
        )
        # Input: Latent, TimestepEmbed, Context, Input Hint
        # Output: Python List of each zeroConv (Zero Convolution Layer)


class DiffusionModelV2(keras.Model):
    def __init__(self, img_height, img_width, max_text_length, name=None):
        context = keras.layers.Input((max_text_length, 1024))
        t_embed_input = keras.layers.Input((320,))
        latent = keras.layers.Input((img_height // 8, img_width // 8, 4))

        t_emb = keras.layers.Dense(1280)(t_embed_input)
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = keras.layers.Dense(1280)(t_emb)

        # Downsampling flow

        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1)(latent)
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(5, 64, fully_connected=True)([x, context])
            outputs.append(x)
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(10, 64, fully_connected=True)([x, context])
            outputs.append(x)
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(20, 64, fully_connected=True)([x, context])
            outputs.append(x)
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(x)

        # Middle flow

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(20, 64, fully_connected=True)([x, context])
        x = ResBlock(1280)([x, t_emb])

        # Upsampling flow

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(20, 64, fully_connected=True)([x, context])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(10, 64, fully_connected=True)([x, context])
        x = Upsample(640)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(5, 64, fully_connected=True)([x, context])

        # Exit flow

        x = GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        output = PaddedConv2D(4, kernel_size=3, padding=1)(x)

        super().__init__([latent, t_embed_input, context], output, name=name)


"""
Blocks
"""


class GroupNormalization(keras.layers.Layer):
    """
    GroupNormalization layer.

    This layer is only here temporarily and will be removed
    as we introduce GroupNormalization in core Keras.
    """

    def __init__(
        self,
        groups=32,
        axis=-1,
        epsilon=1e-5,
        name="GroupNormalization",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon

    def get_config(self):
        config = super().get_config()
        config.update(
            {"groups": self.groups, "axis": self.axis, "epsilon": self.epsilon}
        )
        return config

    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.gamma = self.add_weight(
            shape=(dim,),
            name="gamma",
            initializer="ones",
        )
        self.beta = self.add_weight(
            shape=(dim,),
            name="beta",
            initializer="zeros",
        )

    ## @tf.function
    def call(self, inputs):
        input_shape = keras.ops.shape(inputs)
        reshaped_inputs = self._reshape_into_groups(inputs, input_shape)
        normalized_inputs = self._apply_normalization(
            reshaped_inputs, input_shape
        )
        return keras.ops.reshape(normalized_inputs, input_shape)

    def _reshape_into_groups(self, inputs, input_shape):
        group_shape = [input_shape[i] for i in range(inputs.shape.rank)]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = keras.ops.stack(group_shape)
        return keras.ops.reshape(inputs, group_shape)

    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_reduction_axes = list(range(1, reshaped_inputs.shape.rank))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)
        mean, variance = keras.ops.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )
        gamma, beta = self._get_reshaped_weights(input_shape)
        return keras.ops.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = keras.ops.reshape(self.gamma, broadcast_shape)
        beta = keras.ops.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * input_shape.shape.rank
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape


class HintBlocks(keras.layers.Layer):
    def __init__(self, hint_channels=16, model_channels=320, **kwargs):
        super().__init__(**kwargs)
        self.layers = [
            PaddedConv2D(filters=16, kernel_size=3, padding=1),
            keras.layers.Activation("swish"),
            PaddedConv2D(filters=16, kernel_size=3, padding=1),
            keras.layers.Activation("swish"),
            PaddedConv2D(filters=32, kernel_size=3, padding=1, strides=2),
            keras.layers.Activation("swish"),
            PaddedConv2D(filters=32, kernel_size=3, padding=1),
            keras.layers.Activation("swish"),
            PaddedConv2D(filters=96, kernel_size=3, padding=1, strides=2),
            keras.layers.Activation("swish"),
            PaddedConv2D(filters=96, kernel_size=3, padding=1),
            keras.layers.Activation("swish"),
            PaddedConv2D(filters=256, kernel_size=3, padding=1, strides=2),
            keras.layers.Activation("swish"),
            PaddedConv2D(filters=model_channels, kernel_size=3, padding=1),
        ]

    ## @tf.function
    def call(self, inputs):
        x = inputs
        layerNumber = 0
        layerLength = len(self.layers)
        for layer in self.layers:
            if layerNumber == layerLength:
                for weight in layer.weights:
                    weight.assign(keras.ops.zeros_like(weight))
            x = layer(x)
            layerNumber += 1
        return x


class PaddedConv2D(keras.layers.Layer):
    def __init__(
        self, filters, kernel_size, padding=0, strides=1, name=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.padding2d = keras.layers.ZeroPadding2D(padding, name=name)
        self.conv2d = keras.layers.Conv2D(
            filters, kernel_size, strides=strides, name=name
        )
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

    ## @tf.function
    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "strides": self.strides,
            }
        )
        return config


class ResBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.entry_flow = [
            GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish", name="ResBlock_swish1"),
            PaddedConv2D(output_dim, 3, padding=1, name="inLayers2"),
        ]
        self.embedding_flow = [
            keras.layers.Activation("swish"),
            keras.layers.Dense(output_dim, name="embeddingLayer"),
        ]
        self.exit_flow = [
            GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish", name="ResBlock_swish2"),
            PaddedConv2D(output_dim, 3, padding=1, name="outLayers3"),
        ]

    def build(self, input_shape):
        if input_shape[0][-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    ## @tf.function
    def call(self, inputs):
        inputs, embeddings = inputs
        x = inputs
        for layer in self.entry_flow:
            x = layer(x)
        for layer in self.embedding_flow:
            embeddings = layer(embeddings)
        x = x + embeddings[:, None, None]
        for layer in self.exit_flow:
            x = layer(x)
        return x + self.residual_projection(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
            }
        )
        return config


class SpatialTransformer(keras.layers.Layer):
    def __init__(self, num_heads, head_size, fully_connected=False, **kwargs):
        super().__init__(**kwargs)
        self.norm = GroupNormalization(epsilon=1e-5)
        self.num_heads = num_heads
        self.head_size = head_size
        self.fully_connected = fully_connected
        channels = num_heads * head_size
        if fully_connected:
            self.proj1 = keras.layers.Dense(
                num_heads * head_size, name="proj_in1_fullyConnected"
            )
        else:
            self.proj1 = PaddedConv2D(num_heads * head_size, 1, name="proj_in")
        self.transformer_block = BasicTransformerBlock(
            channels, num_heads, head_size
        )
        if fully_connected:
            self.proj2 = keras.layers.Dense(channels)
        else:
            self.proj2 = PaddedConv2D(
                channels, 1, name="proj_in2_fullyConnected"
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "head_size": self.head_size,
                "fully_connected": self.fully_connected,
            }
        )
        return config

    ## @tf.function
    def call(self, inputs):
        inputs, context = inputs
        _, h, w, c = inputs.shape
        x = self.norm(inputs)
        x = self.proj1(x)
        x = keras.ops.reshape(x, (-1, h * w, c))
        x = self.transformer_block([x, context])
        x = keras.ops.reshape(x, (-1, h, w, c))
        return self.proj2(x) + inputs


class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.attn1 = CrossAttention(num_heads, head_size)

        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.attn2 = CrossAttention(num_heads, head_size)

        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5, name="norm3")
        self.geglu = GEGLU(dim * 4)
        self.dense = keras.layers.Dense(dim)

    ## @tf.function
    def call(self, inputs):
        inputs, context = inputs
        x = self.attn1([self.norm1(inputs), None]) + inputs
        x = self.attn2([self.norm2(x), context]) + x
        return self.dense(self.geglu(self.norm3(x))) + x


class CrossAttention(keras.layers.Layer):
    def __init__(self, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.to_q = keras.layers.Dense(
            num_heads * head_size, use_bias=False, name="to_q"
        )
        self.to_k = keras.layers.Dense(
            num_heads * head_size, use_bias=False, name="to_k"
        )
        self.to_v = keras.layers.Dense(
            num_heads * head_size, use_bias=False, name="to_v"
        )
        self.scale = head_size**-0.5
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_proj = keras.layers.Dense(
            num_heads * head_size, name="out_projection"
        )

    ## @tf.function
    def call(self, inputs):
        inputs, context = inputs
        context = inputs if context is None else context
        q, k, v = self.to_q(inputs), self.to_k(context), self.to_v(context)
        q = keras.ops.reshape(
            q, (-1, inputs.shape[1], self.num_heads, self.head_size)
        )
        k = keras.ops.reshape(
            k, (-1, context.shape[1], self.num_heads, self.head_size)
        )
        v = keras.ops.reshape(
            v, (-1, context.shape[1], self.num_heads, self.head_size)
        )

        q = keras.ops.transpose(
            q, (0, 2, 1, 3)
        )  # (bs, num_heads, time, head_size)
        k = keras.ops.transpose(
            k, (0, 2, 3, 1)
        )  # (bs, num_heads, head_size, time)
        v = keras.ops.transpose(
            v, (0, 2, 1, 3)
        )  # (bs, num_heads, time, head_size)

        score = td_dot(q, k) * self.scale
        weights = keras.activations.softmax(
            score
        )  # (bs, num_heads, time, time)
        attn = td_dot(weights, v)
        attn = keras.ops.transpose(
            attn, (0, 2, 1, 3)
        )  # (bs, time, num_heads, head_size)
        out = keras.ops.reshape(
            attn, (-1, inputs.shape[1], self.num_heads * self.head_size)
        )
        return self.out_proj(out)


class Upsample(keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.ups = keras.layers.UpSampling2D(2)
        self.conv = PaddedConv2D(channels, 3, padding=1, name="Upsample")

    ## @tf.function
    def call(self, inputs):
        return self.conv(self.ups(inputs))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
            }
        )
        return config


class GEGLU(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense = keras.layers.Dense(output_dim * 2)

    ## @tf.function
    def call(self, inputs):
        x = self.dense(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        tanh_res = keras.activations.tanh(
            gate * 0.7978845608 * (1 + 0.044715 * (gate**2))
        )
        return x * 0.5 * gate * (1 + tanh_res)


def td_dot(a, b):
    aa = keras.ops.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = keras.ops.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = keras.backend.batch_dot(aa, bb)
    return keras.ops.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))


def zeroConv(tensor, channels, name=None):
    layer = keras.layers.Conv2D(
        filters=channels, kernel_size=1, padding="same", name=name
    )
    for weight in layer.weights:
        weight.assign(keras.ops.zeros_like(weight))

    tensor = layer(tensor)

    return tensor
