# Copyright 2024 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writingf, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from keras_nlp.src.backend import config
from keras_nlp.src.backend import keras
from keras_nlp.src.backend import ops
from keras_nlp.src.models.paligemma.vision_embeddings import VisionEmbeddings


class VitEncoderBlock(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        intermediate_size,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = None
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size

    def compute_attention(self, x, mask=None):
        mask = None
        if mask is not None:
            mask = ops.cast(mask, dtype=x.dtype) if mask is not None else None

        return self.attn(
            x,
            x,
            x,
            attention_mask=mask,
            return_attention_scores=True,
        )[0]

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]
        self.attn = keras.layers.MultiHeadAttention(
            self.num_heads,
            key_dim=self.hidden_dim // self.num_heads,
            name="multi_head_attention",
        )
        self.layer_norm_1 = keras.layers.LayerNormalization(
            epsilon=1e-5, name="layer_norm_1"
        )
        self.mlp_dense_1 = keras.layers.Dense(
            self.intermediate_size,
            name="mlp_dense_1",
            activation="gelu",
        )
        self.mlp_dense_2 = keras.layers.Dense(
            self.hidden_dim,
            name="mlp_dense_2",
        )
        self.layer_norm_2 = keras.layers.LayerNormalization(
            epsilon=1e-5, name="layer_norm_2"
        )
        self.attn.build(
            [None, None, self.hidden_dim],
            [None, None, self.hidden_dim],
        )
        self.layer_norm_1.build([None, None, self.hidden_dim])
        self.mlp_dense_1.build([None, None, self.hidden_dim])
        self.mlp_dense_2.build([None, None, self.intermediate_size])
        self.layer_norm_2.build([None, None, self.hidden_dim])
        self.built = True

    def call(self, x, mask=None):
        residual = x
        x = self.layer_norm_1(x)
        x = self.compute_attention(x, mask)
        x = x + residual
        residual = x
        x = self.mlp_dense_1(self.layer_norm_2(residual))
        x = self.mlp_dense_2(x)
        return residual + x

    def compute_output_shape(self, inputs_shape):
        return inputs_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


class VitEncoder(keras.layers.Layer):
    def __init__(
        self, hidden_dim, num_layers, num_heads, intermediate_size, **kwargs
    ):
        if not config.keras_3():
            raise ValueError(
                "`PaLIGemma` requires Keras 3. Run `pip install -U keras` "
                "upgrade your Keras version, or see "
                "https://keras.io/getting_started/ "
                "for more info on Keras versions and installation."
            )
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.encoder_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-5, name="encoder_layer_norm"
        )
        self.vision_embeddings = VisionEmbeddings(hidden_dim=hidden_dim)
        self.resblocks = [
            VitEncoderBlock(
                self.num_heads,
                self.intermediate_size,
                name=f"encoder_block_{i}",
            )
            for i in range(self.num_layers)
        ]

    def build(self, input_shape):
        self.vision_embeddings.build(input_shape)
        for block in self.resblocks:
            block.build([None, None, self.hidden_dim])
        self.built = True

    def call(
        self,
        x,
        mask=None,
    ):
        x = self.vision_embeddings(x)
        for block in self.resblocks:
            x = block(x, mask=mask)
        return self.encoder_layer_norm(x)

    def compute_output_shape(self, inputs_shape):
        return [inputs_shape[0], inputs_shape[1], self.hidden_dim]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
            }
        )
        return config


class MultiheadAttentionPooling(keras.layers.Layer):
    def __init__(self, hidden_dim=None, num_heads=12, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

    def build(self, input_shape):
        if self.hidden_dim is None:
            self.hidden_dim = input_shape[-1] * 4
        self.probe = self.add_weight(
            shape=(1, 1, input_shape[-1]),
            initializer="glorot_uniform",
        )
        self.mha = keras.layers.MultiHeadAttention(
            key_dim=input_shape[-1] // self.num_heads,
            num_heads=self.num_heads,
        )
        self.layer_norm = keras.layers.LayerNormalization()
        self.mlp_block = keras.Sequential(
            [
                keras.layers.Dense(self.hidden_dim, activation="gelu"),
                keras.layers.Dropout(self.dropout),
                keras.layers.Dense(input_shape[-1]),
            ]
        )

    def call(self, x):
        batch_size = keras.ops.shape(x)[0]
        probe = keras.ops.tile(self.probe, [batch_size, 1, 1])
        x = self.mha(probe, x)
        y = self.layer_norm(x)
        x = x + self.mlp_block(y)
        return x[:, 0]


class PaLIGemmaViT(keras.Model):
    "Untested. Arguments and names need revision."

    def __init__(
        self,
        num_heads=16,
        hidden_dim=1152,
        num_layers=27,
        intermeidate_dim=4304,
        pooling=None,
        num_classes=2048,
        classifier_activation=None,
        include_rescaling=False,
        name=None,
        **kwargs,
    ):
        inputs = keras.Input(shape=(None, None, 3), name="input_image")
        if include_rescaling:
            x = keras.layers.Rescaling(scale=1 / 255.0)(inputs)

        encoded = VitEncoder(
            hidden_dim,
            num_layers,
            num_heads,
            intermeidate_dim,
            name="image_encoder",
        )(inputs)
        if pooling == "map":
            pooled = MultiheadAttentionPooling(
                num_heads=num_heads, hidden_dim=hidden_dim
            )(encoded)
        elif pooling == "gap":
            pooled = ops.mean(encoded, axis=1)
        elif pooling == "0":
            pooled = x[:, 0]
        elif pooling is None:
            pooled = encoded
        else:
            raise ValueError(
                "Invalid value for argument `pooling`. "
                "Expected one of 'map', 'gap', None. "
                f"Received: pooling={pooling}"
            )

        outputs = keras.layers.Dense(
            num_classes, activation=classifier_activation, name="classifier"
        )(pooled)
        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.intermeidate_dim = intermeidate_dim
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.include_rescaling = include_rescaling
