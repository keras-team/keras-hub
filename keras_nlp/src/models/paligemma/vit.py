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


class PaliGemmaAttention(keras.layers.Layer):
    """
    Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py # noqa: E501
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.hidden_dim // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`"
                f": {self.hidden_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.dropout_layer = keras.layers.Dropout(self.dropout)
        self.scale = self.head_dim**-0.5
        self.query_proj = keras.layers.Dense(
            units=self.hidden_dim,
            name="query_proj",
        )
        self.key_proj = keras.layers.Dense(
            units=self.hidden_dim,
            name="key_proj",
        )
        self.value_proj = keras.layers.Dense(
            units=self.hidden_dim,
            name="value_proj",
        )
        self.out_proj = keras.layers.Dense(
            units=self.hidden_dim,
            name="out_proj",
        )

    def build(self, input_shape):
        self.query_proj.build([None, None, self.hidden_dim])
        self.key_proj.build([None, None, self.hidden_dim])
        self.value_proj.build([None, None, self.hidden_dim])
        self.out_proj.build([None, None, self.hidden_dim])
        self.built = True

    def _transpose_for_scores(self, tensor, batch_size):
        """
        Adapted from https://github.com/huggingface/transformers/blob/8e164c5400b7b413c7b8fb32e35132001effc970/src/transformers/models/bert/modeling_tf_bert.py#L252 # noqa: E501
        """
        # [batch_size, seq_len, all_head_dim] ->
        # [batch_size, seq_len, num_heads, head_dim]
        tensor = ops.reshape(
            tensor, (batch_size, -1, self.num_heads, self.head_dim)
        )
        # [batch_size, seq_len, num_heads, head_dim] ->
        # [batch_size, num_heads, seq_len, head_dim]
        return ops.transpose(tensor, axes=[0, 2, 1, 3])

    def call(
        self,
        x,
        attention_mask=None,
        return_attention_scores=None,
        training=False,
    ):
        batch_size = ops.shape(x)[0]
        mixed_query_layer = self.query_proj(inputs=x)
        mixed_key_layer = self.key_proj(inputs=x)
        mixed_value_layer = self.value_proj(inputs=x)
        query_layer = self._transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self._transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self._transpose_for_scores(mixed_value_layer, batch_size)

        # Scaled dot product between key and query = raw attention scores.
        attention_scores = ops.matmul(
            query_layer, ops.transpose(key_layer, axes=[0, 1, 3, 2])
        )
        dk = ops.cast(ops.sqrt(self.head_dim), dtype=attention_scores.dtype)
        attention_scores = ops.divide(
            attention_scores, dk
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in the
            # call() function)
            attention_scores = ops.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        dropout_attention_probs = self.dropout_layer(
            inputs=attention_probs, training=training
        )

        attn_output = ops.matmul(dropout_attention_probs, value_layer)
        attn_output = ops.transpose(attn_output, axes=[0, 2, 1, 3])

        # (batch_size, seq_len_q, hidden_dim)
        attn_output = ops.reshape(
            attn_output, (batch_size, -1, self.hidden_dim)
        )

        attn_output = self.out_proj(attn_output, training=training)
        return (attn_output, attention_probs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }
        )
        return config


class VitEncoderBlock(keras.layers.Layer):

    def __init__(
        self,
        num_heads,
        intermediate_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = None
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim

    def compute_attention(self, x, mask=None):
        mask = None
        if mask is not None:
            mask = ops.cast(mask, dtype=x.dtype) if mask is not None else None

        return self.attn(
            x,
            attention_mask=mask,
        )[0]

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]
        self.attn = PaliGemmaAttention(
            self.hidden_dim,
            self.num_heads,
            name="multi_head_attention",
        )
        self.layer_norm_1 = keras.layers.LayerNormalization(
            epsilon=1e-6, name="layer_norm_1"
        )
        self.mlp_dense_1 = keras.layers.Dense(
            self.intermediate_dim, name="mlp_dense_1"
        )
        self.mlp_dense_2 = keras.layers.Dense(
            self.hidden_dim,
            name="mlp_dense_2",
        )
        self.layer_norm_2 = keras.layers.LayerNormalization(
            epsilon=1e-6, name="layer_norm_2"
        )
        self.attn.build(None)
        self.layer_norm_1.build([None, None, self.hidden_dim])
        self.mlp_dense_1.build([None, None, self.hidden_dim])
        self.mlp_dense_2.build([None, None, self.intermediate_dim])
        self.layer_norm_2.build([None, None, self.hidden_dim])
        self.built = True

    def call(self, x, mask=None):
        residual = x
        x = self.layer_norm_1(x)
        # mask = ops.ones_like(x) if mask is None else mask
        x = self.compute_attention(x, mask)
        x = x + residual
        residual = x
        x = self.mlp_dense_1(self.layer_norm_2(residual))
        x = keras.activations.gelu(x, approximate=True)
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
                "intermediate_dim": self.intermediate_dim,
            }
        )
        return config


class VitEncoder(keras.layers.Layer):

    def __init__(
        self,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        patch_size,
        **kwargs,
    ):
        if not config.keras_3():
            raise ValueError(
                "`PaliGemmaCausalLM` requires Keras 3. Run `pip install -U keras` "
                "upgrade your Keras version, or see "
                "https://keras.io/getting_started/ "
                "for more info on Keras versions and installation."
            )
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.patch_size = patch_size
        self.encoder_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="encoder_layer_norm"
        )
        self.vision_embeddings = VisionEmbeddings(
            hidden_dim=hidden_dim, patch_size=patch_size
        )
        self.resblocks = [
            VitEncoderBlock(
                self.num_heads,
                self.intermediate_dim,
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
        x = self.encoder_layer_norm(x)
        return x

    def compute_output_shape(self, inputs_shape):
        return [inputs_shape[0], inputs_shape[1], self.hidden_dim]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "patch_size": self.patch_size,
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
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
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


class PaliGemmaViT(keras.Model):
    "Untested. Arguments and names need revision."

    def __init__(
        self,
        num_heads=16,
        hidden_dim=1152,
        num_layers=27,
        intermediate_dim=4304,
        pooling=None,
        num_classes=2048,
        image_resolution=None,
        patch_size=14,
        classifier_activation=None,
        include_rescaling=False,
        name=None,
        **kwargs,
    ):
        inputs = keras.Input(
            shape=(image_resolution, image_resolution, 3), name="images"
        )
        if include_rescaling:
            x = keras.layers.Rescaling(scale=1 / 255.0)(inputs)

        self.pooled = None

        encoded = VitEncoder(
            hidden_dim,
            num_layers,
            num_heads,
            intermediate_dim,
            patch_size=patch_size,
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
        self.pooled = pooled
        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.include_rescaling = include_rescaling
        self.image_resolution = image_resolution
        self.patch_size = patch_size
        self.output_token_length = ops.cast(
            (image_resolution / patch_size) ** 2, dtype="int32"
        )
