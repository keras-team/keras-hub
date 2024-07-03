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
import keras
from keras import ops


class PaliGemmaVitEmbeddings(keras.layers.Layer):
    def __init__(
        self,
        image_size,
        patch_size,
        hidden_dim,
        num_channels=3,
        dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.patch_embedding = keras.layers.Conv2D(
            filters=self.hidden_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            activation=None,
            dtype=dtype,
            name="embedding_conv",
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = keras.layers.Embedding(
            self.num_positions,
            self.hidden_dim,
            dtype=dtype,
            name="position_embedding",
        )

        self.position_ids = ops.expand_dims(
            ops.arange(self.num_positions), axis=0
        )

    def build(self, input_shape):
        self.patch_embedding.build(input_shape)
        self.position_embedding.build([1, self.num_positions])
        self.built = True

    def call(self, input_tokens):
        x = self.patch_embedding(input_tokens)
        input_shape = ops.shape(x)
        x = ops.reshape(x, [input_shape[0], -1, input_shape[-1]])
        x = x + self.position_embedding(self.position_ids)
        return x

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.num_patches,
            self.hidden_dim,
        )


class PaliGemmaVitAttention(keras.layers.Layer):
    """
    Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py # noqa: E501
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout=0.0,
        dtype=None,
        **kwargs,
    ):
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
        self.dropout_layer = keras.layers.Dropout(
            self.dropout,
            dtype=dtype,
            name="dropout",
        )
        self.scale = self.head_dim**-0.5
        self.query_proj = keras.layers.Dense(
            units=self.hidden_dim,
            dtype=dtype,
            name="query_proj",
        )
        self.key_proj = keras.layers.Dense(
            units=self.hidden_dim,
            dtype=dtype,
            name="key_proj",
        )
        self.value_proj = keras.layers.Dense(
            units=self.hidden_dim,
            dtype=dtype,
            name="value_proj",
        )
        self.out_proj = keras.layers.Dense(
            units=self.hidden_dim,
            dtype=dtype,
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


class PaliGemmaVitEncoderBlock(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        intermediate_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim

    def compute_attention(self, x, mask=None):
        mask = None
        if mask is not None:
            mask = ops.cast(mask, dtype=x.dtype) if mask is not None else None
        return self.attn(x, attention_mask=mask)[0]

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.attn = PaliGemmaVitAttention(
            hidden_dim,
            self.num_heads,
            dtype=self.dtype_policy,
            name="multi_head_attention",
        )
        self.layer_norm_1 = keras.layers.LayerNormalization(
            epsilon=1e-6,
            dtype=self.dtype_policy,
            name="layer_norm_1",
        )
        self.mlp_dense_1 = keras.layers.Dense(
            self.intermediate_dim,
            dtype=self.dtype_policy,
            name="mlp_dense_1",
        )
        self.mlp_dense_2 = keras.layers.Dense(
            hidden_dim,
            dtype=self.dtype_policy,
            name="mlp_dense_2",
        )
        self.layer_norm_2 = keras.layers.LayerNormalization(
            epsilon=1e-6,
            dtype=self.dtype_policy,
            name="layer_norm_2",
        )
        self.attn.build(None)
        self.layer_norm_1.build([None, None, hidden_dim])
        self.mlp_dense_1.build([None, None, hidden_dim])
        self.mlp_dense_2.build([None, None, self.intermediate_dim])
        self.layer_norm_2.build([None, None, hidden_dim])
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
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
            }
        )
        return config


class PaliGemmaVitEncoder(keras.layers.Layer):
    def __init__(
        self,
        patch_size,
        image_size,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.encoder_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-6,
            dtype=dtype,
            name="encoder_layer_norm",
        )
        self.vision_embeddings = PaliGemmaVitEmbeddings(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            image_size=image_size,
            dtype=dtype,
            name="encoder_embeddings",
        )
        self.resblocks = [
            PaliGemmaVitEncoderBlock(
                self.num_heads,
                self.intermediate_dim,
                dtype=dtype,
                name=f"encoder_block_{i}",
            )
            for i in range(self.num_layers)
        ]

    def build(self, input_shape):
        self.vision_embeddings.build(input_shape)
        for block in self.resblocks:
            block.build([None, None, self.hidden_dim])
        self.encoder_layer_norm.build([None, None, self.hidden_dim])
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
                "image_size": self.image_size,
            }
        )
        return config


class MultiHeadAttentionPooling(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim=None,
        num_heads=12,
        dropout=0.0,
        **kwargs,
    ):
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
            dtype=self.dtype_policy,
        )
        self.mha = keras.layers.MultiHeadAttention(
            key_dim=input_shape[-1] // self.num_heads,
            num_heads=self.num_heads,
            dtype=self.dtype_policy,
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-6,
            dtype=self.dtype_policy,
        )
        self.mlp_block = keras.Sequential(
            [
                keras.layers.Dense(
                    self.hidden_dim,
                    activation="gelu",
                    dtype=self.dtype_policy,
                ),
                keras.layers.Dropout(
                    self.dropout,
                    dtype=self.dtype_policy,
                ),
                keras.layers.Dense(
                    input_shape[-1],
                    dtype=self.dtype_policy,
                ),
            ]
        )

    def call(self, x):
        batch_size = ops.shape(x)[0]
        probe = ops.tile(self.probe, [batch_size, 1, 1])
        x = self.mha(probe, x)
        y = self.layer_norm(x)
        x = x + self.mlp_block(y)
        return x[:, 0]


class PaliGemmaVit(keras.Model):
    """Vision Transformer (ViT) model for PaliGemma.

    Args:
        image_size: int. The height/width of the image. Both height and width is
            expected to be the same.
        include_rescaling: bool. If true, the image input will be rescaled from
            the range `[0, 255]`, to the range `[0, 1]`.
        patch_size: int. The size of each square patch in the input image.
        num_heads: int. The number of attention heads for the vision(image)
            transformer encoder.
        hidden_dim: int. The size of the transformer hidden state at the end
            of each vision transformer layer.
        num_layers: int. The number of transformer layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for transformer.
        num_classes: int. The number of output classes. If this model is used
            as a image classifier, this value would correspond to the number of
            output classes.
        pooling: string. The encoded vision embeddings are pooled using the
            specified polling setting. The accepted values are `"map"`, `"gap"`,
            `"zero"` or `None`. Defaults to `None`.
        classifier_activation: activation fucntion. The activation that is used
            for final output classification
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done a float32 precision regardless of dtype.

    Example:
    ```python
    image = np.random.rand(224, 224, 3)
    vit_model = PaliGemmaVit(image_size=224)
    # The output will be of shape:
    # [batch_size, image_sequence_length, num_classes]
    output = vit_model([image])
    ```
    """

    def __init__(
        self,
        image_size,
        patch_size,
        num_heads,
        hidden_dim,
        num_layers,
        intermediate_dim,
        num_classes,
        include_rescaling=True,
        pooling=None,
        classifier_activation=None,
        dtype=None,
        **kwargs,
    ):
        # === Functional Model ===
        image_input = keras.Input(
            shape=(image_size, image_size, 3), name="images"
        )
        x = image_input  # Intermediate result.
        if include_rescaling:
            rescaling = keras.layers.Rescaling(
                scale=1.0 / 127.5, offset=-1.0, name="rescaling"
            )
            x = rescaling(image_input)
        x = PaliGemmaVitEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            patch_size=patch_size,
            image_size=image_size,
            dtype=dtype,
            name="image_encoder",
        )(x)
        if pooling == "map":
            x = MultiHeadAttentionPooling(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dtype=dtype,
                name="pooling",
            )(x)
        elif pooling == "gap":
            x = ops.mean(x, axis=1)
        elif pooling == "zero":
            x = x[:, 0]
        elif pooling is None:
            x = x
        else:
            raise ValueError(
                "Invalid value for argument `pooling`. "
                "Expected one of 'map', 'gap', None. "
                f"Received: pooling={pooling}"
            )
        outputs = keras.layers.Dense(
            num_classes,
            activation=classifier_activation,
            dtype=dtype,
            name="image_classifier",
        )(x)
        super().__init__(
            inputs=image_input,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        self.pooling = pooling
        self.num_classes = num_classes
        self.image_size = image_size
        self.include_rescaling = include_rescaling
        self.patch_size = patch_size
        self.classifier_activation = keras.activations.get(
            classifier_activation
        )
        self.image_sequence_length = int((image_size / patch_size) ** 2)
        self.dtype_policy = keras.dtype_policies.get(dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "intermediate_dim": self.intermediate_dim,
                "pooling": self.pooling,
                "num_classes": self.num_classes,
                "classifier_activation": keras.activations.serialize(
                    self.classifier_activation
                ),
                "image_size": self.image_size,
                "include_rescaling": self.include_rescaling,
                "patch_size": self.patch_size,
            }
        )
        return config
