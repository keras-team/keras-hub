import math

from keras import initializers
from keras import layers
from keras import ops

from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import gelu_approximate
from keras_hub.src.utils.keras_utils import standardize_data_format


class SigLIPVisionEmbedding(layers.Layer):
    """A layer that converts images into patches.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        patch_size: int. The size of one side of each patch.
        image_size: int. The size of the input images.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        hidden_dim,
        patch_size,
        image_size,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        self.data_format = standardize_data_format(data_format)
        self.num_positions = (image_size // patch_size) ** 2

        self.patch_embedding = layers.Conv2D(
            hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            kernel_initializer=initializers.LecunNormal(),
            data_format=data_format,
            dtype=self.dtype_policy,
            name="patch_embedding",
        )
        self.position_embedding = layers.Embedding(
            self.num_positions,
            hidden_dim,
            embeddings_initializer=initializers.RandomNormal(
                stddev=1.0 / math.sqrt(hidden_dim)
            ),
            dtype=self.dtype_policy,
            name="position_embedding",
        )

    def build(self, input_shape):
        self.position_ids = self.add_weight(
            shape=(1, self.num_positions),
            initializer="zeros",
            # Let the backend determine the int dtype. For example, tf
            # requires int64 for correct device placement, whereas jax and torch
            # don't.
            dtype=int,
            trainable=False,
            name="position_ids",
        )
        self.position_ids.assign(
            ops.expand_dims(ops.arange(0, self.num_positions), axis=0)
        )
        self.patch_embedding.build(input_shape)
        self.position_embedding.build(self.position_ids.shape)

    def call(self, inputs, training=None):
        x = inputs
        batch_size = ops.shape(x)[0]
        patch_embeddings = self.patch_embedding(x, training=training)
        if self.data_format == "channels_last":
            patch_embeddings = ops.reshape(
                patch_embeddings, (batch_size, -1, self.hidden_dim)
            )
        else:
            patch_embeddings = ops.reshape(
                patch_embeddings, (batch_size, self.hidden_dim, -1)
            )
            patch_embeddings = ops.transpose(patch_embeddings, (0, 2, 1))
        position_embeddings = self.position_embedding(self.position_ids)
        return ops.add(patch_embeddings, position_embeddings)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], None, self.hidden_dim]
        if self.data_format == "channels_last":
            if input_shape[1] is not None and input_shape[2] is not None:
                patch_num = input_shape[1] // self.patch_size
                output_shape[1] = patch_num**2 + 1
        else:
            if input_shape[2] is not None and input_shape[3] is not None:
                patch_num = input_shape[2] // self.patch_size
                output_shape[1] = patch_num**2 + 1
        return output_shape


class SigLIPTextEmbedding(layers.Layer):
    """A layer which sums a token and position embedding.

    Args:
        vocabulary_size: The size of the vocabulary.
        sequence_length: The maximum length of input sequence.
        embedding_dim: The output dimension of the embedding layer
        tie_weights: Boolean, whether or not the matrix for embedding and
            the matrix for the `reverse` projection should share the same
            weights. Defaults to `True`.
        embeddings_initializer: The initializer to use for the Embedding
            Layers. Defaults to `"normal"`.
        mask_zero: Boolean, whether or not the input value 0 is a special
            "padding" value that should be masked out.
            This is useful when using recurrent layers which may take variable
            length input. If this is True, then all subsequent layers in the
            model need to support masking or an exception will be raised.
            If mask_zero` is set to True, as a consequence, index 0 cannot be
            used in the vocabulary
            (input_dim should equal size of vocabulary + 1). Defaults to
            `False`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.
    """

    def __init__(
        self,
        vocabulary_size,
        sequence_length,
        embedding_dim,
        tie_weights=True,
        embeddings_initializer="normal",
        mask_zero=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocabulary_size = int(vocabulary_size)
        self.sequence_length = int(sequence_length)
        self.embedding_dim = int(embedding_dim)
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.token_embedding = ReversibleEmbedding(
            vocabulary_size,
            embedding_dim,
            tie_weights=tie_weights,
            embeddings_initializer=clone_initializer(
                self.embeddings_initializer
            ),
            mask_zero=mask_zero,
            dtype=self.dtype_policy,
            name="token_embedding",
        )
        self.position_embedding = ReversibleEmbedding(
            sequence_length,
            embedding_dim,
            tie_weights=tie_weights,
            embeddings_initializer=clone_initializer(
                self.embeddings_initializer
            ),
            mask_zero=mask_zero,
            dtype=self.dtype_policy,
            name="position_embedding",
        )
        self.supports_masking = self.token_embedding.supports_masking

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self.token_embedding.build(input_shape)
        self.position_embedding.build((1, self.sequence_length))
        self.position_ids = self.add_weight(
            shape=(1, self.sequence_length),
            initializer="zeros",
            # Let the backend determine the int dtype. For example, tf
            # requires int64 for correct device placement, whereas jax and torch
            # don't.
            dtype=int,
            trainable=False,
            name="position_ids",
        )
        self.position_ids.assign(
            ops.expand_dims(ops.arange(0, self.sequence_length), axis=0)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "sequence_length": self.sequence_length,
                "embedding_dim": self.embedding_dim,
                "embeddings_initializer": initializers.serialize(
                    self.embeddings_initializer
                ),
                "tie_weights": self.token_embedding.tie_weights,
                "mask_zero": self.token_embedding.mask_zero,
            }
        )
        return config

    def call(self, inputs):
        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embedding(self.position_ids)
        outputs = ops.add(embedded_tokens, embedded_positions)
        return outputs

    def compute_mask(self, inputs, mask=None):
        return self.token_embedding.compute_mask(inputs, mask=mask)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape) + (self.embedding_dim,)


class SigLIPMLP(layers.Layer):
    """A SigLIP MLP block.

    Args:
        hidden_dim: int. The number of units in the output layer.
        intermediate_dim: int. The number of units in the intermediate layer.
        activation: str of callable. Activation to use in the intermediate
            layer.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(self, hidden_dim, intermediate_dim, activation, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation

        if activation == "gelu_approximate":
            activation = gelu_approximate

        self.fc1 = layers.Dense(
            self.intermediate_dim,
            activation=activation,
            bias_initializer=initializers.RandomNormal(stddev=1e-6),
            dtype=self.dtype_policy,
            name="fc1",
        )
        self.fc2 = layers.Dense(
            self.hidden_dim,
            bias_initializer=initializers.RandomNormal(stddev=1e-6),
            dtype=self.dtype_policy,
            name="fc2",
        )

    def build(self, inputs_shape):
        self.fc1.build(inputs_shape)
        inputs_shape = self.fc1.compute_output_shape(inputs_shape)
        self.fc2.build(inputs_shape)

    def call(self, inputs):
        hidden_states = self.fc1(inputs)
        return self.fc2(hidden_states)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "activation": self.activation,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        outputs_shape = list(inputs_shape)
        outputs_shape[-1] = self.hidden_dim
        return outputs_shape


class SigLIPEncoderLayer(layers.Layer):
    """A SigLIP encoder layer.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        num_heads: int. Number of attention heads.
        intermediate_dim: int. The number of units in the intermediate layers.
        intermediate_activation: str or callable. Activation to use in the
            hidden layers.
        layer_norm_epsilon: float. The epsilon for the layer normalization.
            Defaults to `1e-6`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        intermediate_activation="gelu_approximate",
        use_causal_mask=False,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "`hidden_dim` must be divisible by `num_heads`. "
                f"Received: hidden_dim={hidden_dim}, num_heads={num_heads}"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.intermediate_dim = int(intermediate_dim)
        self.intermediate_activation = intermediate_activation
        self.use_causal_mask = bool(use_causal_mask)
        self.layer_norm_epsilon = layer_norm_epsilon

        self.self_attn = layers.MultiHeadAttention(
            num_heads,
            hidden_dim // num_heads,
            dtype=self.dtype_policy,
            name="self_attn",
        )
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm1",
        )
        self.mlp = SigLIPMLP(
            hidden_dim,
            intermediate_dim,
            intermediate_activation,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm2",
        )

    def build(self, inputs_shape):
        self.layer_norm1.build(inputs_shape)
        self.self_attn.build(inputs_shape, inputs_shape, inputs_shape)
        self.layer_norm2.build(inputs_shape)
        self.mlp.build(inputs_shape)

    def compute_output_shape(self, inputs_shape):
        outputs_shape = list(inputs_shape)
        outputs_shape[-1] = self.hidden_dim
        return outputs_shape

    def call(self, inputs, training=None):
        residual = inputs
        x = self.layer_norm1(inputs)
        x = self.self_attn(
            x, x, x, training=training, use_causal_mask=self.use_causal_mask
        )
        x = ops.add(residual, x)

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x, training=training)
        x = ops.add(residual, x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "intermediate_activation": self.intermediate_activation,
                "use_causal_mask": self.use_causal_mask,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class SigLIPMultiHeadAttentionPooling(layers.Layer):
    """A SigLIP multi-headed attention pooling layer.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        intermediate_dim: int. The number of units in the intermediate layers.
        num_heads: int. Number of attention heads.
        activation: str or callable. Activation to use in the MLP.
        layer_norm_epsilon: float. The epsilon for the layer normalization.
            Defaults to `1e-6`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_heads,
        activation,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.num_heads = int(num_heads)
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon

        self.attention = layers.MultiHeadAttention(
            num_heads,
            hidden_dim // num_heads,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layernorm",
        )
        self.mlp = SigLIPMLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            activation=self.activation,
            dtype=self.dtype_policy,
            name="mlp",
        )

    def build(self, inputs_shape):
        self.probe = self.add_weight(
            (1, 1, self.hidden_dim),
            initializer=initializers.GlorotUniform(),
            dtype=self.dtype_policy.variable_dtype,
        )
        self.attention.build(
            query_shape=(inputs_shape[0], 1, self.hidden_dim),
            value_shape=inputs_shape,
        )
        inputs_shape = self.attention.compute_output_shape(
            query_shape=(inputs_shape[0], 1, self.hidden_dim),
            value_shape=inputs_shape,
        )
        self.layer_norm.build(inputs_shape)
        self.mlp.build(inputs_shape)

    def call(self, inputs, training=None):
        batch_size = ops.shape(inputs)[0]
        probes = ops.repeat(self.probe, repeats=batch_size, axis=0)
        hidden_states = self.attention(
            probes, inputs, inputs, training=training
        )
        residuals = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = ops.add(residuals, self.mlp(hidden_states))
        return hidden_states[:, 0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "activation": self.activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        return (inputs_shape[0], self.hidden_dim)


class SigLIPHead(layers.Layer):
    """The head layer of SigLIP.

    `SigLIP` takes `vision_embedding` and `text_embedding` as inputs to
    compute the corresponding logits. Both embeddings are L2 normalized and used
    to compute pairwise cosine similarity. The resulting logits are then scaled
    and added by learnable `logit_scale`  and `logit_bias` parameters.

    Call arguments:
        vision_embedding: A tensor of shape `(batch_size, hidden_dim)`.
        text_embedding: A tensor of shape `(batch_size, hidden_dim)`.
    """

    def build(self, input_shape):
        self.logit_scale = self.add_weight(
            shape=(),
            initializer=initializers.Constant(math.log(1.0)),
            trainable=True,
            dtype=self.variable_dtype,
            name="logit_scale",
        )
        self.logit_bias = self.add_weight(
            shape=(),
            initializer=initializers.Zeros(),
            trainable=True,
            dtype=self.variable_dtype,
            name="logit_bias",
        )

    def call(self, vision_embedding, text_embedding):
        normalized_vision_embedding = ops.sqrt(
            ops.sum(ops.power(vision_embedding, 2), axis=-1, keepdims=True)
        )
        normalized_text_embedding = ops.sqrt(
            ops.sum(ops.power(text_embedding, 2), axis=-1, keepdims=True)
        )
        vision_embedding = ops.divide(
            vision_embedding, normalized_vision_embedding
        )
        text_embedding = ops.divide(text_embedding, normalized_text_embedding)
        text_logits = ops.add(
            ops.multiply(
                ops.matmul(text_embedding, ops.transpose(vision_embedding)),
                ops.exp(self.logit_scale),
            ),
            self.logit_bias,
        )
        vision_logits = ops.transpose(text_logits)
        return vision_logits, text_logits

    def compute_output_shape(
        self, vision_embedding_shape, text_embedding_shape
    ):
        vision_logits_shape = (
            vision_embedding_shape[0],
            text_embedding_shape[0],
        )
        text_logits_shape = (
            text_embedding_shape[0],
            vision_embedding_shape[0],
        )
        return vision_logits_shape, text_logits_shape
