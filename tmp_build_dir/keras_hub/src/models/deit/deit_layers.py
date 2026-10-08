import keras
from keras import ops

from keras_hub.src.utils.keras_utils import standardize_data_format


class DeiTEmbeddings(keras.layers.Layer):
    """Patches the image and embeds the patches.

    Args:
        image_size: tuple. Size of the input image (height, width).
        patch_size: tuple. patch_size: tuple. The size of each image
            patch as (patch_height, patch_width).
        hidden_dim: int. Dimensionality of the patch embeddings.
        num_channels: int. Number of channels in the input image. Defaults to
            `3`.
        data_format: str. `"channels_last"` or `"channels_first"`. Defaults to
            `None` (which uses `"channels_last"`).
        use_mask_token: bool. Whether to use a mask token. Defaults to `False`.
        dropout_rate: float. Dropout rate. Between 0 and 1. Defaults to
            `0.0`.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(
        self,
        image_size,
        patch_size,
        hidden_dim,
        num_channels=3,
        data_format=None,
        use_mask_token=False,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        num_patches = (image_size[0] // patch_size[0]) * (
            image_size[1] // patch_size[1]
        )
        num_positions = num_patches + 2

        # === Config ===
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.num_positions = num_positions
        self.data_format = standardize_data_format(data_format)
        self.use_mask_token = use_mask_token
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        if self.use_mask_token:
            self.mask_token = self.add_weight(
                shape=(1, 1, self.hidden_dim),
                initializer="zeros",
                dtype=self.variable_dtype,
                name="mask_token",
            )
        self.class_token = self.add_weight(
            shape=(
                1,
                1,
                self.hidden_dim,
            ),
            initializer="zeros",
            dtype=self.variable_dtype,
            name="class_token",
        )
        self.distillation_token = self.add_weight(
            shape=(
                1,
                1,
                self.hidden_dim,
            ),
            initializer="zeros",
            dtype=self.variable_dtype,
            name="distillation_token",
        )
        self.patch_embedding = keras.layers.Conv2D(
            filters=self.hidden_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            activation=None,
            dtype=self.dtype_policy,
            data_format=self.data_format,
            name="patch_embedding",
        )
        self.patch_embedding.build(input_shape)
        self.position_embedding = self.add_weight(
            shape=(
                1,
                self.num_positions,
                self.hidden_dim,
            ),  # Matches the shape in PyTorch
            initializer=keras.initializers.RandomNormal(
                stddev=0.02
            ),  # Equivalent to torch.randn()
            dtype=self.variable_dtype,
            trainable=True,
            name="position_embedding",
        )
        self.dropout = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout"
        )

        self.built = True

    def call(self, inputs, bool_masked_pos=None):
        patch_embeddings = self.patch_embedding(inputs)
        if self.data_format == "channels_first":
            patch_embeddings = ops.transpose(
                patch_embeddings, axes=(0, 2, 3, 1)
            )
        embeddings_shape = ops.shape(patch_embeddings)
        patch_embeddings = ops.reshape(
            patch_embeddings, [embeddings_shape[0], -1, embeddings_shape[-1]]
        )

        if bool_masked_pos is not None and self.use_mask_token:
            # Expand dimensions to match the embeddings
            bool_masked_pos_expanded = ops.expand_dims(
                bool_masked_pos, axis=-1
            )  # (batch_size, num_patches, 1)
            mask_token_expanded = ops.expand_dims(
                self.mask_token, axis=0
            )  # (1, 1, hidden_size)
            # Apply masking
            embeddings = ops.where(
                bool_masked_pos_expanded, mask_token_expanded, patch_embeddings
            )

        class_token = ops.tile(self.class_token, (embeddings_shape[0], 1, 1))
        distillation_token = ops.tile(
            self.distillation_token, (embeddings_shape[0], 1, 1)
        )
        embeddings = ops.concatenate(
            [class_token, distillation_token, patch_embeddings], axis=1
        )
        position_embedding = self.position_embedding
        embeddings = ops.add(embeddings, position_embedding)
        embeddings = self.dropout(embeddings)
        return embeddings

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.num_positions,
            self.hidden_dim,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "num_channels": self.num_channels,
                "num_patches": self.num_patches,
                "num_positions": self.num_positions,
                "use_mask_token": self.use_mask_token,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class DeiTIntermediate(keras.layers.Layer):
    """DeiTIntermediate block.
    Args:
        intermediate_dim: int. Dimensionality of the intermediate MLP layer.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(
        self,
        intermediate_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # === Config ===
        self.intermediate_dim = intermediate_dim

    def build(self, input_shape):
        self.dense = keras.layers.Dense(
            units=self.intermediate_dim,
            activation="gelu",
            dtype=self.dtype_policy,
            name="dense",
        )
        self.dense.build(input_shape)
        self.built = True

    def call(self, inputs):
        out = self.dense(inputs)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
            }
        )
        return config


class DeiTOutput(keras.layers.Layer):
    """DeiT Output layer implementation.
    Args:
        hidden_dim: int. Dimensionality of the patch embeddings.
        dropout_rate: float. Dropout rate. Between 0 and 1. Defaults to
            `0.0`.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(self, hidden_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.dense = keras.layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="output"
        )
        self.dense.build(input_shape)

        self.dropout = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout"
        )
        # Mark this layer as built
        self.built = True

    def call(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)  # Linear transformation
        hidden_states = self.dropout(hidden_states)  # Apply dropout
        hidden_states = hidden_states + input_tensor  # Residual connection
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class DeiTEncoderBlock(keras.layers.Layer):
    """DeiT encoder block.
    Args:
        num_heads: int. Number of attention heads.
        hidden_dim: int. Dimensionality of the hidden representations.
        intermediate_dim: int. Dimensionality of the intermediate MLP layer.
        use_mha_bias: bool. Whether to use bias in the multi-head attention
            layer. Defaults to `True`.
        dropout_rate: float. Dropout rate. Between 0 and 1. Defaults to
            `0.0`.
        attention_dropout: float. Dropout rate for the attention mechanism.
            Between 0 and 1. Defaults to `0.0`.
        layer_norm_epsilon: float. Small float value for layer normalization
            stability. Defaults to `1e-6`.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        intermediate_dim,
        use_mha_bias=True,
        dropout_rate=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        key_dim = hidden_dim // num_heads

        # === Config ===
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.key_dim = key_dim
        self.use_mha_bias = use_mha_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        # Attention block
        self.layer_norm_1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="ln_1",
            dtype=self.dtype_policy,
        )
        self.layer_norm_1.build(input_shape)
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            use_bias=self.use_mha_bias,
            dropout=self.attention_dropout,
            name="mha",
            dtype=self.dtype_policy,
        )
        self.mha.build(input_shape, input_shape)

        # MLP block
        self.layer_norm_2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="ln_2",
            dtype=self.dtype_policy,
        )
        self.layer_norm_2.build((None, None, self.hidden_dim))

        # Intermediate Layer
        self.mlp = DeiTIntermediate(
            self.intermediate_dim, dtype=self.dtype_policy, name="mlp"
        )
        self.mlp.build((None, None, self.hidden_dim))

        # Output Layer
        self.output_layer = DeiTOutput(
            self.hidden_dim,
            self.dropout_rate,
            dtype=self.dtype_policy,
            name="output_layer",
        )

        self.output_layer.build((None, None, self.intermediate_dim))

        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        return_attention_scores=False,
    ):
        attention_scores = None
        x = self.layer_norm_1(hidden_states)
        if return_attention_scores:
            x, attention_scores = self.mha(
                x,
                x,
                attention_mask=attention_mask,
                return_attention_scores=return_attention_scores,
            )
        else:
            x = self.mha(
                x,
                x,
                attention_mask=attention_mask,
            )

        x = x + hidden_states
        y = self.layer_norm_2(x)
        y = self.mlp(y)
        y = self.output_layer(y, x)

        return y, attention_scores

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "key_dim": self.key_dim,
                "use_mha_bias": self.use_mha_bias,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class DeiTEncoder(keras.layers.Layer):
    """DeiT Encoder class.
    Args:
        num_layers: int. Number of Transformer encoder blocks.
        num_heads: int. Number of attention heads.
        hidden_dim: int. Dimensionality of the hidden representations.
        intermediate_dim: int. Dimensionality of the intermediate MLP layer.
        use_mha_bias: bool. Whether to use bias in the multi-head attention
            layer. Defaults to `True`.
        dropout_rate: float. Dropout rate. Between 0 and 1. Defaults to
            `0.0`.
        attention_dropout: float. Dropout rate for the attention mechanism.
            Between 0 and 1. Defaults to `0.0`.
        layer_norm_epsilon: float. Small float value for layer normalization
            stability. Defaults to `1e-6`.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        use_mha_bias=True,
        dropout_rate=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # === Config ===
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.use_mha_bias = use_mha_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        self.encoder_layers = []
        for i in range(self.num_layers):
            encoder_block = DeiTEncoderBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                use_mha_bias=self.use_mha_bias,
                dropout_rate=self.dropout_rate,
                attention_dropout=self.attention_dropout,
                layer_norm_epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name=f"transformer_block_{i + 1}",
            )
            encoder_block.build((None, None, self.hidden_dim))
            self.encoder_layers.append(encoder_block)

        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="ln",
        )
        self.layer_norm.build((None, None, self.hidden_dim))

        self.built = True

    def call(
        self,
        hidden_states,
        attention_masks=None,
        output_hidden_states=False,
        return_attention_scores=False,
    ):
        seq_len = ops.shape(hidden_states)[1]  # Sequence length
        hidden_dim = ops.shape(hidden_states)[2]  # Hidden size

        # Ensure valid tensor output even if disabled
        all_hidden_states = (
            ops.empty(shape=(0, seq_len, hidden_dim), dtype=hidden_states.dtype)
            if not output_hidden_states
            else ()
        )

        all_self_attentions_scores = (
            ops.empty(
                shape=(0, self.num_heads, seq_len, seq_len),
                dtype=hidden_states.dtype,
            )
            if not return_attention_scores
            else ()
        )

        for i in range(self.num_layers):
            attention_mask = (
                attention_masks[i] if attention_masks is not None else None
            )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, scores = self.encoder_layers[i](
                hidden_states,
                attention_mask=attention_mask,
                return_attention_scores=return_attention_scores,
            )
            if return_attention_scores:
                all_self_attentions_scores = all_self_attentions_scores + (
                    scores,
                )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, all_hidden_states, all_self_attentions_scores

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "key_dim": self.key_dim,
                "use_mha_bias": self.use_mha_bias,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
