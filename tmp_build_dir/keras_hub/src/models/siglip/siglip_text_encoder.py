from keras import initializers
from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.siglip.siglip_layers import SigLIPEncoderLayer
from keras_hub.src.models.siglip.siglip_layers import SigLIPTextEmbedding


@keras_hub_export("keras_hub.models.SigLIPTextEncoder")
class SigLIPTextEncoder(Backbone):
    """SigLIP text core network with hyperparameters.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        embedding_dim: int. The output dimension of the embedding layer.
        hidden_dim: int. The size of the transformer hidden state at the end
            of each transformer layer.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        intermediate_activation: activation function. The activation that
            is used for the first Dense layer in a two-layer feedforward network
            for each transformer. Defaults to `"gelu_approximate"`.
        layer_norm_epsilon: float. The epsilon for the layer normalization.
            Defaults to `1e-6`.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. Defaults to `64`.
        projection_dim: int. The size of the projection in the head. If not
            specified, set to `hidden_dim`. Defaults to `None`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done a float32 precision regardless of dtype.
    """

    def __init__(
        self,
        vocabulary_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        intermediate_activation="gelu_approximate",
        layer_norm_epsilon=1e-6,
        max_sequence_length=64,
        projection_dim=None,
        dtype=None,
        name=None,
        **kwargs,
    ):
        projection_dim = projection_dim or hidden_dim
        # `prefix` is used to prevent duplicate name when utilizing multiple
        # SigLIP encoders within a single model.
        prefix = str(name) + "_" if name is not None else ""

        # === Layers ===
        self.embedding = SigLIPTextEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            dtype=dtype,
            name=f"{prefix}embedding",
        )
        self.encoder_layers = [
            SigLIPEncoderLayer(
                hidden_dim,
                num_heads,
                intermediate_dim,
                intermediate_activation,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                name=f"{prefix}encoder_block_{i}",
            )
            for i in range(num_layers)
        ]
        self.post_layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name=f"{prefix}post_layer_norm",
        )
        self.head = layers.Dense(
            projection_dim,
            kernel_initializer=initializers.LecunNormal(),
            dtype=dtype,
            name=f"{prefix}head",
        )

        # === Functional Model ===
        token_id_input = layers.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        x = self.embedding(token_id_input)
        for _, block in enumerate(self.encoder_layers):
            x = block(x)
        x = self.post_layer_norm(x)

        # Assuming "sticky" EOS tokenization, last token is always EOS.
        x = x[:, -1, :]
        x = self.head(x)
        outputs = x
        super().__init__(
            inputs={"token_ids": token_id_input},
            outputs=outputs,
            dtype=dtype,
            name=name,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = intermediate_activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.max_sequence_length = max_sequence_length
        self.projection_dim = projection_dim

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "intermediate_activation": self.intermediate_activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "max_sequence_length": self.max_sequence_length,
                "projection_dim": self.projection_dim,
            }
        )
        return config
