from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.clip.clip_encoder_block import CLIPEncoderBlock


@keras_hub_export("keras_hub.models.CLIPTextEncoder")
class CLIPTextEncoder(Backbone):
    """CLIP text core network with hyperparameters.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        embedding_dim: int. The output dimension of the embedding layer.
        hidden_dim: int. The size of the transformer hidden state at the end
            of each transformer layer.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        intermediate_activation: activation function. The activation that
            is used for the first Dense layer in a two-layer feedforward network
            for each transformer.
        intermediate_output_index: optional int. The index of the intermediate
            output. If specified, the output will become a dictionary with two
            keys `"sequence_output"` and `"intermediate_output"`.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume.
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
        intermediate_activation="quick_gelu",
        intermediate_output_index=None,
        max_sequence_length=77,
        dtype=None,
        name=None,
        **kwargs,
    ):
        if (
            intermediate_output_index is not None
            and intermediate_output_index < 0
        ):
            intermediate_output_index += num_layers

        # `prefix` is used to prevent duplicate name when utilizing multiple
        # CLIP models within a single model, such as in StableDiffusion3.
        prefix = str(name) + "_" if name is not None else ""

        # === Layers ===
        self.embedding = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            dtype=dtype,
            name=f"{prefix}embedding",
        )
        self.encoder_layers = [
            CLIPEncoderBlock(
                hidden_dim,
                num_heads,
                intermediate_dim,
                intermediate_activation,
                dtype=dtype,
                name=f"{prefix}encoder_block_{i}",
            )
            for i in range(num_layers)
        ]
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-6, dtype=dtype, name=f"{prefix}layer_norm"
        )

        # === Functional Model ===
        token_id_input = layers.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        x = self.embedding(token_id_input)
        intermediate_output = None
        for i, block in enumerate(self.encoder_layers):
            x = block(x)
            if i == intermediate_output_index:
                intermediate_output = x
        x = self.layer_norm(x)
        sequence_output = x

        if intermediate_output_index is not None:
            outputs = {
                "sequence_output": sequence_output,
                "intermediate_output": intermediate_output,
            }
        else:
            outputs = sequence_output
        super().__init__(
            inputs={"token_ids": token_id_input},
            outputs=outputs,
            dtype=dtype,
            name=name,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = intermediate_activation
        self.intermediate_output_index = intermediate_output_index

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
                "intermediate_output_index": self.intermediate_output_index,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config
