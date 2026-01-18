"""MetaCLIP 2 text encoder implementation."""

from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.metaclip_2.metaclip_2_layers import (
    MetaCLIP2EncoderLayer,
)


@keras_hub_export("keras_hub.models.MetaCLIP2TextEncoder")
class MetaCLIP2TextEncoder(Backbone):
    """MetaCLIP 2 text encoder.

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
            for each transformer. Defaults to "quick_gelu".
        intermediate_output_index: optional int. The index of the intermediate
            output. If specified, the output will include an additional
            `"intermediate_output"` key.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. Defaults to 77.
        eos_token_id: int. The token ID for the EOS (end of sequence) token.
            Used for pooling. Defaults to 2.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done at float32 precision regardless of dtype.

    Output:
        A dictionary with keys:
        - `"sequence_output"`: The full sequence output with layer_norm applied,
            of shape `(batch_size, sequence_length, hidden_dim)`.
        - `"pooled_output"`: The pooled EOS token output of shape
            `(batch_size, hidden_dim)`.
        - `"intermediate_output"` (optional): If `intermediate_output_index`
            is specified, the output at that layer.
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
        eos_token_id=2,
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
        # MetaCLIP 2 models within a single model.
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
            MetaCLIP2EncoderLayer(
                hidden_dim,
                num_heads,
                intermediate_dim,
                intermediate_activation,
                use_causal_mask=True,  # `True` in the text encoder.
                dtype=dtype,
                name=f"{prefix}encoder_block_{i}",
            )
            for i in range(num_layers)
        ]
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-5, dtype=dtype, name=f"{prefix}layer_norm"
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
        # Apply layer_norm to full sequence (before pooling)
        x = self.layer_norm(x)
        sequence_output = x

        # Pool: extract at EOS token position
        # Find the position of EOS token (highest token ID position as fallback)
        eos_mask = ops.cast(
            ops.equal(token_id_input, eos_token_id), dtype="int32"
        )
        eos_positions = ops.argmax(eos_mask, axis=-1)
        eos_positions = ops.expand_dims(eos_positions, axis=-1)
        eos_positions = ops.expand_dims(eos_positions, axis=-1)
        pooled_output = ops.take_along_axis(sequence_output, eos_positions, axis=1)
        pooled_output = ops.squeeze(pooled_output, axis=1)

        outputs = {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
        }
        if intermediate_output_index is not None:
            outputs["intermediate_output"] = intermediate_output

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
        self.eos_token_id = eos_token_id

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
                "eos_token_id": self.eos_token_id,
            }
        )
        return config
