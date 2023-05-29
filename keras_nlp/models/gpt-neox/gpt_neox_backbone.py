import tensorflow as tf
from tensorflow import keras

from keras_nlp.models.backbone import Backbone
from keras_nlp.layers.position_embedding import PositionEmbedding
from tensorflow.experimental import dtensor
from tensorflow.experimental.dtensor import Layout

from gpt_neox_decoder import GPTNeoXDecoder

def _gpt_2_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)

class GPT2Backbone(Backbone):

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=1024,
        **kwargs,
    ):
        # Inputs
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed tokens, positions.
        token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=_gpt_2_kernel_initializer(stddev=0.01),
            name="token_embedding",
        )(token_ids)

        # Can't use `TokenAndPositionEmbedding` layer here because of different
        # initializers.
        position_embedding = PositionEmbedding(
            initializer=_gpt_2_kernel_initializer(stddev=0.02),
            sequence_length=max_sequence_length,
            name="position_embedding",
        )(token_embedding)

        # Sum and apply dropout to embeddings.
        x = keras.layers.Add(name="embeddings_add")(
            (token_embedding, position_embedding)
        )
        x = keras.layers.Dropout(
            dropout,
            name="embeddings_dropout",
        )(x)

        # Apply successive transformer decoder blocks.
        for i in range(num_layers):
            x = GPTNeoXDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                layer_norm_epsilon=1e-05,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=True
                ),
                kernel_initializer=_gpt_2_kernel_initializer(stddev=0.02),
                normalize_first=True,
                name=f"transformer_layer_{i}",
            )(x, decoder_padding_mask=padding_mask)

        sequence_output = keras.layers.LayerNormalization(
            name="layer_norm",
            axis=-1,
            epsilon=1e-05,
            dtype=tf.float32,
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=sequence_output,
            **kwargs,
        )
        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config

    @property
    def token_embedding(self):
        return self.get_layer("token_embedding")

    @classmethod
    def create_layout_map(cls, mesh):

        # We assert the mesh is 2D, and assume the first mesh dim is for data
        # parallel and the second dim is for model parallel.
        mesh_shape = mesh.shape()
        if len(mesh_shape) != 2:
            raise ValueError(
                f"Expect to create layout based on 2D mesh, received {mesh}"
            )
        _, model_dim = mesh.dim_names
        unshard_dim = dtensor.UNSHARDED

        layout_map = keras.dtensor.experimental.LayoutMap(mesh=mesh)
        # Embedding sharding
        layout_map[r".*embeddings"] = Layout([unshard_dim, model_dim], mesh)

        # Transformer block sharding
        layout_map[r".*_(query|key|value)_dense.kernel"] = Layout(
            [unshard_dim, unshard_dim, model_dim], mesh
        )
        layout_map[r".*_(query|key|value)_dense.bias"] = Layout(
            [model_dim, unshard_dim], mesh
        )
        layout_map[r".*_feedforward_intermediate_dense.kernel"] = Layout(
            [unshard_dim, model_dim], mesh
        )
        layout_map[r".*_feedforward_intermediate_dense.bias"] = Layout(
            [model_dim], mesh
        )
        layout_map[r".*_feedforward_output_dense.kernel"] = Layout(
            [model_dim, unshard_dim], mesh
        )
        layout_map[r".*_feedforward_output_dense.bias"] = Layout(
            [unshard_dim], mesh
        )
        return layout_map   