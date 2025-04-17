import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.layers.PositionEmbedding")
class PositionEmbedding(keras.layers.Layer):
    """A layer which learns a position embedding for inputs sequences.

    This class assumes that in the input tensor, the last dimension corresponds
    to the features, and the dimension before the last corresponds to the
    sequence.

    This layer does not supporting masking, but can be combined with a
    `keras.layers.Embedding` for padding mask support.

    Args:
        sequence_length: The maximum length of the dynamic sequence.
        initializer: The initializer to use for the embedding weights. Defaults
            to `"glorot_uniform"`.
        seq_axis: The axis of the input tensor where we add the embeddings.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.

    Call arguments:
        inputs: The tensor inputs to compute an embedding for, with shape
            `(batch_size, sequence_length, hidden_dim)`. Only the input shape
            will be used, as the position embedding does not depend on the
            input sequence content.
        start_index: An integer or integer tensor. The starting position to
            compute the position embedding from. This is useful during cached
            decoding, where each position is predicted separately in a loop.

    Example:

    Called directly on input.
    >>> layer = keras_hub.layers.PositionEmbedding(sequence_length=10)
    >>> layer(np.zeros((8, 10, 16)))

    Combine with a token embedding.
    ```python
    seq_length = 50
    vocab_size = 5000
    embed_dim = 128
    inputs = keras.Input(shape=(seq_length,))
    token_embeddings = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim
    )(inputs)
    position_embeddings = keras_hub.layers.PositionEmbedding(
        sequence_length=seq_length
    )(token_embeddings)
    outputs = token_embeddings + position_embeddings
    ```

    Reference:
     - [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
    """

    def __init__(
        self,
        sequence_length,
        initializer="glorot_uniform",
        hierarchical_alpha=0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError(
                "`sequence_length` must be an Integer, received `None`."
            )
        self.sequence_length = int(sequence_length)
        self.hierarchical_alpha = hierarchical_alpha
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "hierarchical_alpha": self.hierarchical_alpha,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, inputs_shape):
        feature_size = inputs_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, start_index=0):
        shape = ops.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        # trim to match the length of the input sequence, which might be less
        # than the sequence_length of the layer.
        position_embeddings = ops.convert_to_tensor(self.position_embeddings)
        if sequence_length < self.sequence_length:
            position_embeddings = ops.slice(
                position_embeddings,
                (start_index, 0),
                (sequence_length, feature_length),
            )

        else:
            embeddings = (
                position_embeddings
                - self.hierarchical_alpha * position_embeddings[:1]
            )
            embeddings = embeddings / (1 - self.hierarchical_alpha)
            position_ids = ops.arange(sequence_length, dtype="int32")
            embeddings_x = ops.take(
                embeddings, position_ids // self.sequence_length, axis=0
            )
            embeddings_y = ops.take(
                embeddings, position_ids % self.sequence_length, axis=0
            )
            position_embeddings = (
                self.hierarchical_alpha * embeddings_x
                + (1 - self.hierarchical_alpha) * embeddings_y
            )
        return ops.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        return input_shape
